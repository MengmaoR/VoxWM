"""Compare native 30/48-block draw ranges with independent 20–30-block fog."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np

from .fog_calibration import CalibrationScene, panel_bbox
from .fog_runtime import read_fog, validate_fog_absolute
from .material_relight import probe_material
from .native import HERE, RenderSession, SETUP, _builds
from .protocol import RejectedScene, V3_FOG48_RGB, V3_FOG48_VERSION, V3_VERSION, v3_plan
from .relight import LightInTransition, aim_recorded_view, capture, recorded_scene, restore_saved_cells
from .storage import committed_paths, read_json, read_vocabulary, write_json


DISTANCES = (12, 18, 20, 22, 24, 27, 29, 30, 32, 36, 40, 44, 48, 52)
MODES = {"A": (False, 48), "B": (True, 30), "C": (True, 48)}
SETTINGS = dict(V3_FOG48_RGB,
                voxwm_material_lighting_probe=True, voxwm_fog_probe=True, voxwm_exact_draw_range=True)
PANEL_FACE = np.array([0., 2., 32.5])
MARKER_NODE = [0, 2, 33]
MARKER_BLOCK = [0, 0, 2]


def select_mode(native, mode):
    enabled, viewing_range = MODES[mode]
    path = native.session.world / "client_clock.txt.fog_request"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{int(enabled)} 20 30 {viewing_range}\n")
    temporary.replace(path)
    first_valid_tick = None
    while time.monotonic() < native.search_deadline:
        native.step()
        probe = read_fog(native)
        try:
            validate_fog_absolute(probe, enabled=enabled, viewing_range=viewing_range)
        except ValueError:
            first_valid_tick = None
            continue
        if first_valid_tick is None:
            first_valid_tick = probe["render_tick"]
        # Game::updateFrame refreshes the native draw list every 0.2 s.
        # Twelve fixed24 ticks allow two complete refresh periods after
        # the new range first reaches the actual renderer.
        if probe["render_tick"] - first_valid_tick >= 12:
            return
    raise TimeoutError(f"Independent fog range did not settle in mode {mode}: {probe}")


def frame_probe(native, mode, light):
    material = probe_material(native)
    native.check_material_light_frame(light, .35)
    probe = read_fog(native)
    validate_fog_absolute(probe, enabled=MODES[mode][0], viewing_range=MODES[mode][1],
                          tick=light["render_tick"])
    if probe["render_tick"] != material["render_tick"]:
        raise RejectedScene("Native fog and material probes belong to different RGB frames")
    return dict(probe, material_lighting=material)


def calibration(craftium, output, renderer, seconds):
    from .fog48_report import write_marker

    results = []
    with RenderSession("procedural", 42, renderer, craftium,
                       mapgen="singlenode", scene_script=HERE / "fog_calibration.lua",
                       native_settings=dict(SETUP, **SETTINGS, voxwm_lighting_probe=True),
                       image_size=(640, 360), fixed_step=True, voxel_observation=False,
                       render_rgb=True) as session:
        native = CalibrationScene(session, seconds)
        for distance in DISTANCES:
            desired_eye = PANEL_FACE - [0., 0., distance]
            foot = desired_eye - [0., 1.6, 0.]
            for _ in range(2):
                native.command("pose", foot_xyz=foot.tolist())
                for _ in range(16):
                    _, info = native.step()
                correction = desired_eye - np.asarray(info["cam_pos"], dtype=np.float64)
                foot += correction
            if np.linalg.norm(correction) > .0001:
                raise RejectedScene(f"Marker camera did not settle at {distance}: {correction}")
            images, probes, reference_pose = {}, {}, None
            for mode in MODES:
                select_mode(native, mode)
                # A and C must really submit the marker's native mesh before
                # its appearance can demonstrate the independent fog endpoint.
                if mode in ("A", "C") and distance <= 48:
                    while MARKER_BLOCK not in read_fog(native)["submitted_mapblocks_xyz"]:
                        if time.monotonic() >= native.search_deadline:
                            raise TimeoutError(f"Native renderer never submitted marker MapBlock at {distance} in {mode}")
                        native.step()
                row, rgb, info = native.command("capture")
                probes[mode] = frame_probe(native, mode, native.check_light_frame(row, 0.))
                if mode in ("A", "C") and distance <= 48:
                    if MARKER_BLOCK not in probes[mode]["submitted_mapblocks_xyz"]:
                        raise RejectedScene(f"Captured frame omitted the marker MapBlock at distance {distance}")
                eye = np.asarray(info["cam_pos"], dtype=np.float64)
                direction = np.asarray(info["cam_dir"], dtype=np.float64)
                direction /= np.linalg.norm(direction)
                pose = dict(camera_world_xyz=eye.tolist(), cam_dir_xyz=direction.tolist(),
                            fov_x_rad=float(info["cam_fov_x"]), fov_y_rad=float(info["cam_fov_y"]))
                if reference_pose is None:
                    reference_pose = pose
                if not all(np.allclose(pose[key], reference_pose[key], atol=1e-6, rtol=0) for key in pose):
                    raise RejectedScene("Range calibration A/B/C camera mismatch")
                if (row["weather"] != "none" or abs(row["world_time"] - .5) > 1e-5
                        or not np.allclose(direction, [0., 0., 1.], atol=1e-6)):
                    raise RejectedScene("Marker weather, time or direction differs")
                images[mode] = rgb.copy()
            measured = float(np.linalg.norm(PANEL_FACE - eye))
            if abs(measured - distance) > .0001:
                raise RejectedScene(f"Expected marker face distance {distance}, measured {measured}")
            row = dict(distance=distance, measured_distance=measured, target_world_xyz=PANEL_FACE.tolist(),
                       marker_node_xyz=MARKER_NODE, marker_mapblock_xyz=MARKER_BLOCK,
                       name="native red wool center", node_name="mcl_wool:red", param2=0,
                       pose=reference_pose, camera_world_xyz=eye.tolist(), probes=probes,
                       pixel_center=[320, 180],
                       pixel_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, .28),
                       crop_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, 3.8),
                       geometry="7x7 native wool panel, face [0,2,32.5] in engine XYZ")
            results.append(write_marker(output, row, images))
            print(f"Range48 distance {distance}: actual native A/B/C captured", flush=True)
    return results


def replay_scene(scene_path, output, config, vocabulary, craftium, seconds):
    from .fog48_report import write_existing

    results = []
    with recorded_scene(scene_path, config, vocabulary, craftium, seconds,
                        replay_surveys=False, render_settings=SETTINGS) as (native, meta, scene, restored):
        for index, view in enumerate(meta["views"]):
            native.command("pose", relocate=True, replay_hold=True,
                           foot_xyz=view["actual_player_world_xyz"], anchor_kind=meta["anchor_kind"],
                           yaw=view["requested_yaw_deg"], pitch=view["requested_pitch_deg"])
            aim_recorded_view(native, view)
            restorations = []
            while time.monotonic() < native.search_deadline:
                repaired = restore_saved_cells(native, meta, scene, vocabulary)
                if repaired:
                    restorations.append(repaired)
                    for _ in range(8):
                        native.step()
                images, probes, lights = {}, {}, []
                try:
                    for mode in ("B", "C"):
                        select_mode(native, mode)
                        rgb, light, raw_light = capture(native, meta, view, scene, 0.)
                        probes[mode] = frame_probe(native, mode, light)
                        images[mode] = rgb
                        lights.append(raw_light)
                except LightInTransition:
                    continue
                except RejectedScene as exc:
                    if "voxels differ from the saved scene" not in str(exc):
                        raise
                    print(f"{meta['kind']} view {index + 1}: {exc}; restoring disposable world", flush=True)
                    continue
                if np.array_equal(lights[0], lights[1]):
                    break
            else:
                raise TimeoutError("Raw saved voxel/light state did not stabilize for the B/C pair")
            evidence = dict(node_id_identical=True, raw_param1_identical=True, raw_param2_identical=True,
                            camera_pose_identical=True, white_identical=True,
                            white_data_source="read-only original White Model, face IDs and Water Count",
                            replay_camera_control="saved native player pose held in disposable RGB replay",
                            replay_held_player_xyz=view["actual_player_world_xyz"],
                            restored_saved_cells=restored,
                            saved_cell_restorations_before_pair=restorations)
            row = dict(scene_id=meta["scene_id"], kind=meta["kind"], world_seed=meta["world_seed"],
                       view_index=index, pose=view, probes=probes, evidence=evidence)
            results.append(write_existing(output, row, images, scene_path))
        print(f"{meta['kind']}: five fixed-pose native range30/range48 comparisons", flush=True)
    return results


def main():
    from .fog48_report import finish_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.scene_timeout <= 0:
        parser.error("Use a new output directory and a positive scene timeout")
    source = read_json(args.dataset / "manifest.json")
    config, paths = source["config"], committed_paths(args.dataset)
    if (source["version"] != V3_VERSION or len(paths) != 3 or config["views_per_scene"] != 5
            or paths.keys() != {slot["scene_id"] for slot in v3_plan(config)}):
        parser.error("Use the original completed three-scene, fifteen-view smoke dataset")
    craftium, vocabulary = args.craftium_dir.resolve(), read_vocabulary(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fog-range48-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        markers, rows = [], []
        try:
            markers = calibration(craftium, stage, config["renderer"], args.scene_timeout)
            write_json(stage / "calibration_rows.json", markers)
            with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
                jobs = [pool.submit(replay_scene, path, stage, config, vocabulary, craftium,
                                    args.scene_timeout) for path in paths.values()]
                for job in as_completed(jobs):
                    rows.extend(job.result())
            finish_report(stage, markers, sorted(rows, key=lambda r: (r["scene_id"], r["view_index"])),
                          dict(source=str(args.dataset), engine_build=_builds(craftium)[0],
                               game_build=_builds(craftium)[1], source_commit=subprocess.check_output(
                                   ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip(),
                               modes={key: dict(enable_fog=enabled, viewing_range=draw,
                                                fog_start_distance=20, fog_end_distance=30)
                                      for key, (enabled, draw) in MODES.items()},
                               render_settings=SETTINGS, renderer=config["renderer"],
                               protocol_version=V3_FOG48_VERSION,
                               calibration="one disposable air world, camera translated toward one fixed panel",
                               production_scene_count=0, saved_view_count=15,
                               voxel_observation_shape=[48, 48, 48], voxel_observation_half_extent=24,
                               rgb_viewing_range=48, fog_start_distance=20, fog_end_distance=30))
        except Exception as exc:
            failure = args.output.with_name(args.output.name + f".failed-{time.time_ns()}")
            write_json(stage / "failure.json", dict(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                                                   completed_markers=markers, completed_scene_views=rows))
            stage.rename(failure)
            raise RuntimeError(f"Native range48 test failed; preserved outputs at {failure}") from exc
        stage.rename(args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
