"""Verify final RGB draw32, independent fog20–30 and the original fifteen views."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np

from .fog_calibration import CalibrationScene, panel_bbox
from .fog_range48 import frame_probe, select_mode
from .fog_runtime import read_fog
from .native import HERE, RenderSession, SETUP, _builds
from .protocol import RejectedScene, V3_FINAL_RGB, V3_FINAL_VERSION
from .relight import LightInTransition, aim_recorded_view, capture, recorded_scene, restore_saved_cells
from .storage import committed_paths, read_json, read_vocabulary, write_json


DISTANCES = (10, 18, 20, 22, 24, 25, 27, 29, 30, 31, 32, 34)
MODES = {"A": (False, 32), "C": (True, 32)}
SETTINGS = dict(V3_FINAL_RGB, voxwm_material_lighting_probe=True, voxwm_fog_probe=True)
CAMERA_EYE = np.array([0., 2., .5])


def calibration(craftium, output, renderer, seconds):
    from .final_report import write_marker

    results, fixed_pose = [], None
    with RenderSession("procedural", 42, renderer, craftium,
                       mapgen="singlenode", scene_script=HERE / "final_calibration.lua",
                       native_settings=dict(SETUP, **SETTINGS, voxwm_lighting_probe=True),
                       image_size=(640, 360), fixed_step=True, voxel_observation=False,
                       render_rgb=True) as session:
        native = CalibrationScene(session, seconds)
        # Calibrate the nominal native eye once. Every marker then uses this
        # unchanged observer, rather than translating the camera between tests.
        foot = CAMERA_EYE - [0., 1.6, 0.]
        for _ in range(2):
            native.command("pose", foot_xyz=foot.tolist())
            for _ in range(16):
                _, info = native.step()
            correction = CAMERA_EYE - np.asarray(info["cam_pos"], dtype=np.float64)
            foot += correction
        if np.linalg.norm(correction) > .0001:
            raise RejectedScene(f"Fixed calibration camera did not settle: {correction}")
        for distance in DISTANCES:
            native.command("marker", distance=distance)
            for _ in range(24):
                native.step()
            target = np.array([0., 2., distance + .5])
            block = [0, 0, (distance + 1) // 16]
            images, probes = {}, {}
            for mode in MODES:
                select_mode(native, mode, modes=MODES)
                native.wait_lighting(0., ambient=.35)
                if distance <= 32:
                    while block not in read_fog(native)["submitted_mapblocks_xyz"]:
                        if time.monotonic() >= native.search_deadline:
                            raise TimeoutError(f"Marker MapBlock was not submitted at {distance} in {mode}")
                        native.step()
                row, rgb, info = native.command("capture")
                probes[mode] = frame_probe(native, mode, native.check_light_frame(row, 0.), modes=MODES)
                if distance <= 32 and block not in probes[mode]["submitted_mapblocks_xyz"]:
                    raise RejectedScene(f"Captured frame omitted marker geometry at {distance}")
                eye = np.asarray(info["cam_pos"], dtype=np.float64)
                direction = np.asarray(info["cam_dir"], dtype=np.float64)
                direction /= np.linalg.norm(direction)
                pose = dict(camera_world_xyz=eye.tolist(), cam_dir_xyz=direction.tolist(),
                            fov_x_rad=float(info["cam_fov_x"]), fov_y_rad=float(info["cam_fov_y"]))
                if fixed_pose is None:
                    fixed_pose = pose
                if not all(np.allclose(pose[key], fixed_pose[key], atol=1e-6, rtol=0) for key in pose):
                    raise RejectedScene("The fixed calibration camera moved between markers or modes")
                if (row["weather"] != "none" or abs(row["world_time"] - .5) > 1e-5
                        or not np.allclose(direction, [0., 0., 1.], atol=1e-6)):
                    raise RejectedScene("Calibration weather, time or direction differs")
                images[mode] = rgb.copy()
            measured = float(np.linalg.norm(target - eye))
            if abs(measured - distance) > .0001:
                raise RejectedScene(f"Marker distance {distance} measured as {measured}")
            row = dict(distance=distance, measured_distance=measured, target_world_xyz=target.tolist(),
                       marker_node_xyz=[0, 2, distance + 1], marker_mapblock_xyz=block,
                       name="native red wool center", node_name="mcl_wool:red", param2=0,
                       pose=fixed_pose, camera_world_xyz=eye.tolist(), probes=probes,
                       pixel_center=[320, 180],
                       pixel_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, .28),
                       crop_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, 3.8),
                       geometry="7x7 native wool panel; one distance at a time, fixed camera [0,2,0.5]")
            results.append(write_marker(output, row, images))
            print(f"Final32 marker {distance}: native OFF / Fog20–30 captured", flush=True)
    return results


def replay_scene(scene_path, output, config, vocabulary, craftium, seconds):
    from .final_report import write_existing

    rows = []
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
                try:
                    select_mode(native, "C", modes=MODES)
                    rgb, light, _ = capture(native, meta, view, scene, 0.)
                    probe = frame_probe(native, "C", light, modes=MODES)
                except LightInTransition:
                    continue
                except RejectedScene as exc:
                    if "voxels differ from the saved scene" not in str(exc):
                        raise
                    print(f"{meta['kind']} View {index + 1}: restoring saved cells", flush=True)
                    continue
                break
            else:
                raise TimeoutError(f"{meta['kind']} View {index + 1} did not stabilize")
            evidence = dict(node_id_identical=True, raw_param2_identical=True, camera_pose_identical=True,
                            saved_voxel_shape=list(scene[0].shape),
                            replay_camera_control="saved native pose held only in disposable RGB replay",
                            replay_held_player_xyz=view["actual_player_world_xyz"],
                            restored_saved_cells=restored, saved_cell_restorations=restorations)
            row = dict(scene_id=meta["scene_id"], kind=meta["kind"], world_seed=meta["world_seed"],
                       view_index=index, pose=view, probes={"C": probe}, evidence=evidence)
            rows.append(write_existing(output, row, {"C": rgb}, scene_path))
            print(f"Final32 {meta['kind']} View {index + 1}: RGB and saved White Model exported", flush=True)
    return rows


def main():
    from .final_report import finish_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/vox2pix_final_rendering_32"))
    parser.add_argument("--scene-timeout", type=int, default=240)
    args = parser.parse_args()
    if args.output.exists() or args.scene_timeout <= 0:
        parser.error("Use a new output directory and a positive scene timeout")
    source = read_json(args.dataset / "manifest.json")
    config, paths = source["config"], committed_paths(args.dataset)
    kinds = {read_json(path / "scene.json")["kind"] for path in paths.values()}
    if (len(paths) != 3 or kinds != {"surface", "cave", "underwater"}
            or config["views_per_scene"] != 5 or (config["width"], config["height"]) != (640, 360)):
        parser.error("Use the original three-scene, fifteen-view 640x360 smoke dataset")
    craftium, vocabulary = args.craftium_dir.resolve(), read_vocabulary(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".final-rendering32-", dir=args.output.parent) as temporary:
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
            manifest = dict(source=str(args.dataset), source_protocol_version=source["version"],
                            engine_build=_builds(craftium)[0], game_build=_builds(craftium)[1],
                            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                  cwd=HERE.parent, text=True).strip(),
                            modes={key: dict(enable_fog=enabled, viewing_range=draw,
                                             fog_start_distance=20, fog_end_distance=30)
                                   for key, (enabled, draw) in MODES.items()},
                            render_settings=SETTINGS, renderer=config["renderer"],
                            protocol_version=V3_FINAL_VERSION,
                            calibration="fixed camera, sequential native panels at twelve exact distances; no occlusion",
                            unit_verified=False, visual_verified=False,
                            production_scene_count=0, saved_view_count=15,
                            voxel_observation_shape=[48, 48, 48], voxel_observation_half_extent=24,
                            rgb_viewing_range=32, fog_start_distance=20, fog_end_distance=30)
            summary = finish_report(stage, markers, sorted(rows, key=lambda r: (r["scene_id"], r["view_index"])), manifest)
            if summary["status"] == "FAIL":
                raise ValueError("Native calibration or saved-array regression failed; inspect summary.json")
        except Exception as exc:
            failure = args.output.with_name(args.output.name + f".failed-{time.time_ns()}")
            write_json(stage / "failure.json", dict(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                                                   completed_markers=markers, completed_scene_views=rows))
            stage.rename(failure)
            raise RuntimeError(f"Final rendering test failed; preserved output at {failure}") from exc
        stage.rename(args.output)
    print(args.output / "report/index.html")


if __name__ == "__main__":
    main()
