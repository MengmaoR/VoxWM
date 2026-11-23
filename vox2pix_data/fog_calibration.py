"""Calibrate native distance fog and replay the existing saved v3 camera poses."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from .fog_runtime import read_fog, validate_fog
from .material_relight import probe_material
from .native import HERE, NativeScene, RenderSession, SETUP, _builds
from .protocol import RejectedScene, V3_RGB, V3_VERSION, v3_plan
from .relight import LightInTransition, aim_recorded_view, capture, recorded_scene
from .storage import committed_paths, read_json, read_vocabulary


DISTANCES = (20, 22, 24, 25, 27, 29, 30, 32)
MODES = {"A": (False, .6666667), "B": (True, .8), "C": (True, .6666667)}
SETTINGS = dict(V3_RGB, lighting_floor=0., solid_ambient_floor=.35,
                fog_start=.6666667, voxwm_material_lighting_probe=True,
                voxwm_fog_probe=True, voxwm_exact_draw_range=True)


def select_fog(native, mode):
    enabled, ratio = MODES[mode]
    path = native.session.world / "client_clock.txt.fog_request"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{int(enabled)} {ratio}\n")
    temporary.replace(path)
    while time.monotonic() < native.search_deadline:
        native.step()
        probe = read_fog(native)
        try:
            validate_fog(probe, enabled=enabled, ratio=ratio)
        except ValueError:
            continue
        return probe
    raise TimeoutError(f"Native fog settings did not settle in mode {mode}")


def fog_frame(native, capture_row, mode):
    light = native.check_light_frame(capture_row, 0.)
    material = probe_material(native)
    native.check_material_light_frame(light, .35)
    fog = read_fog(native)
    validate_fog(fog, enabled=MODES[mode][0], ratio=MODES[mode][1], tick=light["render_tick"])
    if material["render_tick"] != fog["render_tick"]:
        raise RejectedScene("Fog and material probes label different native frames")
    return dict(fog, material_lighting=material)


class CalibrationScene(NativeScene):
    """Use the existing native command channel in a disposable marker world."""

    def __init__(self, session, seconds):
        self.session = session
        self.seq = 0
        self.search_deadline = time.monotonic() + seconds


def panel_bbox(width, height, fov_x, fov_y, distance, half_side):
    fx, fy = width / (2 * np.tan(fov_x / 2)), height / (2 * np.tan(fov_y / 2))
    extent_x, extent_y = fx * half_side / distance, fy * half_side / distance
    return [int(np.ceil(width / 2 - extent_x)), int(np.ceil(height / 2 - extent_y)),
            int(np.floor(width / 2 + extent_x)), int(np.floor(height / 2 + extent_y))]


def calibration(craftium_dir, output, renderer, seconds):
    from .fog_report import write_marker

    target = np.array([0., 2., 32.5])
    rows = []
    with RenderSession("procedural", 42, renderer, craftium_dir,
                       mapgen="singlenode", scene_script=HERE / "fog_calibration.lua",
                       native_settings=dict(SETUP, **SETTINGS, voxwm_lighting_probe=True),
                       image_size=(640, 360), fixed_step=True, voxel_observation=False,
                       render_rgb=True) as session:
        native = CalibrationScene(session, seconds)
        for distance in DISTANCES:
            desired_eye = target - [0., 0., distance]
            foot = desired_eye - [0., 1.6, 0.]
            for _ in range(2):
                native.command("pose", foot_xyz=foot.tolist())
                for _ in range(16):
                    _, info = native.step()
                error = desired_eye - np.asarray(info["cam_pos"], dtype=np.float64)
                foot += error
            if np.linalg.norm(error) > .0001:
                raise RejectedScene(f"Calibration camera failed to settle at distance {distance}: {error}")
            images, probes = {}, {}
            reference_pose = None
            for mode in MODES:
                select_fog(native, mode)
                row, rgb, info = native.command("capture")
                probes[mode] = fog_frame(native, row, mode)
                eye = np.asarray(info["cam_pos"], dtype=np.float64)
                direction = np.asarray(info["cam_dir"], dtype=np.float64)
                direction /= np.linalg.norm(direction)
                pose = dict(camera_world_xyz=eye.tolist(), cam_dir_xyz=direction.tolist(),
                            fov_x_rad=float(info["cam_fov_x"]), fov_y_rad=float(info["cam_fov_y"]))
                if reference_pose is None:
                    reference_pose = pose
                if not all(np.allclose(pose[key], reference_pose[key], atol=1e-6, rtol=0)
                           for key in pose):
                    raise RejectedScene("Calibration A/B/C camera mismatch")
                if (row["weather"] != "none" or abs(row["world_time"] - .5) > 1e-5
                        or not np.allclose(direction, [0., 0., 1.], atol=1e-6)):
                    raise RejectedScene("Calibration weather, time or view direction differs")
                images[mode] = rgb.copy()
            measured = float(np.linalg.norm(target - eye))
            if abs(measured - distance) > .0001:
                raise RejectedScene(f"Marker face distance differs from {distance}: {measured}")
            row = dict(distance=distance, measured_distance=measured, target_world_xyz=target.tolist(),
                       name="native red wool center", pose=reference_pose, probes=probes,
                       camera_world_xyz=eye.tolist(), pixel_center=[320, 180],
                       pixel_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, .28),
                       crop_bbox=panel_bbox(640, 360, info["cam_fov_x"], info["cam_fov_y"], measured, 3.8),
                       node_name="mcl_wool:red", param2=0,
                       geometry="7x7 native wool panel, center face at [0,2,32.5] in engine XYZ")
            rows.append(write_marker(output, row, images))
            print(f"Calibration distance {distance}: native A/B/C captured", flush=True)
    return rows


def replay_scene(scene_path, output, config, vocabulary, craftium_dir, seconds):
    from .fog_report import write_existing

    results = []
    with recorded_scene(scene_path, config, vocabulary, craftium_dir, seconds,
                        replay_surveys=False, render_settings=SETTINGS) as (native, meta, scene, restored):
        for index, view in enumerate(meta["views"]):
            aim_recorded_view(native, view)
            while time.monotonic() < native.search_deadline:
                images, probes, lights = {}, {}, []
                try:
                    for mode in MODES:
                        select_fog(native, mode)
                        rgb, light, raw_light = capture(native, meta, view, scene, 0.)
                        row = dict(native_clock=dict(server_tick=light["render_tick"] + 1),
                                   hud_wielditem_hidden=True, wielded_item="")
                        probes[mode] = fog_frame(native, row, mode)
                        images[mode] = rgb
                        lights.append(raw_light)
                except LightInTransition:
                    continue
                if all(np.array_equal(lights[0], values) for values in lights[1:]):
                    break
            else:
                raise TimeoutError("Raw light did not stabilize across fog comparison")
            evidence = dict(node_id_identical=True, raw_param1_identical=True, raw_param2_identical=True,
                            camera_pose_identical=True, white_identical=True,
                            white_data_source="read-only saved White Model and Water Count",
                            restored_saved_cells=restored)
            row = dict(scene_id=meta["scene_id"], kind=meta["kind"], world_seed=meta["world_seed"],
                       view_index=index, pose=view, probes=probes, evidence=evidence)
            results.append(write_existing(output, row, images, scene_path))
        print(f"{meta['kind']}: five unchanged saved poses with native fog A/B/C", flush=True)
    return results


def main():
    from .fog_report import finish_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.scene_timeout <= 0:
        parser.error("Choose a fresh output path and a positive scene timeout")
    manifest = read_json(args.dataset / "manifest.json")
    config, paths = manifest["config"], committed_paths(args.dataset)
    if (manifest["version"] != V3_VERSION or len(paths) != 3 or config["views_per_scene"] != 5
            or paths.keys() != {slot["scene_id"] for slot in v3_plan(config)}):
        parser.error("Replay the existing completed three-scene, fifteen-view v3 dataset")
    craftium = args.craftium_dir.resolve()
    vocabulary = read_vocabulary(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fog-calibration-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        marker_rows = calibration(craftium, stage, config["renderer"], args.scene_timeout)
        rows = []
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(replay_scene, path, stage, config, vocabulary, craftium,
                                args.scene_timeout) for path in paths.values()]
            for job in as_completed(jobs):
                rows.extend(job.result())
        finish_report(stage, marker_rows, sorted(rows, key=lambda r: (r["scene_id"], r["view_index"])),
                      dict(source=str(args.dataset), engine_build=_builds(craftium)[0],
                           game_build=_builds(craftium)[1], modes={key: dict(enable_fog=e, fog_start=s)
                           for key, (e,s) in MODES.items()}, render_settings=SETTINGS,
                           calibration="one disposable air world and one panel, camera translated along its view axis",
                           production_scene_count=0, saved_view_count=15, observation_shape=[48,48,48]))
        stage.rename(args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
