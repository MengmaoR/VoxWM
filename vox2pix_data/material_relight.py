"""Compare native, global-floor and material-ambient renders at saved v3 cameras."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from .material_report import finish_report, write_comparison
from .protocol import RejectedScene, V3_RGB, V3_VERSION, v3_plan
from .relight import LightInTransition, aim_recorded_view, capture, recorded_scene
from .storage import committed_paths, read_json, read_vocabulary


MODES = {"A": (0., 0.), "B": (.35, 0.), "C": (0., .35)}
RENDER_SETTINGS = dict(V3_RGB, lighting_floor=0., solid_ambient_floor=0.)
MATERIAL_PROBE_SETTINGS = dict(RENDER_SETTINGS, voxwm_material_lighting_probe=True)


def probe_material(native):
    path = native.session.world / "client_clock.txt.material_lighting"
    try:
        tick, gamma, global_floor, ambient, zero, maximum, pending, bound, version = path.read_text().split()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid native material renderer probe: {path.name}") from exc
    fog = (native.session.world / "client_clock.txt.material_fog").read_text().split()
    if len(fog) != 12 or int(fog[0]) != int(tick):
        raise RejectedScene("Native fog and material probes do not label the same RGB tick")
    return dict(render_tick=int(tick), display_gamma=float(gamma), lighting_floor=float(global_floor),
                solid_ambient_floor=float(ambient), lut_zero=int(zero), lut_max=int(maximum),
                pending_meshes=int(pending), shader_bound=bool(int(bound)), version=version,
                fog_rgba=list(map(int, fog[1:5])), fog_distance=list(map(float, fog[5:7])),
                native_eye_post_effect_rgba=list(map(int, fog[7:11])),
                eligible_material_shader_drawn=bool(int(fog[11])))


def set_material(native, global_floor, ambient):
    path = native.session.world / "client_clock.txt.lighting_request"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{global_floor} {ambient}\n")
    temporary.replace(path)


def material_frame(native, meta, view, scene, mode):
    global_floor, ambient = MODES[mode]
    set_material(native, global_floor, ambient)
    while time.monotonic() < native.search_deadline:
        native.step()
        p = probe_material(native)
        if (p["lighting_floor"], p["solid_ambient_floor"], p["pending_meshes"]) == (global_floor, ambient, 0):
            break
    else:
        raise TimeoutError(f"Material renderer did not settle for mode {mode}")
    rgb, light_proof, light = capture(native, meta, view, scene, global_floor)
    p = probe_material(native)
    if (p["lighting_floor"], p["solid_ambient_floor"], p["pending_meshes"]) != (global_floor, ambient, 0):
        raise RejectedScene(f"Renderer changed modes during {mode}")
    if (p["version"] != "voxwm-material-ambient-v1" or not p["shader_bound"] or p["render_tick"] != light_proof["render_tick"]
            or p["display_gamma"] != 3.0 or p["lut_zero"] != light_proof["lut_zero"] or p["lut_max"] != 255):
        raise RejectedScene(f"Unmatched material renderer and RGB evidence in {mode}")
    return rgb, p, light


def triplet(native, meta, view, scene):
    while time.monotonic() < native.search_deadline:
        images, probes, lights = {}, {}, {}
        try:
            for mode in MODES:
                images[mode], probes[mode], lights[mode] = material_frame(native, meta, view, scene, mode)
        except LightInTransition:
            continue
        if all(np.array_equal(lights["A"], light) for light in lights.values()):
            for key in ("fog_rgba", "fog_distance", "native_eye_post_effect_rgba"):
                if probes["C"][key] != probes["A"][key]:
                    raise RejectedScene(f"Native {key} changed between A and C")
            return images, probes, lights["A"]
    raise TimeoutError("Identical raw light did not stabilize across A/B/C")


def legacy_frame(scene_path, config, vocabulary, craftium_dir, seconds):
    with recorded_scene(scene_path, config, vocabulary, craftium_dir, seconds, replay_surveys=False,
                        render_settings=RENDER_SETTINGS) as (native, meta, scene, restored):
        view = meta["views"][4]
        aim_recorded_view(native, view)
        while time.monotonic() < native.search_deadline:
            try:
                return capture(native, meta, view, scene, 0.)
            except LightInTransition:
                continue
    raise TimeoutError("Pre-material native control did not stabilize")


def compare_scene(scene_path, output, config, vocabulary, craftium_dir, legacy_dir, seconds):
    old_rgb, old_probe, old_light = legacy_frame(scene_path, config, vocabulary, legacy_dir, seconds)
    with recorded_scene(scene_path, config, vocabulary, craftium_dir, seconds, replay_surveys=False,
                        render_settings=MATERIAL_PROBE_SETTINGS) as (native, meta, scene, restored):
        results = []
        for index, view in enumerate(meta["views"]):
            aim_recorded_view(native, view)
            images, probes, light = triplet(native, meta, view, scene)
            x, y, z = (int(np.floor(n + 24.5)) for n in view["projection_origin_local_xyz"])
            evidence = dict(
                probes=probes, renderer=probes, camera_pose={k: view[k] for k in
                    ("actual_camera_world_xyz", "cam_dir_xyz", "yaw_deg", "pitch_deg", "fov_x_rad", "fov_y_rad")},
                native_eye_param1=int(light[x, z, y]), raw_light_sha256=hashlib.sha256(light.tobytes()).hexdigest(),
                node_id_identical=True, raw_param1_identical=True, raw_param2_identical=True,
                camera_pose_identical=True, white_data_source="read-only original projection",
                white_identical=True, restored_saved_cells=restored,
            )
            if index == 4:
                delta = np.abs(images["A"].astype(np.int16) - old_rgb)
                evidence["legacy_control"] = dict(probe=old_probe, raw_param1_identical=np.array_equal(old_light, light),
                                                  pixel_identical=not delta.any(), mean_abs=float(delta.mean()),
                                                  p95_abs=float(np.percentile(delta, 95)))
            results.append(write_comparison(output, scene_path, config, vocabulary, meta, index, images, evidence))
    print(f"{meta['kind']}: five exact-pose triplets, restored {restored} saved cells in temporary world", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--legacy-craftium-dir", type=Path, required=True)
    parser.add_argument("--scene-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.scene_timeout <= 0:
        parser.error("Choose a new output and a positive scene timeout")
    data = read_json(args.dataset / "manifest.json")
    config = data["config"]
    paths = committed_paths(args.dataset)
    if (data["version"] != V3_VERSION or len(paths) != 3 or config["views_per_scene"] != 5
            or paths.keys() != {s["scene_id"] for s in v3_plan(config)}):
        parser.error("Use the completed three-scene, fifteen-view v3 smoke dataset")
    vocabulary = read_vocabulary(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".material_relight-", dir=args.output.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
            jobs = {pool.submit(compare_scene, path, stage, config, vocabulary, args.craftium_dir.resolve(),
                                args.legacy_craftium_dir.resolve(), args.scene_timeout): ident
                    for ident, path in paths.items()}
            results, errors = [], []
            for job in as_completed(jobs):
                try:
                    results.extend(job.result())
                except (RejectedScene, OSError, RuntimeError, ValueError) as exc:
                    errors.append(f"Scene {jobs[job]:04d}: {exc}")
        if errors:
            parser.exit(1, "\n".join(errors) + "\n")
        finish_report(stage, sorted(results, key=lambda row: (row["scene_id"], row["view_index"])),
                      dict(source=str(args.dataset), modes=MODES, gamma=3.0, native_view_count=15,
                           projection="the saved v3 White Model and Water Count", diagnostics_only=True))
        stage.rename(args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
