"""Replay the saved v3 cameras in native worlds and publish RGB comparisons only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from html import escape
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
from PIL import Image
import zarr

from .native import HERE, REUSED_LUA, SETUP, NativeScene, RenderSession, _builds
from .protocol import RejectedScene, V3_RGB, V3_VERSION, v3_plan
from .storage import RAW_FIELDS, committed_paths, read_json, read_vocabulary, write_json
from .v3_report import brightness


def capture(native, meta, view, scene, floor):
    native.wait_lighting(floor)
    origin, kind = meta["region_origin_xyz"], meta["anchor_kind"]
    before, prior, _, _ = native.geometry_snapshot(origin, kind)
    light = native.region_light
    after, row, rgb, info = native.geometry_snapshot(origin, kind)
    proof = native.check_light_frame(row, floor)
    if proof["render_tick"] != prior["native_clock"]["server_tick"]:
        raise RejectedScene("RGB is not bracketed by successive native snapshots")
    for state in (before, after):
        changed = np.count_nonzero((state[0] != scene[0]) | (state[1] != scene[1]))
        if changed:
            raise RejectedScene(f"{changed} voxels differ from the saved scene")
    for measured, key in (("cam_pos", "actual_camera_world_xyz"), ("cam_dir", "cam_dir_xyz"),
                          ("cam_fov_x", "fov_x_rad"), ("cam_fov_y", "fov_y_rad")):
        actual = np.asarray(info[measured], dtype=np.float64)
        if measured == "cam_dir":
            actual = actual / np.linalg.norm(actual)
        if not np.allclose(actual, view[key], rtol=0, atol=.0001 if measured == "cam_pos" else 1e-6):
            raise RejectedScene(f"Actual camera {measured} differs from the saved view: {actual}")
    if not np.array_equal(light, native.region_light):
        raise RejectedScene("Native light changed across the captured frame")
    return rgb.copy(), proof, light


def replay(scene_path, output, config, vocabulary, craftium_dir, floor, seconds):
    meta = read_json(scene_path / "scene.json")
    if _builds(craftium_dir) != (meta["engine_build"], meta["game_build"]):
        raise ValueError("The saved scene requires its original Craftium and VoxeLibre revisions")
    group = zarr.open_group(str(scene_path / "voxels.zarr"), mode="r")
    scene = tuple(group[key][:] for key in RAW_FIELDS)
    path = output / scene_path.name
    path.mkdir()
    size = config["width"], config["height"]
    settings = dict(SETUP, **V3_RGB, vox2pix_v3=True, voxwm_lighting_probe=True, voxwm_skip_rgb=True)
    records = []
    with RenderSession("procedural", meta["world_seed"], config["renderer"], craftium_dir,
                       scene_script=HERE / "scene.lua",
                       scene_assets=(REUSED_LUA / "biome_search.lua", REUSED_LUA / "scene_geometry.lua"),
                       native_settings=settings, image_size=size, fixed_step=True,
                       voxel_observation=False, render_rgb=True) as session:
        native = NativeScene(session)
        if native.vocabulary != vocabulary:
            raise ValueError("The engine's registered node vocabulary differs from the source")
        native.search_deadline = time.monotonic() + seconds
        origin = meta["region_origin_xyz"]
        response, _, _ = native.command("emerge_capture", low_xyz=origin, high_xyz=[n + 47 for n in origin])
        if response["status"] != "capture_emerged":
            raise RejectedScene("The saved crop did not emerge")
        descent = meta.get("underwater_descent")
        spawn = descent["spawn_player_xyz"] if descent else meta["foot_anchor_xyz"]
        native.command("pose", relocate=True, foot_xyz=spawn, anchor_kind=meta["anchor_kind"], yaw=0., pitch=0.)
        if descent:
            native.sink_to_floor(spawn)
        for _ in range(72):
            native.step()
        for index, view in enumerate(meta["views"]):
            native.request_floor(V3_RGB["lighting_floor"])
            native.command("pose", relocate=False, yaw=view["requested_yaw_deg"], pitch=view["requested_pitch_deg"])
            for _ in range(12):
                native.step()
            control, _, light = capture(native, meta, view, scene, V3_RGB["lighting_floor"])
            native.request_floor(floor)
            rgb, proof, proposed_light = capture(native, meta, view, scene, floor)
            if not np.array_equal(light, proposed_light):
                raise RejectedScene("Native light changed between comparison frames")
            with Image.open(scene_path / "rgb" / f"{index:03d}.png") as original:
                original_rgb = np.asarray(original, dtype=np.int16)
            diff = np.mean(np.abs(control.astype(np.int16) - original_rgb))
            Image.fromarray(control).save(path / f"{index:03d}_control.png")
            Image.fromarray(rgb).save(path / f"{index:03d}_relit.png")
            records.append(dict(index=index, control=brightness(control), relit=brightness(rgb),
                                renderer_lighting=proof, control_original_mean_absolute_rgb_difference=float(diff)))
    print(f"{meta['kind']}: {len(records)} RGB views replayed with identical voxel state and camera", flush=True)
    return dict(scene_id=meta["scene_id"], kind=meta["kind"], world_seed=meta["world_seed"], views=records)


def report(root, rows, floor):
    sections = []
    for row in rows:
        path = f"scene_{row['scene_id']:04d}"
        panels = []
        for view in row["views"]:
            panels.append(f"""<h3>View {view['index'] + 1}</h3><div class="pair">
<figure><img src="{path}/{view['index']:03d}_control.png"><figcaption>0.35 · mean luma
{view['control']['mean_luma']:.2f}</figcaption></figure>
<figure><img src="{path}/{view['index']:03d}_relit.png"><figcaption>{floor:.2f} · mean luma
{view['relit']['mean_luma']:.2f}</figcaption></figure></div>""")
        sections.append(f"<section><h2>{escape(row['kind'].title())} · seed {row['world_seed']}</h2>"
                        + "".join(panels) + "</section>")
    (root / "index.html").write_text("""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vox2Pix lighting comparison</title><style>
body{margin:24px auto;padding:0 16px;max-width:1320px;color:#eef3f8;background:#111723;font:16px system-ui}
section{margin:32px 0}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
figure{margin:0}img{display:block;width:100%}figcaption{padding:5px 0;color:#bfd1e0}
</style><h1>Vox2Pix native RGB comparison</h1>
<p>Gamma 3.0. The paired RGB frames share a world, voxel state, native light and camera pose.</p>"""
                                     + "".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=.20)
    parser.add_argument("--scene-timeout", type=int, default=90)
    args = parser.parse_args()
    if not 0 <= args.floor <= 1 or args.scene_timeout <= 0:
        parser.error("floor must be in [0,1] and timeout must be positive")
    if args.output.exists():
        parser.error("Choose a new RGB comparison output directory")
    manifest = read_json(args.dataset / "manifest.json")
    config = manifest["config"]
    paths = committed_paths(args.dataset)
    if manifest["version"] != V3_VERSION or paths.keys() != {s["scene_id"] for s in v3_plan(config)}:
        parser.error("Expected a complete v3 native dataset")
    vocab = read_vocabulary(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".relight-", dir=args.output.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        with ProcessPoolExecutor(max_workers=min(3, len(paths)), mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(replay, path, stage, config, vocab, args.craftium_dir.resolve(),
                                args.floor, args.scene_timeout) for path in paths.values()]
            rows = [job.result() for job in jobs]
        write_json(stage / "summary.json", dict(floor=args.floor, source=str(args.dataset), scenes=rows))
        report(stage, rows, args.floor)
        stage.rename(args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
