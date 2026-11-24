"""Replay saved View 3/4/5 with RGB draw128, Fog OFF and absolute Fog20–30."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from html import escape
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .fog_range48 import SETTINGS, frame_probe, select_mode
from .protocol import RejectedScene
from .relight import LightInTransition, aim_recorded_view, capture, recorded_scene, restore_saved_cells
from .storage import committed_paths, read_json, read_vocabulary, write_json


def replay(scene_path, output, config, vocabulary, craftium, seconds, views, draw):
    modes = {"no_fog": (False, draw), "fog20_30": (True, draw)}
    settings = dict(SETTINGS, viewing_range=draw)
    rows = []
    with recorded_scene(scene_path, config, vocabulary, craftium, seconds,
                        replay_surveys=False, render_settings=settings) as (native, meta, scene, _):
        folder = output / meta["kind"]
        folder.mkdir()
        for number in views:
            view = meta["views"][number - 1]
            native.command("pose", relocate=True, replay_hold=True,
                           foot_xyz=view["actual_player_world_xyz"], anchor_kind=meta["anchor_kind"],
                           yaw=view["requested_yaw_deg"], pitch=view["requested_pitch_deg"])
            aim_recorded_view(native, view)
            while time.monotonic() < native.search_deadline:
                if restore_saved_cells(native, meta, scene, vocabulary):
                    for _ in range(8):
                        native.step()
                images, probes, lights = {}, {}, []
                try:
                    for mode in modes:
                        select_mode(native, mode, modes=modes)
                        rgb, light, raw_light = capture(native, meta, view, scene, 0.)
                        probes[mode] = frame_probe(native, mode, light, modes=modes)
                        images[mode] = rgb
                        lights.append(raw_light)
                except LightInTransition:
                    continue
                except RejectedScene as exc:
                    if "voxels differ from the saved scene" in str(exc):
                        continue
                    raise
                if np.array_equal(lights[0], lights[1]):
                    break
            else:
                raise TimeoutError(f"{meta['kind']} View {number}: saved state did not stabilize")
            paths = {}
            for mode, rgb in images.items():
                name = f"view_{number:02d}_{mode}.png"
                Image.fromarray(rgb).save(folder / name)
                paths[mode] = f"{meta['kind']}/{name}"
            row = dict(kind=meta["kind"], view=number, scene_id=meta["scene_id"],
                       world_seed=meta["world_seed"], saved_pose=view, probes=probes, images=paths)
            write_json(folder / f"view_{number:02d}.json", row)
            rows.append(row)
            print(f"{meta['kind']} View {number}: draw{draw}, Fog OFF / 20–30 saved", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewing-range", type=int, default=128)
    parser.add_argument("--views", type=int, nargs="+", default=[3, 4, 5], choices=range(1, 6))
    parser.add_argument("--kinds", nargs="+", default=["surface", "cave", "underwater"],
                        choices=["surface", "cave", "underwater"])
    parser.add_argument("--scene-timeout", type=int, default=240)
    args = parser.parse_args()
    if args.output.exists() or args.viewing_range <= 0 or args.scene_timeout <= 0:
        parser.error("Use a new output directory and positive range/timeout")
    config = read_json(args.dataset / "manifest.json")["config"]
    paths = [path for path in committed_paths(args.dataset).values()
             if read_json(path / "scene.json")["kind"] in args.kinds]
    if len(paths) != len(set(args.kinds)):
        parser.error("The source must contain one saved scene per requested kind")
    args.output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=len(paths), mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(replay, path, args.output, config, read_vocabulary(args.dataset),
                            args.craftium_dir.resolve(), args.scene_timeout,
                            sorted(set(args.views)), args.viewing_range) for path in paths]
        for job in jobs:
            rows.extend(job.result())
    rows.sort(key=lambda row: (args.kinds.index(row["kind"]), row["view"]))
    write_json(args.output / "replay.json", dict(source=str(args.dataset),
               viewing_range=args.viewing_range, fog_start=20, fog_end=30, views=rows))
    html = ["<!doctype html><meta charset='utf-8'><title>RGB draw128</title>",
            "<style>body{font:16px sans-serif;background:#19212b;color:#eee;margin:24px}"
            ".pair{display:flex;gap:12px}figure{margin:0;width:48%}img{width:100%}</style>",
            f"<h1>RGB draw {args.viewing_range} · Fog OFF / 20–30</h1>"]
    for row in rows:
        html.append(f"<h2>{escape(row['kind'])} · View {row['view']}</h2><div class='pair'>")
        for mode, label in (("no_fog", "Fog OFF"), ("fog20_30", "Fog 20–30")):
            src = escape(row["images"][mode])
            html.append(f"<figure><figcaption>{label}</figcaption><a href='{src}'>"
                        f"<img src='{src}'></a></figure>")
        html.append("</div>")
    (args.output / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(args.output / "index.html", flush=True)


if __name__ == "__main__":
    main()
