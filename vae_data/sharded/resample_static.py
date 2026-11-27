"""Capture one native Static replacement per planned overlap, using fresh seeds."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from .generate import prepare_static
from .native import Native
from .protocol import Vocabulary, dump
from .static_height import validate_static
from .validate import validate_native


def capture(target, seeds, output, craftium, renderer, site_limit, world_seconds):
    original = target["original_static"]
    ident = original["id"].replace(":", "_")
    folder = output / ident
    group, stratum = original["group"], original["stratum"]
    expected_transition = original["biome_transition"]
    with (output / f"{ident}.attempts.jsonl").open("a") as attempts:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            nearby = deque()
            started = time.monotonic()
            try:
                with Native(seed, Vocabulary(), started + world_seconds, craftium, renderer) as world:
                    seen_origins = set()
                    for _ in range(site_limit):
                        origin = nearby.popleft() if nearby else (rng.integers(-16000, 16001, 2) // 16 * 16).tolist()
                        if tuple(origin) in seen_origins:
                            continue
                        seen_origins.add(tuple(origin))
                        attempt = {"world_seed": seed, "search_origin": origin}
                        try:
                            frame = prepare_static(world, {"biome": group, "stratum": stratum}, origin, rng, attempt)
                            validate_native(frame)
                            validate_static(frame, world.v)
                            if frame.meta["biome_transition"] != expected_transition:
                                raise ValueError("The captured biome transition does not match the original")
                        except ValueError as exc:
                            site = attempt.get("biome_search", {}).get("site")
                            offsets = [(16, 0), (-16, 0), (0, 16), (0, -16),
                                       (32, 0), (-32, 0), (0, 32), (0, -32)]
                            if site and not nearby:
                                nearby.extend([site[0] + dx, site[1] + dz] for dx, dz in offsets
                                              if (site[0] + dx, site[1] + dz) not in seen_origins)
                            attempt["rejected"] = str(exc)
                            attempts.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                            attempts.flush()
                            continue
                        with TemporaryDirectory(prefix=".static-", dir=output) as temp:
                            stage = Path(temp) / "snapshot"
                            stage.mkdir()
                            np.savez_compressed(stage / "snapshot.npz", tokens=frame.tokens, light=frame.light)
                            world.v.save(stage / "vocabulary.json")
                            frame.meta.update(category="Static", original_static_id=original["id"],
                                              wall_seconds=time.monotonic() - started)
                            dump(stage / "metadata.json", frame.meta)
                            stage.rename(folder)
                        attempt["accepted"] = True
                        attempts.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                        return {"id": original["id"], "world_seed": seed, "biome": group,
                                "stratum": stratum, "elapsed": frame.meta["wall_seconds"]}
            except (RuntimeError, OSError, ValueError) as exc:
                attempts.write(json.dumps({"world_seed": seed, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                attempts.flush()
    raise RuntimeError(f"No matching {group}/{stratum} snapshot in {len(seeds)} fresh worlds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--craftium-dir", type=Path, default=Path("third_party/craftium"))
    parser.add_argument("--renderer", choices=("auto", "window", "software"), default="auto")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worlds-per-scene", type=int, default=8)
    parser.add_argument("--sites-per-world", type=int, default=8)
    parser.add_argument("--world-seconds", type=int, default=75)
    args = parser.parse_args()
    if min(args.workers, args.worlds_per_scene, args.sites_per_world, args.world_seconds) < 1:
        parser.error("Worker, world, site and time limits must be positive")
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    used = set(plan["used_world_seeds"])
    pending = []
    for target in plan["targets"]:
        path = args.output / target["original_static"]["id"].replace(":", "_") / "metadata.json"
        if path.is_file():
            used.add(json.loads(path.read_text())["world_seed"])
        else:
            pending.append(target)
    for log in args.output.glob("*.attempts.jsonl"):
        used.update(json.loads(line)["world_seed"] for line in log.read_text().splitlines())

    rng = np.random.default_rng(args.seed)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for target in pending:
            seeds = []
            while len(seeds) < args.worlds_per_scene:
                seed = int(rng.integers(1, 2**31 - 1))
                if seed not in used:
                    used.add(seed)
                    seeds.append(seed)
            future = pool.submit(capture, target, seeds, args.output.resolve(), args.craftium_dir.resolve(),
                                 args.renderer, args.sites_per_world, args.world_seconds)
            jobs[future] = target["original_static"]["id"]
        failures = {}
        for future in as_completed(jobs):
            ident = jobs[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except (ValueError, RuntimeError, OSError) as exc:
                failures[ident] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"id": ident, "error": failures[ident]}), flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
