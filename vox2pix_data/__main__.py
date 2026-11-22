"""User-run Vox2Pix pilot collection and read-only QA."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from voxwm_gt.session import DEFAULT_CRAFTIUM


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parser():
    root = argparse.ArgumentParser(prog="python -m vox2pix_data", description="Vox2Pix native 1K compression pilot")
    commands = root.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="collect or resume the 1K pilot")
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--scenes", type=int, choices=[200], default=200)
    collect.add_argument("--views-per-scene", type=int, choices=[5], default=5)
    collect.add_argument("--width", type=int, choices=[640], default=640)
    collect.add_argument("--height", type=int, choices=[360], default=360)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--workers", type=positive, default=1)
    collect.add_argument("--search-round", type=int, default=0, help="first seed round (zero is the original search)")
    collect.add_argument("--search-rounds", type=positive, default=1, help="rounds of eight worlds per unfilled slot")
    collect.add_argument("--parallel-rounds", action="store_true", help="search several seed rounds concurrently per missing slot")
    collect.add_argument("--sites-per-world", type=positive, default=4)
    collect.add_argument("--survey-columns", type=int, choices=[16, 32, 48], default=16,
                         help="side of the footprint of candidate anchor columns")
    collect.add_argument("--land-priority", action="store_true", help="rank native land and biome hints for cave surveys")
    collect.add_argument("--renderer", choices=["software"], default="software")
    collect.add_argument("--craftium-dir", type=Path, default=DEFAULT_CRAFTIUM)
    for name, help_text in (
        ("check", "validate a complete pilot without writing to disk"),
        ("compression-report", "write measured compression and coverage reports"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", type=Path, required=True)
    dashboard = commands.add_parser("dashboard", help="watch read-only scene and worker progress")
    dashboard.add_argument("--dataset", type=Path, required=True)
    dashboard.add_argument("--log", type=Path, help="read a specific collector log")
    dashboard.add_argument("--interval", type=positive, default=5, help="refresh interval in seconds")
    dashboard.add_argument("--once", action="store_true", help="print one progress snapshot and exit")
    return root


def collect_pilot(args):
    from .geometry import build_library
    from .native import SEED_SPACE, WORLDS_PER_SLOT, capture_scene
    from .protocol import RejectedScene, plan
    from .storage import DatasetWriter

    config = {key: getattr(args, key) for key in
              ("scenes", "views_per_scene", "width", "height", "seed", "renderer")}
    if (args.search_round + args.search_rounds) * args.scenes * WORLDS_PER_SLOT > SEED_SPACE:
        raise ValueError("Search rounds exceed the independent world seed space")
    search = dict(search_round=args.search_round, search_rounds=args.search_rounds,
                  sites_per_world=args.sites_per_world, survey_columns=args.survey_columns,
                  land_priority=args.land_priority)
    with DatasetWriter(args.dataset, config) as writer:
        committed = writer.completed
        completed = len(committed)
        remaining = [slot for slot in plan(args.seed) if slot["scene_id"] not in committed]
        if not remaining:
            print("All 200 scenes are committed. Run check to validate them.")
            return 0
        print(f"Search {len(remaining)} unfilled slots | {json.dumps(search)} "
              f"| parallel rounds {args.parallel_rounds}", flush=True)
        failures = {}
        with TemporaryDirectory(prefix="vox2pix-projector-") as tmp:
            library = build_library(Path(tmp))
            craftium_dir = args.craftium_dir.resolve()
            if args.parallel_rounds:
                first, last = args.search_round, args.search_round + args.search_rounds
                tasks = ((slot, dict(search, search_round=round_id, search_rounds=1))
                         for round_id in range(first, last) for slot in remaining)
            else:
                tasks = ((slot, search) for slot in remaining)
            # Spawned workers cannot retain the dataset writer's file lock.
            workers = min(args.workers, len(remaining) * (args.search_rounds if args.parallel_rounds else 1))
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                pending = {}

                def submit_next():
                    for slot, budget in tasks:
                        scene_id = slot["scene_id"]
                        if scene_id in committed:
                            continue
                        stop = writer.scenes_dir / f"scene_{scene_id:04d}" if args.parallel_rounds else None
                        future = pool.submit(capture_scene, slot, config, craftium_dir, library,
                                             committed_scene=stop, **budget)
                        pending[future] = slot
                        return

                for _ in range(workers):
                    submit_next()
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        slot = pending.pop(future)
                        scene_id = slot["scene_id"]
                        try:
                            capture = future.result()
                        except RejectedScene as exc:
                            failures[scene_id] = str(exc)
                            label = "Search task for scene" if args.parallel_rounds else "Scene"
                            print(f"{label} {scene_id} {slot['biome']} {slot['height']}: {exc}",
                                  file=sys.stderr, flush=True)
                        else:
                            if capture is not None and scene_id not in committed:
                                writer.commit(*capture)
                                committed.add(scene_id)
                                completed += 1
                                meta = capture[0]
                                print(f"Committed scene {scene_id} ({completed}/200) | {slot['height']} "
                                      f"| seed round {meta['search_round']} | world {meta['world_attempt']} "
                                      f"| site {meta['site_attempt']}", flush=True)
                        submit_next()
        if args.parallel_rounds:
            for slot in remaining:
                scene_id = slot["scene_id"]
                if scene_id not in committed:
                    print(f"Scene {scene_id} {slot['biome']} {slot['height']}: {failures[scene_id]}",
                          file=sys.stderr, flush=True)
        print(f"{completed}/200 scenes committed; {args.scenes - completed} slots remain unfilled.", flush=True)
        return int(completed != args.scenes)


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    if args.command == "collect" and args.seed < 0:
        options.error("--seed must be nonnegative")
    if args.command == "collect" and args.search_round < 0:
        options.error("--search-round must be nonnegative")
    try:
        if args.command == "collect":
            return collect_pilot(args)
        if args.command == "dashboard":
            from .dashboard import show_dashboard
            return show_dashboard(args.dataset, args.log, args.interval, args.once)
        from .qa import check, compression_report
        if args.command == "check":
            print(json.dumps(check(args.dataset), indent=2))
        else:
            compression_report(args.dataset)
            print(args.dataset / "compression_report.md")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Vox2Pix {args.command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
