"""User-run Vox2Pix pilot collection and read-only QA."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from voxwm_gt.session import DEFAULT_CRAFTIUM

from .protocol import V3_FINAL_PROFILE, V3_FOG48_PROFILE, V3_FOG_PROFILE, V3_MATERIAL_PROFILE


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def workers(value):
    return value if value == "auto" else positive(value)


def parser():
    root = argparse.ArgumentParser(prog="python -m vox2pix_data", description="Vox2Pix native scene collection")
    commands = root.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="collect or resume native scenes")
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--mode", choices=["pilot-v2", "smoke", "production"], default="pilot-v2",
                         help="production collects the final v3 50K-view plan; smoke preserves the 15-frame protocol")
    collect.add_argument("--lighting-profile",
                         choices=[V3_FINAL_PROFILE, "global-lut", V3_MATERIAL_PROFILE,
                                  V3_FOG_PROFILE, V3_FOG48_PROFILE],
                         default=V3_FINAL_PROFILE,
                         help="v3 default: RGB draw32, absolute Fog20–30; older profiles are deprecated")
    collect.add_argument("--scene-kinds", nargs=3, default=["surface", "cave", "underwater"])
    collect.add_argument("--scenes-per-kind", type=int, choices=[1, 3], default=1)
    collect.add_argument("--scene-timeout", type=positive, default=180, help="v3 seconds available per scene")
    collect.add_argument("--survey-y", type=int, nargs=2, metavar=("MIN", "MAX"),
                         help="v3 world Y range to explore (default: -112 240)")
    collect.add_argument("--scenes", type=positive,
                         help="production default: 10000 scenes; historical pilot-v2: 200")
    collect.add_argument("--views-per-scene", type=int, choices=[5], default=5)
    collect.add_argument("--width", type=int, choices=[640], default=640)
    collect.add_argument("--height", type=int, choices=[360], default=360)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--workers", type=workers,
                         help="positive parallel count or auto; production defaults to available CPUs")
    collect.add_argument("--max-scenes", type=positive,
                         help="production: attempt at most this many missing scene slots in this invocation")
    collect.add_argument("--scene-ids", type=nonnegative, nargs="+",
                         help="production: collect only these existing plan slots")
    collect.add_argument("--worlds-per-task", type=positive, default=8,
                         help="production: maximum native worlds per task; use 1 for a tiny local check")
    collect.add_argument("--task-retries", type=nonnegative, default=2,
                         help="production: extra tasks per failed slot this invocation; 0 means one task only")
    collect.add_argument("--search-round", type=int, default=0, help="first seed round (zero is the original search)")
    collect.add_argument("--search-rounds", type=positive, default=1, help="rounds of eight worlds per unfilled slot")
    collect.add_argument("--parallel-rounds", action="store_true", help="search several seed rounds concurrently per missing slot")
    collect.add_argument("--sites-per-world", type=positive, default=4)
    collect.add_argument("--survey-columns", type=int, choices=[16, 32, 48], default=16,
                         help="side of the footprint of candidate anchor columns")
    collect.add_argument("--land-priority", action="store_true", help="rank native land and biome hints for cave surveys")
    collect.add_argument("--cave-hints", action="store_true", help="survey nearby native large-cave endpoints")
    collect.add_argument("--renderer", choices=["software", "window", "auto"], default="auto")
    collect.add_argument("--craftium-dir", type=Path, default=DEFAULT_CRAFTIUM)
    for name, help_text in (
        ("check", "validate a complete pilot without writing to disk"),
        ("compression-report", "write measured compression and coverage reports"),
        ("validate", "validate a completed v3 native smoke dataset"),
        ("report", "export the v3 native smoke report as offline HTML"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", type=Path, required=True)
    dashboard = commands.add_parser("dashboard", help="watch read-only scene and worker progress")
    dashboard.add_argument("--dataset", type=Path, required=True)
    dashboard.add_argument("--log", type=Path, help="read a specific collector log")
    dashboard.add_argument("--interval", type=positive, default=5, help="refresh interval in seconds")
    dashboard.add_argument("--once", action="store_true", help="print one progress snapshot and exit")
    viewer = commands.add_parser("viewer", help="open a read-only Vox2Pix scene Viewer")
    viewer.add_argument("--dataset", type=Path, required=True)
    viewer.add_argument("--port", type=int, default=8767)
    return root


def collect_smoke(args):
    from .geometry import build_library
    from .native import WORLDS_PER_SLOT, SEED_SPACE
    from .protocol import V3_KINDS, RejectedScene
    from .storage import DatasetWriter
    from .v3_native import capture_v3
    from .v3_sampling import SURVEY_Y

    if set(args.scene_kinds) != set(V3_KINDS):
        raise ValueError("Smoke collection requires surface, cave and underwater")
    bounds = list(args.survey_y or SURVEY_Y)
    if not -30900 <= bounds[0] < bounds[1] <= 30900 or not 48 <= bounds[1] - bounds[0] + 1 <= 512:
        raise ValueError("Survey must span 48–512 cells within the playable Lua map")
    worlds = args.search_rounds * WORLDS_PER_SLOT
    first = args.search_round * WORLDS_PER_SLOT
    if (first + worlds) * 200 > SEED_SPACE:
        raise ValueError("Search rounds exceed the independent world seed space")
    config = {key: getattr(args, key) for key in ("mode", "views_per_scene", "width", "height", "seed", "renderer")}
    config.update(scenes=3 * args.scenes_per_kind, scenes_per_kind=args.scenes_per_kind,
                  scene_kinds=list(V3_KINDS), survey_y=bounds)
    if args.lighting_profile != "global-lut":
        config["lighting_profile"] = args.lighting_profile
    with DatasetWriter(args.dataset, config) as writer:
        missing = [slot for scene_id, slot in writer.slots.items() if scene_id not in writer.completed]
        if missing:
            with TemporaryDirectory(prefix="vox2pix-v3-projector-") as tmp:
                library = build_library(Path(tmp))
                with ProcessPoolExecutor(max_workers=min(args.workers, len(missing)),
                                         mp_context=get_context("spawn")) as pool:
                    pending = {
                        pool.submit(capture_v3, slot, config, args.craftium_dir.resolve(), library,
                                    worlds, args.sites_per_world, args.survey_columns, args.scene_timeout, first): slot
                        for slot in missing
                    }
                    for future in as_completed(pending):
                        try:
                            result = future.result()
                            writer.commit(*result)
                        except RejectedScene as exc:
                            print(f"V3 {pending[future]['kind']}: {exc}", file=sys.stderr, flush=True)
                            continue
                        meta = result[0]
                        print(f"V3 committed {meta['kind']} scene {meta['scene_id']} in "
                              f"{meta['search_seconds']:.1f}s: {meta['search_stats']}", flush=True)
        count = len(writer.completed)
        print(f"V3 committed {count}/{config['scenes']} scenes, {count * config['views_per_scene']} frames.", flush=True)
        return int(count != config["scenes"])


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
                  land_priority=args.land_priority, cave_hints=args.cave_hints)
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
    if args.command == "collect":
        from .production import worker_count
        from voxwm_gt.native import renderer_config

        if args.scenes is None:
            args.scenes = 10000 if args.mode == "production" else 200
        if args.mode == "pilot-v2" and args.scenes != 200:
            options.error("Historical pilot-v2 has exactly 200 scene slots")
        if args.mode != "production" and (args.max_scenes is not None or args.scene_ids is not None):
            options.error("--max-scenes and --scene-ids require --mode production")
        args.renderer = renderer_config(args.renderer)[0]
        args.workers = worker_count(args.workers if args.workers is not None
                                   else "auto" if args.mode == "production" else 1)
    if args.command == "collect" and args.seed < 0:
        options.error("--seed must be nonnegative")
    if args.command == "collect" and args.search_round < 0:
        options.error("--search-round must be nonnegative")
    if args.command == "collect" and args.lighting_profile == V3_MATERIAL_PROFILE and args.mode not in ("smoke", "production"):
        options.error("--lighting-profile material-ambient requires --mode smoke")
    try:
        if args.command == "collect":
            if args.mode == "production":
                from .production import collect_production

                return collect_production(args)
            return collect_smoke(args) if args.mode == "smoke" else collect_pilot(args)
        if args.command == "dashboard":
            from .dashboard import show_dashboard
            return show_dashboard(args.dataset, args.log, args.interval, args.once)
        if args.command == "viewer":
            from .preview import serve
            serve(args.dataset, args.port)
            return 0
        if args.command in ("validate", "report"):
            from .v3_report import report, validate

            summary = (report if args.command == "report" else validate)(args.dataset)
            print(f"{summary['status']} | {summary['capture_status']}")
            if args.command == "report":
                print(args.dataset / "report" / "index.html")
            return 0
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
