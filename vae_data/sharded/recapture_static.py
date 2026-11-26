"""Recapture planned 48³ Static windows from fresh native worlds at their exact seeds and origins."""

import argparse
import json
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import combinations
from pathlib import Path

import numpy as np

from vox2pix_data.geometry import is_water_liquid
from vox2pix_data.native import (
    HERE,
    REUSED_LUA,
    SETUP,
    NativeScene,
    RenderSession,
    _builds,
    decode_region,
)
from vox2pix_data.v3_native import site_candidates, survey
from voxel_vae.outer_chunk_rank import blocks

from .protocol import BIOMES, biome_group


def xyz(enu):
    return [enu[0], enu[2], enu[1]]


def height_type(center_y, surface_y):
    depth = surface_y - center_y
    if depth >= 25:
        return "deep_cave"
    if depth >= 8:
        return "shallow_underground"
    if depth >= -8:
        return "surface"
    return "elevated_complex"


def require_status(reply, expected):
    if reply.get("status") != expected:
        raise RuntimeError(f"Expected {expected}, got {reply.get('status')}: {reply.get('reason', '')}")


class StaticSession(RenderSession):
    failure_log = None

    def __exit__(self, exc_type, exc, traceback):
        if exc and self.failure_log and self.env is not None:
            run_dir = Path(self.env.unwrapped.mt.run_dir)
            sections = []
            for name in ("stderr.txt", "stdout.txt", "debug.txt"):
                path = run_dir / name
                if path.is_file():
                    sections.append(name + "\n" + path.read_text(errors="replace")[-16000:])
            self.failure_log.write_text(str(exc) + "\n" + "\n\n".join(sections))
        return super().__exit__(exc_type, exc, traceback)


def prime_surveys(native, source, trace):
    """Repeat the source's mapgen surveys before placing the observer."""
    row = trace["scenes"][source["scene"]]
    rng = np.random.default_rng(row["world_seed"])
    hints = []
    sites = []
    kind = row["kind"]
    y_bounds = trace["config"]["survey_y"]
    for definition in native.definitions.values():
        definition["water"] = is_water_liquid(definition)
    for attempt in range(row["site_attempt"]):
        if kind == "cave" and hints and attempt % 2:
            site = hints.pop()
        else:
            x, z = map(int, rng.integers(-24000, 24001, size=2))
            if row["biome"]:
                reply, _, _ = native.command(
                    "search",
                    x=x,
                    z=z,
                    biome_groups=BIOMES,
                    biome_group=row["biome"],
                    resolution=64,
                    max_checks=16384,
                )
                require_status(reply, "found")
            else:
                reply, _, _ = native.command("search_v3", x=x, z=z, scene_kind=kind)
            site = reply["site"]
        terrain, low, found = survey(native, site, 16, y_bounds)
        sites.append(site)
        if kind == "cave":
            hints.extend((x, z) for x, _, z in found if max(abs(x - site[0]), abs(z - site[1])) >= 8)
        if attempt + 1 < row["site_attempt"]:
            relative = [(int(x - low[0]), int(y - low[1]), int(z - low[2])) for x, y, z in found]
            # Consume the source floor ordering before drawing the next site.
            list(site_candidates(terrain, native.definitions, kind, rng, Counter(), low[1], hints=relative))
    return sites


def choose_blocks(frame, slots, vocabulary, targets, expected, exclusive, reselect):
    choices = set(slots)
    flags = {
        vocabulary.index(name) * 256 + param: (name, param) for name, param in targets if name in vocabulary
    }
    if reselect:
        lookup = np.zeros(len(vocabulary) * 256, dtype=bool)
        lookup[list(flags)] = True
        choices.update(np.flatnonzero(lookup[frame].any(axis=1)).tolist())
    present = {i: {flags[k] for k in np.unique(frame[i]).tolist() if k in flags} for i in choices}
    variants = combinations(sorted(choices), 3) if reselect else [tuple(slots)]
    score, chosen, hit = (-1, -1, -1, -1), slots, set()
    for variant in variants:
        states = set.union(*(present[i] for i in variant))
        ranking = len(states & exclusive), len(states & expected), len(states), len(set(variant) & set(slots))
        if ranking > score:
            score, chosen, hit = ranking, variant, states
    return score, chosen, hit


def recapture(row, source, source_meta, craftium, output, targets, max_wait, trace, emerge_margin):
    started = time.monotonic()
    scene = row["source_vox2pix_scene"]
    seed = row["world_seed"]
    origin = row["region_origin_enu"]
    low = xyz(origin)
    center = xyz(row["center_enu"])
    game = _builds(craftium)
    if game != (source_meta["engine_build"], source_meta["game_build"]):
        raise ValueError(f"{scene}: local engine or VoxeLibre revision differs from the source")
    if seed != source["world_seed"] or origin != source["region_origin_enu"]:
        raise ValueError(f"{scene}: source and planned seed/origin differ")
    expected = {tuple(token) for token in row["observed_new_states"]}
    exclusive = {tuple(token) for token in row["exclusive_new_states_in_this_plan"]}
    settings = dict(
        SETUP,
        vox2pix_v3=True,
        voxwm_static_recapture=True,
        voxwm_skip_rgb=True,
        vox2pix_cave_hints=bool(trace and source["kind"] == "cave"),
    )
    best = None
    timings = []
    variants = []
    engine = StaticSession(
        "procedural",
        seed,
        "auto",
        craftium,
        scene_script=HERE / "scene.lua",
        scene_assets=(REUSED_LUA / "biome_search.lua", REUSED_LUA / "scene_geometry.lua"),
        native_settings=settings,
        image_size=(64, 36),
        fixed_step=True,
        voxel_observation=False,
        render_rgb=False,
    )
    engine.failure_log = output / f"{scene}.failure.log"
    with engine as session:
        native = NativeScene(session)
        native.search_deadline = started + (180 if trace or max_wait > 384 else 100)
        sites = prime_surveys(native, source, trace) if trace else []
        result, _, _ = native.command(
            "emerge_capture",
            low_xyz=[low[0] - emerge_margin, -112, low[2] - emerge_margin],
            high_xyz=[low[0] + 47 + emerge_margin, 240, low[2] + 47 + emerge_margin],
        )
        require_status(result, "capture_emerged")
        result, _, _ = native.command("static_surface", center_xyz=center)
        require_status(result, "located")
        surface = int(result["surface_y"])
        category = {
            "group": biome_group(result["biome"], BIOMES),
            "stratum": height_type(center[1], surface),
            "surface_y": surface,
            "biome": result["biome"],
        }
        result, _, _ = native.command(
            "pose",
            relocate=True,
            foot_xyz=source["survey_foot_xyz"],
            anchor_kind=source["anchor_kind"],
            yaw=0.0,
            pitch=0.0,
        )
        require_status(result, "posed")
        if source["kind"] == "underwater":
            recent = deque(maxlen=12)
            spawn = source["survey_foot_xyz"][1]
            for _ in range(480):
                _, info = native.step()
                recent.append(np.asarray(info["player_pos"]))
                if (
                    len(recent) == 12
                    and recent[-1][1] < spawn - 0.005
                    and np.ptp(recent, axis=0).max() <= 0.003
                ):
                    break
        vocabulary = native.vocabulary
        ignore = vocabulary.index("ignore")
        cursor = 0
        checkpoints = [72, 96, 144, 192, 264, 384, *range(408, max_wait + 1, 24)]
        for step in checkpoints:
            if best is not None and time.monotonic() + 3 >= native.search_deadline:
                break
            for _ in range(step - cursor):
                native.step()
            cursor = step
            reply, _, info = native.command("snapshot", low_xyz=low, anchor_kind=source["anchor_kind"])
            require_status(reply, "captured")
            raw = decode_region((session.world / "vox2pix_region.bin").read_bytes())
            nodes = native.lookup[raw["node"]]
            if np.any(nodes == 65535) or np.any(nodes == ignore):
                raise ValueError(f"{scene}: unregistered or unloaded native voxel")
            codes = nodes.astype(np.int32) * 256 + raw["param2"]
            score, selected, hit = choose_blocks(
                blocks(codes),
                row["block_indices"],
                vocabulary,
                targets,
                expected,
                exclusive,
                trace is not None,
            )
            tick = int(info["native_client_tick"])
            timings.append(
                {
                    "server_tick": tick,
                    "observer_wait_ticks": step,
                    "target_types": score[1],
                    "exclusive_types": score[0],
                    "all_new_types": score[2],
                }
            )
            improve = best is None or score > best[0]
            different = trace and hit and not any(hit <= previous[4] for previous in variants)
            if improve or different:
                candidate = (score, nodes, raw["param2"].copy(), raw["light"].copy(), hit, tick, selected)
                if improve:
                    best = candidate
                if different:
                    variants = [previous for previous in variants if not previous[4] < hit]
                    variants.append(candidate)
            if expected <= hit:
                break
    _, nodes, param2, light, hit, tick, selected = best
    folder = output / scene
    folder.mkdir()
    np.savez_compressed(
        folder / "snapshot.npz", node_id=nodes, param2=param2, light=light, node_names=np.asarray(vocabulary)
    )
    meta = {
        "scene": scene,
        "world_seed": seed,
        "region_origin_enu": origin,
        "block_indices": list(selected),
        "primed_survey_sites": sites,
        "engine_build": game[0],
        "game_build": game[1],
        "capture_tick": tick,
        "mapgen": "v7",
        "actual_type": category,
        "expected_type": row["expected_static_type"],
        "same_type": all(category[k] == row["expected_static_type"][k] for k in ("group", "stratum")),
        "present_target_states": sorted(hit),
        "missing_target_states": sorted(expected - hit),
        "missing_exclusive_states": sorted(exclusive - hit),
        "captures": timings,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    (folder / "capture.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    for _, alt_nodes, alt_param2, alt_light, alt_hit, alt_tick, alt_selected in variants:
        if alt_tick == tick:
            continue
        location = folder / f"tick_{alt_tick}"
        location.mkdir()
        np.savez_compressed(
            location / "snapshot.npz",
            node_id=alt_nodes,
            param2=alt_param2,
            light=alt_light,
            node_names=np.asarray(vocabulary),
        )
        alt_meta = dict(
            meta,
            capture_tick=alt_tick,
            block_indices=list(alt_selected),
            present_target_states=sorted(alt_hit),
            missing_target_states=sorted(expected - alt_hit),
            missing_exclusive_states=sorted(exclusive - alt_hit),
        )
        (location / "capture.json").write_text(json.dumps(alt_meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--craftium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--survey-metadata", type=Path)
    parser.add_argument("--max-wait-ticks", type=int, default=384)
    parser.add_argument("--emerge-margin", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_wait_ticks < 384 or args.emerge_margin < 0:
        parser.error("Use positive workers, at least 384 observation ticks and a nonnegative margin")
    plan = json.loads(args.plan.read_text())["targets"]
    targets = {tuple(token) for row in plan for token in row["observed_new_states"]}
    trace = json.loads(args.survey_metadata.read_text()) if args.survey_metadata else None
    maps = {m["scene"]: m for m in json.loads(args.inventory.read_text())["maps"]}
    source_meta = json.loads(args.source_metadata.read_text())
    if args.scene:
        plan = [r for r in plan if r["source_vox2pix_scene"] in args.scene]
        if len(plan) != len(set(args.scene)):
            parser.error("Unknown scene in the replacement plan")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for row in plan:
            scene = row["source_vox2pix_scene"]
            if (args.output / scene / "capture.json").is_file():
                continue
            job = pool.submit(
                recapture,
                row,
                maps[scene],
                source_meta[scene],
                args.craftium.resolve(),
                args.output,
                targets,
                args.max_wait_ticks,
                trace,
                args.emerge_margin,
            )
            pending[job] = scene
        for job in as_completed(pending):
            scene = pending[job]
            try:
                result = job.result()
                print(
                    f"{scene} captured in {result['wall_seconds']}s "
                    f"type={result['actual_type']['group']}/{result['actual_type']['stratum']} "
                    f"missing={len(result['missing_target_states'])} "
                    f"exclusive_missing={result['missing_exclusive_states']}",
                    flush=True,
                )
            except (ValueError, RuntimeError, OSError, BrokenProcessPool) as exc:
                failure = {"scene": scene, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
    (args.output / "capture_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
