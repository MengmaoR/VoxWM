"""Bounded native smoke search using the v3 scene and visibility rules."""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .geometry import is_water_liquid
from .native import HERE, REUSED_LUA, SETUP, NativeScene, RenderSession, _builds, scene_world_seed
from .protocol import SIDE, RejectedScene, v3_rgb
from .v3_sampling import scene_visibility, site_candidates


def survey(native, site, columns, y_bounds):
    x, z = site
    bottom, top = y_bounds
    margin = 24 + (columns - 16) // 2
    side = SIDE + columns - 1
    low = x - margin, bottom, z - margin
    high = low[0] + side - 1, top, low[2] + side - 1
    response, _, _ = native.command("survey", low_xyz=low, high_xyz=high)
    if response["status"] != "surveyed":
        raise RejectedScene("Native v3 survey did not emerge")
    nodes = np.fromfile(native.session.world / "vox2pix_survey.bin", dtype="<u2")
    return nodes.reshape(side, side, top - bottom + 1), low, response.get("cave_hints") or []


def search_world(native, kind, config, rng, library_path, stats, sites, columns):
    local_hints = []
    for site_index in range(sites):
        if time.monotonic() >= native.search_deadline:
            break
        if kind == "cave" and local_hints and site_index % 2:
            site = local_hints.pop()
        else:
            x, z = map(int, rng.integers(-24000, 24001, size=2))
            row, _, _ = native.command("search_v3", x=x, z=z, scene_kind=kind)
            site = row["site"]
        stats["surveys"] += 1
        try:
            nodes, low, hints = survey(native, site, columns, config["survey_y"])
        except RejectedScene as exc:
            stats[str(exc)] += 1
            continue
        stats["cave_hints"] += len(hints)
        if kind == "cave":
            local_hints.extend((x, z) for x, _, z in hints
                               if max(abs(x - site[0]), abs(z - site[1])) >= columns // 2)
        relative_hints = [(int(x - low[0]), int(y - low[1]), int(z - low[2])) for x, y, z in hints]
        for (x, y, z), surface_y, anchor_kind in site_candidates(
                nodes, native.definitions, kind, rng, stats, low[1], hints=relative_hints):
            if time.monotonic() >= native.search_deadline:
                raise TimeoutError("V3 per-scene search deadline expired")
            foot = tuple(int(offset + origin) for offset, origin in zip((x, y, z), low, strict=True))
            try:
                if kind != "underwater":
                    scene_visibility(nodes[x-24:x+24, z-24:z+24, y-24:y+24], native.definitions, kind)
                    anchor, _, _ = native.command("anchor", foot_xyz=foot, surface_y=surface_y,
                                                  anchor_kind=anchor_kind)
                    if not anchor["valid"]:
                        raise RejectedScene("Native standing anchor mismatch")
                stats["capture_attempts"] += 1
                result = native.acquire(foot, anchor_kind, config, rng, library_path, scene_kind=kind)
                if kind == "underwater":
                    anchor, _, _ = native.command("anchor", foot_xyz=result[0]["foot_anchor_xyz"],
                                                  surface_y=surface_y, anchor_kind=anchor_kind)
            except RejectedScene as exc:
                stats[str(exc)] += 1
                continue
            result[0].update(surface_y=surface_y, site_attempt=site_index + 1,
                             native_biome=anchor["surface_biome"],
                             surface_biome=anchor["surface_biome"], foot_biome=anchor["foot_biome"])
            return result
        stats["sites_without_accepted_anchor"] += 1
    return None


def capture_v3(slot, config, craftium_dir, library_path, attempts, sites, columns, seconds, first=0):
    engine_build, game_build = _builds(Path(craftium_dir))
    started = time.monotonic()
    stats = Counter(world_starts=0, surveys=0, cave_hints=0, candidate_floors=0, capture_attempts=0)
    rgb = v3_rgb(config)
    settings = dict(SETUP, **rgb, vox2pix_v3=True, voxwm_lighting_probe=True,
                    vox2pix_cave_hints=slot["kind"] == "cave", voxwm_skip_rgb=True)
    if "solid_ambient_floor" in rgb:
        settings["voxwm_material_lighting_probe"] = True
    for attempt in range(first, first + attempts):
        if time.monotonic() >= started + seconds:
            break
        seed = scene_world_seed(slot["scene_id"], config["seed"], attempt)
        stats["world_starts"] += 1
        try:
            with RenderSession(
                "procedural", seed, config["renderer"], craftium_dir, scene_script=HERE / "scene.lua",
                scene_assets=(REUSED_LUA / "biome_search.lua", REUSED_LUA / "scene_geometry.lua"),
                native_settings=settings, image_size=(config["width"], config["height"]),
                fixed_step=True, voxel_observation=False, render_rgb=True,
            ) as session:
                native = NativeScene(session)
                native.search_deadline = started + seconds
                for definition in (*native.definitions.values(), *native.stable_defs.values()):
                    definition["water"] = is_water_liquid(definition)
                result = search_world(native, slot["kind"], config, np.random.default_rng(seed),
                                      library_path, stats, sites, columns)
        except (ConnectionError, TimeoutError, RejectedScene) as exc:
            stats[str(exc)] += 1
            result = None
        if result is not None:
            result[0].update(slot, world_seed=seed, engine_build=engine_build, game_build=game_build,
                             search_seconds=time.monotonic() - started, search_stats=dict(stats))
            return result
        print(f"V3 rejected world {seed} for {slot['kind']}: {json.dumps(dict(stats))}", flush=True)
    raise RejectedScene(f"{slot['kind']} search exhausted after {time.monotonic() - started:.1f}s: {dict(stats)}")
