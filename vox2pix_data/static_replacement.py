"""Read category-repacked Static scenes and simulate same-cell replacements."""

import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np

from vae_data.sharded.protocol import (
    BIOMES,
    STRATA,
    WORLD_BOTTOM,
    WORLD_TOP,
    biome_group,
)
from vae_data.sharded.static_height import height_bounds, rejection
from voxel_vae.full48_scan import _tokens_array
from voxel_vae.full48_source import iter_archive_shards


def static_catalog(archive, output):
    """Retain integer state counts and quota keys only. Raw inputs are read-only."""
    with gzip.open(output, "wt", compresslevel=1) as stream:
        for ns, shard in enumerate(iter_archive_shards(archive), 1):
            with closing(sqlite3.connect(":memory:")) as db:
                db.deserialize(shard.files["journal.sqlite"])
                rows = db.execute("""SELECT idx, json_extract(metadata,'$.biome_group','$.stratum',
                       '$.world_seed','$.region_origin','$.biome_transition','$.repack_source.clip',
                       '$.repack_source.shard','$.repack_source.snapshot_id')
                       FROM samples WHERE category='Static' ORDER BY idx""").fetchall()
            size = json.loads(shard.files["READY"])["samples"]
            if len(rows) != size or shard.root_name != "vae_100k_static":
                raise ValueError("Unexpected static pack or record count")
            array = _tokens_array(shard.files, "snapshots", size, shard.shard_name)
            for idx, fields in rows:
                counts = np.bincount(array[idx].ravel())
                states = np.flatnonzero(counts)
                row = [
                    f"{shard.shard_name}:{idx}",
                    *json.loads(fields),
                    dict(zip(states.tolist(), counts[states].tolist(), strict=True)),
                ]
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if ns % 25 == 0:
                print(f"Static: {ns} shards", flush=True)


def classify(m):
    east, north, low = m["region_origin_enu"]
    center = low + 24
    surface = m["surface_y"]
    if m["survey_foot_xyz"][::2] != m["foot_anchor_xyz"][::2]:
        raise ValueError(f"Survey column moved: {m['scene']}")
    group = biome_group(m["surface_biome"], BIOMES)
    delta = center - surface
    band = (
        "elevated_complex"
        if delta >= 9
        else "surface"
        if delta >= -7
        else "shallow_underground"
        if delta >= -24
        else "gap"
        if delta > -32
        else "deep_cave"
    )
    status = "eligible"
    if low < WORLD_BOTTOM or low + 47 > WORLD_TOP:
        status = "outside_world_bounds"
    elif band == "gap":
        status = "height_gap"
    elif group is None or m["ignore"]:
        status = "unmapped_biome_or_ignore"
    else:
        lo, hi = height_bounds(surface, band)
        if not lo <= center <= hi:
            raise ValueError("Stratum classification disagrees with the native Static rules")
        status = rejection(band, m["center_air"] / 4096, m["center_void"] / 4096) or status

    # Additional, narrower evidence sample: the stored center column witnesses
    # the reported survey surface. Ocean-floor queries can omit overlying water.
    # These three leaf nodes in the actual columns have the native leaves group.
    skipped = {"air", "mcl_core:spruceleaves", "mcl_core:jungleleaves", "mcl_core:darkleaves"}
    y = surface - low
    if not 0 <= y < 48:
        witness = "survey_surface_outside_crop"
    elif m["candidate_surface_lower_y"] > surface:
        witness = "qualifying_node_above_survey_surface"
    elif m["observed_column_top"] != surface or any(n not in skipped for n in m["center_column"][y + 1 :]):
        witness = "column_unresolved"
    elif y == 47 and m["kind"] == "cave":
        witness = "cave_surface_at_crop_ceiling"
    else:
        witness = "survey_and_column_agree"
    return {
        "group": group,
        "band": band,
        "status": status,
        "witness": witness,
        "center_y": center,
        "delta": delta,
        "aligned": (east % 16 == 0 and north % 16 == 0),
    }


def rank(counts, frequency):
    # Same complete-vector ordering as voxel_vae.outer_chunk_rank._rank_entry,
    # applied to a full sparse 48³ histogram instead of a dense 16³ array.
    ordered = sorted((int(frequency[i]), -n, -i) for i, n in counts.items())
    return tuple(tuple(column) for column in zip(*ordered, strict=True))


def read_candidates(catalog, demand, overlap_ids, frequency, seed_inventory):
    from bisect import insort

    winners, endpoints = defaultdict(list), {}
    cells, transitions, totals = Counter(), Counter(), Counter()
    with gzip.open(catalog, "rt") as file:
        for line in file:
            ident, group, band, seed, origin, transition, clip, shard, idx, raw = json.loads(line)
            cell = (group, band)
            counts = {int(t): n for t, n in raw.items()}
            totals.update(counts)
            cells[cell] += 1
            transitions[(ident.split(":")[0], group)] += int(transition)
            if seed not in seed_inventory:
                raise ValueError(f"Unknown VAE seed for {ident}")
            if not (demand[cell] or ident in overlap_ids):
                continue
            rec = {
                "id": ident,
                "cell": cell,
                "seed": seed,
                "origin": origin,
                "transition": bool(transition),
                "clip": clip,
                "shard": shard,
                "idx": idx,
                "hist": counts,
                "rank": rank(counts, frequency),
            }
            if ident in overlap_ids:
                endpoints[ident] = rec
            if demand[cell]:
                # Retain the n least-rare full maps. Endpoints are retained separately.
                group_rows = winners[cell]
                insort(group_rows, (rec["rank"], ident, rec))
                if len(group_rows) > demand[cell]:
                    group_rows.pop(0)
    if set(endpoints) != overlap_ids:
        raise ValueError("Earlier overlapping Static IDs do not match the archive")
    return winners, endpoints, cells, transitions, totals


def maximum_pairs(pairs, capacities):
    """Maximum number of distinct old overlap pairs eliminable with these quotas."""
    assigned = {}

    def augment(i, visited):
        for cell in pairs[i]:
            for k in range(capacities[cell]):
                slot = (cell, k)
                if slot in visited:
                    continue
                visited.add(slot)
                if slot not in assigned or augment(assigned[slot], visited):
                    assigned[slot] = i
                    return True
        return False

    return sum(augment(i, set()) for i in pairs)


def choose(eligible, winners, endpoints, overlaps):
    capacity = Counter((m["group"], m["band"]) for m in eligible)
    options, indices = {}, {}
    for i, pair in enumerate(overlaps):
        options[i] = []
        for item, category in zip(pair["ids"], pair["categories"], strict=True):
            if category == "Static":
                options[i].append(endpoints[item]["cell"])
                if item in indices:
                    raise ValueError("This inventory expects disjoint historical overlap pairs")
                indices[item] = i
    remaining = maximum_pairs(options, capacity)
    best_possible = remaining
    chosen = defaultdict(list)
    removed = set()
    for item, rec in sorted(endpoints.items(), key=lambda e: (e[1]["rank"], e[0]), reverse=True):
        i, cell = indices[item], rec["cell"]
        if not remaining or i not in options or not capacity[cell]:
            continue
        # Prefer the least rare endpoint consistent with maximum overlap removal.
        rest = {j: cells for j, cells in options.items() if j != i}
        capacity[cell] -= 1
        if maximum_pairs(rest, capacity) >= remaining - 1:
            chosen[cell].append((rec, "overlap", overlaps[i]["seed"]))
            removed.add(item)
            options = rest
            remaining -= 1
        else:
            capacity[cell] += 1
    assert remaining == 0
    for cell, need in capacity.items():
        available = [rec for _, item, rec in reversed(winners[cell]) if item not in removed]
        if len(available) < need:
            raise ValueError(f"Insufficient Static victims in {cell}")
        chosen[cell].extend((r, "least_rare_in_cell", None) for r in available[:need])
    matches = []
    for m in sorted(eligible, key=lambda r: int(r["scene"].split("_")[1])):
        cell = (m["group"], m["band"])
        rec, why, seed = chosen[cell].pop(0)
        assert rec["cell"] == cell
        matches.append((m, rec, why, seed))
    if len({r["id"] for _, r, _, _ in matches}) != len(matches):
        raise ValueError("Static victim reused")
    retired = {r["id"] for _, r, _, _ in matches}
    closed = [p for p in overlaps if retired.intersection(p["ids"])]
    assert len(closed) == best_possible
    return matches, closed


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def run(args):
    from voxel_vae.full48_source import REPORTS, load_reference
    from voxel_vae.outer_chunk_audit import FREQUENCY, full_counts_by_category

    report = REPORTS / "scene_overlap_20261126"
    source = json.loads(args.maps.read_text())
    targets = {int(k): n for k, n in source["targets"].items()}
    names = source["names"]

    def semantic(code):
        return names[code // 256], code % 256

    def label(code):
        name, param = semantic(code)
        return f"{name}@{param}"

    ref = load_reference()
    freq = full_counts_by_category(FREQUENCY, ref)
    initial = freq["Static"] + freq["Dynamic"]
    ids = {p: i for i, p in enumerate(ref["pairs"])}
    # Append genuinely new semantic states to an in-memory ID space for ranking.
    new_pairs = sorted({semantic(code) for code in targets} - ids.keys())
    ids.update((p, ref["K"] + i) for i, p in enumerate(new_pairs))
    ext = {code: ids[semantic(code)] for m in source["maps"] for code in map(int, m["counts"])}
    if any(ext[t] < len(initial) and initial[ext[t]] for t in targets):
        raise ValueError("Target set includes an existing observed VAE state")
    for m in source["maps"]:
        m.update(classify(m))
    eligible = [m for m in source["maps"] if m["status"] == "eligible"]
    demand = Counter((m["group"], m["band"]) for m in eligible)
    overlaps = json.loads((report / "summary.json").read_text())
    pairs = overlaps["static"]["overlaps"] + overlaps["static_dynamic"]["overlaps"]
    endpoints = {i for p in pairs for i, c in zip(p["ids"], p["categories"], strict=True) if c == "Static"}
    with gzip.open(report / "scenes_per_seed.csv.gz", "rt") as f:
        seeds = {int(r["world_seed"]) for r in csv.DictReader(f)}
    incoming_seeds = {m["world_seed"] for m in source["maps"]}
    if seeds & incoming_seeds or len(incoming_seeds) != len(source["maps"]):
        raise ValueError(
            "New maps share seeds. Check spatial intersections before claiming zero new overlaps"
        )
    winners, ends, cells, transitions, totals = read_candidates(
        args.catalog, demand, endpoints, initial, seeds
    )
    if totals != {i: int(n) for i, n in enumerate(freq["Static"]) if n}:
        raise ValueError("Static archive frequencies do not reproduce the full48 audit")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    map_rows = []
    for m in sorted(source["maps"], key=lambda r: int(r["scene"].split("_")[1])):
        lo = m["region_origin_enu"][2]
        h = {int(t): n for t, n in m["hits"].items()}
        counts.update(h)
        map_rows.append(
            {
                "scene": m["scene"],
                "seed": m["world_seed"],
                "kind": m["kind"],
                "anchor": m["anchor_kind"],
                "center_east": m["foot_anchor_xyz"][0],
                "center_north": m["foot_anchor_xyz"][2],
                "center_up": m["center_y"],
                "region_bottom": lo,
                "region_top": lo + 47,
                "survey_surface_y": m["surface_y"],
                "observed_surface_lower_bound": m["candidate_surface_lower_y"],
                "center_minus_survey": m["delta"],
                "surface_biome": m["surface_biome"],
                "foot_biome": m["foot_biome"],
                "vae_biome_group": m["group"],
                "relative_height_band": m["band"],
                "recorded_quota_status": m["status"],
                "column_evidence": m["witness"],
                "vae_horizontal_alignment": m["aligned"],
                "center_air_voxels": m["center_air"],
                "center_void_voxels": m["center_void"],
                "new_state_types": len(h),
                "new_state_voxels": sum(h.values()),
                "center_hit_voxels": sum(m["center_hits"].values()),
                "states": " | ".join(f"{label(t)}={h[t]}" for t in sorted(h)),
            }
        )
    if counts != targets:
        raise ValueError("Target frequencies did not reconcile")
    write_csv(args.output / "maps.csv", map_rows)

    experiments = {
        "recorded_quota": eligible,
        "column_supported": [m for m in eligible if m["witness"] == "survey_and_column_agree"],
    }
    inventory = {}
    allplans = []
    token_plans = {}
    for mode, selection in experiments.items():
        matches, closed = choose(selection, winners, ends, pairs)
        replaced = Counter()
        added = Counter()
        changes = Counter()
        for m, old, why, pair_seed in matches:
            replaced.update(old["hist"])
            added.update({ext[int(t)]: n for t, n in m["counts"].items()})
            changes[(old["id"].split(":")[0], m["group"])] += int(old["transition"])
            allplans.append(
                {
                    "plan": mode,
                    "incoming_scene": m["scene"],
                    "incoming_seed": m["world_seed"],
                    "group": m["group"],
                    "stratum": m["band"],
                    "incoming_center_y": m["center_y"],
                    "incoming_surface_y": m["surface_y"],
                    "victim": old["id"],
                    "victim_clip": old["clip"],
                    "original_shard": old["shard"],
                    "original_snapshot_id": old["idx"],
                    "victim_seed": old["seed"],
                    "victim_enu_origin": ",".join(map(str, old["origin"])),
                    "victim_center_y": old["origin"][2] + 24,
                    "victim_transition": old["transition"],
                    "victim_rarest_token_global_frequency": old["rank"][0][0],
                    "reason": why,
                    "resolved_overlap_seed": pair_seed,
                }
            )
        after_static = np.zeros(len(ids), dtype=np.int64)
        after_static[: len(initial)] = freq["Static"]
        for i, n in replaced.items():
            after_static[i] -= n
        for i, n in added.items():
            after_static[i] += n
        after = after_static.copy()
        after[: len(initial)] += freq["Dynamic"]
        if min(after_static) < 0 or after_static.sum() != freq["Static"].sum():
            raise ValueError("Inconsistent simulated volume")

        # New-state coverage under the previously discussed center + two rare outer chunks.
        # Only incoming maps can contain these states; re-rank them at post-swap frequencies.
        selected, center = Counter(), Counter()
        for m in selection:
            blocks = [{ext[int(t)]: n for t, n in h.items()} for h in m["block_counts"]]
            outer = sorted((rank(h, after), i) for i, h in enumerate(blocks) if i != 13)
            center.update(blocks[13])
            for i in [13, outer[0][1], outer[1][1]]:
                selected.update(blocks[i])
        token_plans[mode] = {t: (added[ext[t]], center[ext[t]], selected[ext[t]]) for t in targets}
        old_observed = np.flatnonzero(initial)
        removed_types = [ref["pairs"][i] for i in old_observed if not after[i]]
        lost_static = [ref["pairs"][i] for i in np.flatnonzero(freq["Static"]) if not after_static[i]]
        inventory[mode] = {
            "replacements": len(matches),
            "overlap_first": len(closed),
            "fallback": len(matches) - len(closed),
            "overlap_pairs_left": len(pairs) - len(closed),
            "static_pairs_left": sum(p["categories"].count("Static") == 2 for p in pairs if p not in closed),
            "cross_pairs_left": sum(p["categories"].count("Static") == 1 for p in pairs if p not in closed),
            "introduced_overlaps": 0,
            "lost_preexisting_states": removed_types,
            "lost_preexisting_static_states": lost_static,
            "gained_states": sum(added[ext[t]] > 0 for t in targets),
            "gained_voxels": sum(added[ext[t]] for t in targets),
            "center_gained_states": sum(center[ext[t]] > 0 for t in targets),
            "center_plus_two_gained_states": sum(selected[ext[t]] > 0 for t in targets),
            "observed_static_after": int(np.count_nonzero(after_static)),
            "observed_pooled_after": int(np.count_nonzero(after)),
            "matched_cells": len({r["cell"] for _, r, _, _ in matches}),
            "removed_biome_transitions": sum(changes.values()),
            "previous_shard_biome_cells_falling_below_soft_target_if_no_incoming_transition": sum(
                transitions[k] >= 4 and transitions[k] - n < 4 for k, n in changes.items()
            ),
            "original_categories_preserved": Counter((m["group"], m["band"]) for m, _, _, _ in matches)
            == Counter(r["cell"] for _, r, _, _ in matches),
        }
    write_csv(args.output / "replacement_plan.csv", allplans)
    token_rows = []
    for t in sorted(targets, key=semantic):
        group = [m for m in source["maps"] if str(t) in m["hits"]]
        row = {
            "node_name": semantic(t)[0],
            "raw_param2": semantic(t)[1],
            "total_voxels": targets[t],
            "distinct_maps": len(group),
            "maps": " ".join(sorted(m["scene"] for m in group)),
        }
        for mode, results in token_plans.items():
            row.update(
                zip(
                    (f"{mode}_{x}_voxels" for x in ("full", "center", "center_plus_two")),
                    results[t],
                    strict=True,
                )
            )
        token_rows.append(row)
    write_csv(args.output / "token_coverage.csv", token_rows)
    groups = []
    for group in sorted(BIOMES):
        r = [m for m in source["maps"] if m["group"] == group]
        n = {st: sum(m["status"] == "eligible" and m["band"] == st for m in r) for st in STRATA}
        groups.append(
            dict(
                group=group,
                maps=len(r),
                **n,
                height_gap=sum(m["status"] == "height_gap" for m in r),
                outside_bounds=sum(m["status"] == "outside_world_bounds" for m in r),
                column_supported=sum(
                    m["status"] == "eligible" and m["witness"] == "survey_and_column_agree" for m in r
                ),
            )
        )
    write_csv(args.output / "biome_height.csv", groups)
    allmaps = source["maps"]

    def lohi(field):
        return [min(m[field] for m in allmaps), max(m[field] for m in allmaps)]

    summary = {
        "total_vox2pix_maps": source["total_maps"],
        "new_states": len(targets),
        "new_voxel_occurrences": sum(targets.values()),
        "maps": len(allmaps),
        "seeds": len(incoming_seeds),
        "distinct_full_map_histograms": len({tuple(sorted(m["counts"].items())) for m in allmaps}),
        "eligible_surface_evidence": Counter(m["witness"] for m in eligible),
        "kinds": Counter(m["kind"] for m in allmaps),
        "groups": len({m["group"] for m in allmaps}),
        "center_y_range": lohi("center_y"),
        "survey_surface_y_range": lohi("surface_y"),
        "stored_band_counts": Counter(m["band"] for m in allmaps),
        "recorded_quota_status": Counter(m["status"] for m in allmaps),
        "strata": Counter(m["band"] for m in eligible),
        "aligned": sum(m["aligned"] for m in allmaps),
        "unique_center_hits": len({t for m in allmaps for t in m["center_hits"]}),
        "absolute_height_state_ceiling": len(
            {t for m in allmaps if m["status"] != "outside_world_bounds" for t in m["hits"]}
        ),
        "prior_static": int(np.count_nonzero(freq["Static"])),
        "prior_pooled": int(np.count_nonzero(initial)),
        "original_static_scenes": sum(cells.values()),
        "original_overlap_pairs": len(pairs),
        "simulations": inventory,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cat = sub.add_parser("catalog")
    cat.add_argument("archive", type=Path)
    cat.add_argument("output", type=Path)
    plan = sub.add_parser("simulate")
    for name in ("maps", "catalog", "output"):
        plan.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "catalog":
        static_catalog(args.archive, args.output)
    else:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
