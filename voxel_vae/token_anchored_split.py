"""Assign whole VAE worlds after reserving one uniformly drawn voxel per observed token."""

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")
PERCENT = dict(zip(SPLITS, (80, 10, 10), strict=True))
BLOCK_VOXELS = 16**3
DYNAMIC = ("D1", "D2", "D3", "D4")


def snapshots(paths):
    for path in paths:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                yield json.loads(line)


def inventory(paths, size):
    occurrence_counts = Counter()
    groups = defaultdict(Counter)
    seen = set()
    for row in snapshots(paths):
        slots = [block["index"] for block in row["blocks"]]
        if (
            row["scene"] in seen
            or len(slots) != 3
            or len(set(slots)) != 3
            or not all(0 <= i < 27 for i in slots)
        ):
            raise ValueError(f"Duplicate snapshot or invalid selection: {row['scene']}")
        seen.add(row["scene"])
        groups[row["world_seed"]][tuple(row["quota_cell"])] += 1
        for block in row["blocks"]:
            counts = dict(block["counts"])
            if (
                len(counts) != len(block["counts"])
                or sum(counts.values()) != BLOCK_VOXELS
                or any(not 0 <= token < size or amount <= 0 for token, amount in counts.items())
            ):
                raise ValueError(f"Invalid selected voxel counts at {row['scene']} block {block['index']}")
            occurrence_counts.update(counts)
    return occurrence_counts, groups


def draw_anchors(paths, occurrence_counts, token_pairs, seed):
    """Map a uniform global occurrence ordinal to a snapshot, chunk and matching voxel ordinal."""
    rng = random.Random(seed)
    ranks = {token: rng.randrange(n) for token, n in sorted(occurrence_counts.items())}
    remaining = ranks.copy()
    anchors = {}
    for row in snapshots(paths):
        for block in row["blocks"]:
            for token, count in block["counts"]:
                left = remaining.get(token)
                if left is None:
                    continue
                if left >= count:
                    remaining[token] -= count
                    continue
                name, param2 = token_pairs[token]
                anchors[token] = {
                    "token_id": token,
                    "node_name": name,
                    "raw_param2": param2,
                    "total_candidate_voxels": occurrence_counts[token],
                    "random_global_ordinal": ranks[token],
                    "world_seed": row["world_seed"],
                    "scene": row["scene"],
                    "region_origin_enu": " ".join(map(str, row["region_origin_enu"])),
                    "block_index": block["index"],
                    "voxel_occurrence_ordinal_within_block": left,
                }
                del remaining[token]
        if not remaining:
            break
    if remaining:
        raise ValueError("Some sampled voxel ordinals have no corresponding selected voxel")
    return anchors


def allocate(snapshots_by_seed, pinned, seed):
    """Assign grouped scenes, then use randomly shuffled single-unit seeds to fill exact quotas."""
    groups = {}
    total = Counter()
    for world, by_cell in snapshots_by_seed.items():
        units = Counter()
        for cell, n in by_cell.items():
            if cell[0] in DYNAMIC:
                if n % 5:
                    raise ValueError(f"World {world} has a partial five-frame chain in {cell[0]}")
                units[cell] = n // 5
            else:
                units[cell] = n
        groups[world] = units
        total.update(units)

    if any(n % 10 for n in total.values()):
        raise ValueError("Exact 80/10/10 needs each category's unit count to be divisible by ten")
    quota = {split: Counter({c: n * pct // 100 for c, n in total.items()}) for split, pct in PERCENT.items()}
    remaining = {split: counts.copy() for split, counts in quota.items()}
    assignments = {}
    filler = defaultdict(list)
    larger = []
    rng = random.Random(seed)

    def put(world, split):
        remaining[split].subtract(groups[world])
        assignments[world] = split

    for world in sorted(groups):
        counts = groups[world]
        if world in pinned:
            put(world, "train")
        elif counts.total() == 1:
            filler[next(iter(counts))].append(world)
        else:
            larger.append(world)
    if any(n < 0 for n in remaining["train"].values()):
        raise ValueError("The random token anchors alone exceed a Train category quota")

    rng.shuffle(larger)
    for world in larger:
        allowed = [
            split for split in SPLITS if all(remaining[split][c] >= n for c, n in groups[world].items())
        ]
        if not allowed:
            raise ValueError(f"No split has remaining quota for all scenes of world {world}")
        put(world, rng.choices(allowed, [PERCENT[split] for split in allowed])[0])

    for cell in sorted(total):
        worlds = filler[cell]
        rng.shuffle(worlds)
        offset = 0
        for split in SPLITS:
            size = remaining[split][cell]
            for world in worlds[offset : offset + size]:
                put(world, split)
            offset += size
    if set(groups) != assignments.keys() or any(any(counts.values()) for counts in remaining.values()):
        raise ValueError("The grouped random split did not fill the exact quotas")
    return assignments, quota


def write_dicts(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_split(index_path, output, seed):
    manifest = json.loads(index_path.read_text())
    paths = [index_path.parent / name for name in manifest["files"]]
    token_pairs = manifest["token_pairs"]
    if len({tuple(pair) for pair in token_pairs}) != len(token_pairs):
        raise ValueError("Duplicate (node_name, raw param2) token entries")
    counts, groups = inventory(paths, len(token_pairs))
    anchors = draw_anchors(paths, counts, token_pairs, seed)
    pinned = {row["world_seed"] for row in anchors.values()}
    assigned, quota = allocate(groups, pinned, seed)

    token_totals = {split: Counter() for split in SPLITS}
    for row in snapshots(paths):
        tally = token_totals[assigned[row["world_seed"]]]
        for block in row["blocks"]:
            tally.update(dict(block["counts"]))
    train = set(token_totals["train"])
    missing = {split: sorted(set(token_totals[split]) - train) for split in ("validation", "test")}
    if (
        any(missing.values())
        or train != counts.keys()
        or not all(assigned[world] == "train" for world in pinned)
    ):
        raise ValueError(f"Train token coverage failed: {missing}")

    output.mkdir(parents=True)
    write_dicts(output / "token_anchors.csv", [anchors[token] for token in sorted(anchors)])
    with gzip.open(output / "world_seed_splits.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["world_seed", "split", "snapshot_count", "selected_block_count", "anchored_in_train"]
        )
        for world, split in sorted(assigned.items()):
            n = groups[world].total()
            writer.writerow([world, split, n, 3 * n, int(world in pinned)])
    token_rows = []
    for token, (name, param2) in enumerate(token_pairs):
        token_rows.append(
            {
                "token_id": token,
                "node_name": name,
                "raw_param2": param2,
                **{s: token_totals[s][token] for s in SPLITS},
            }
        )
    write_dicts(output / "token_voxel_counts_by_split.csv", token_rows)
    with (output / "quota_counts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["category", "biome_group", "height_type", "unit", *SPLITS])
        for cell in sorted(quota["train"]):
            name = cell[0]
            biomes = cell[1:] if name == "Static" else ("", "")
            writer.writerow(
                [
                    name,
                    *biomes,
                    "five_frame_chain" if name in DYNAMIC else "snapshot",
                    *(quota[s][cell] for s in SPLITS),
                ]
            )

    def snapshot_count(split):
        return sum(n * (5 if cell[0] in DYNAMIC else 1) for cell, n in quota[split].items())

    world_counts = Counter(assigned.values())
    full_counts = manifest["full48_counts"]
    summary = {
        "replacement_projection": manifest["replacement_projection"],
        "random_seed": seed,
        "anchor_sampling_unit": "uniform selected voxel occurrence",
        "voxel_ordinal": "0-based among matches within the selected 16³ block in ENU C-order",
        "observed_tokens": len(counts),
        "anchors": len(anchors),
        "anchored_train_worlds": len(pinned),
        "anchored_train_snapshots": sum(groups[world].total() for world in pinned),
        "full48_observed_but_not_selected": [i for i, n in enumerate(full_counts) if n and i not in counts],
        "zero_candidate_voxels": sorted(set(range(len(token_pairs))) - counts.keys()),
        "heldout_missing_from_train": missing,
        "split": {
            s: {
                "worlds": world_counts[s],
                "snapshots": snapshot_count(s),
                "selected_chunks": 3 * snapshot_count(s),
                "tokens": len(token_totals[s]),
            }
            for s in SPLITS
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_chunks", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    make_split(args.frozen_chunks, args.output, args.seed)


if __name__ == "__main__":
    main()
