"""Freeze three selected chunks per snapshot before assigning any VAE split."""

import argparse
import gzip
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np

from .full48_scan import _tokens_array
from .full48_source import iter_archive_shards, load_reference, validate_source_header
from .outer_chunk_audit import FREQUENCY, chain_indices, pooled_counts
from .outer_chunk_rank import CENTER_INDEX, blocks, mixed_outer_indices, motion_scores


def write_row(stream, scene, seed, cell, origin, selected):
    stream.write(
        json.dumps(
            {
                "scene": scene,
                "world_seed": seed,
                "quota_cell": cell,
                "region_origin_enu": origin,
                "blocks": [{"index": i, "counts": histogram} for i, histogram in selected],
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def write_archive_row(stream, prefix, sample, frame, frequency, motion=None):
    index, category, metadata = sample
    seed, group, stratum, origin = json.loads(metadata)
    cell = [category, group, stratum] if category == "Static" else [category]
    picked = [CENTER_INDEX, *mixed_outer_indices(frame, frequency, motion, limits=(2,))[2]]
    selected = []
    for slot in sorted(picked):
        counts = np.bincount(frame[slot])
        ids = np.flatnonzero(counts)
        selected.append((slot, list(zip(ids.tolist(), counts[ids].tolist(), strict=True))))
    write_row(stream, f"{prefix}:{index}", seed, cell, origin, selected)


def freeze_archive(archive, label, path, frequency, retired):
    reference = load_reference()
    emitted = 0
    removed = set()
    with gzip.open(path, "wt", compresslevel=1) as stream:
        for number, part in enumerate(iter_archive_shards(archive), 1):
            if number == 1 and validate_source_header(part.header, reference)["category"] != label:
                raise ValueError(f"Expected the {label} archive")
            size = json.loads(part.files["READY"])["samples"]
            snapshots = _tokens_array(part.files, "snapshots", size, part.shard_name)
            with closing(sqlite3.connect(":memory:")) as db:
                db.deserialize(part.files["journal.sqlite"])
                rows = db.execute("""
                    SELECT idx, category, json_extract(metadata, '$.world_seed', '$.biome_group',
                        '$.stratum', '$.region_origin') FROM samples ORDER BY idx
                """).fetchall()
            if [idx for idx, _, _ in rows] != list(range(size)):
                raise ValueError(f"Missing indices in {part.shard_name}")

            prefix = f"{part.root_name}:{part.shard_name}"
            if label == "Static":
                for index in range(size):
                    key = f"{part.shard_name}:{index}"
                    if key in retired:
                        removed.add(key)
                        continue
                    write_archive_row(stream, prefix, rows[index], blocks(snapshots[index]), frequency)
                    emitted += 1
            else:
                for indices in chain_indices(part.files, size):
                    chain = np.stack([blocks(snapshots[index]) for index in indices])
                    motion = motion_scores(chain)
                    for index, frame in zip(indices, chain, strict=True):
                        write_archive_row(stream, prefix, rows[index], frame, frequency, motion)
                        emitted += 1
            if number % 25 == 0:
                print(f"{label}: {number} shards, {emitted:,} selected snapshots", flush=True)
    expected = 100_000 - len(retired)
    if removed != retired or emitted != expected:
        raise ValueError(f"{label}: expected {expected:,} snapshots and all recorded replacements")


def overlay(archives_frequency, pairs, plan_path, vox_path, catalog_path, output):
    """Project the saved Vox2Pix maps into the planned trial without generating worlds."""
    inventory = json.loads(vox_path.read_text())
    names = inventory["names"]
    maps = {row["scene"]: row for row in inventory["maps"]}
    plan = json.loads(plan_path.read_text())["targets"]
    retired = {row["original_static"]["id"] for row in plan}
    if len(retired) != len(plan):
        raise ValueError("Duplicate replacement of one original Static snapshot")

    def semantic(code):
        node, param = divmod(int(code), 256)
        return names[node], param

    frequencies = archives_frequency.copy()
    found = set()
    with gzip.open(catalog_path, "rt") as stream:
        for line in stream:
            ident, *_, counts = json.loads(line)
            if ident in retired:
                for code, amount in counts.items():
                    frequencies[int(code)] -= amount
                found.add(ident)
    if found != retired or np.any(frequencies < 0):
        raise ValueError("The Static catalog does not match the planned removals")

    incoming = {semantic(code) for row in plan for code in maps[row["source_vox2pix_scene"]]["counts"]}
    pairs = [*pairs, *sorted(incoming - set(pairs))]
    ids = {pair: i for i, pair in enumerate(pairs)}
    frequencies = np.pad(frequencies, (0, len(pairs) - len(frequencies)))
    with gzip.open(output, "wt", compresslevel=1) as stream:
        for row in plan:
            scene = row["source_vox2pix_scene"]
            saved = maps[scene]
            origin = row["region_origin_enu"]
            seed = row["world_seed"]
            slots = row["block_indices"]
            if origin != saved["region_origin_enu"] or seed != saved["world_seed"] or len(set(slots)) != 3:
                raise ValueError(f"{scene}: saved map and replacement plan differ")
            for code, amount in saved["counts"].items():
                frequencies[ids[semantic(code)]] += amount
            picked = [
                (slot, sorted((ids[semantic(code)], n) for code, n in saved["block_counts"][slot].items()))
                for slot in sorted(slots)
            ]
            label = row["expected_static_type"]
            write_row(
                stream, f"Vox2Pix:{scene}", seed, ["Static", label["group"], label["stratum"]], origin, picked
            )
    if frequencies.sum() != archives_frequency.sum():
        raise ValueError("Replacement projection changed the total number of voxels")
    return pairs, frequencies, retired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_archive", type=Path)
    parser.add_argument("dynamic_archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement-plan", type=Path)
    parser.add_argument("--map-inventory", type=Path)
    parser.add_argument("--static-catalog", type=Path)
    args = parser.parse_args()
    overlays = (args.replacement_plan, args.map_inventory, args.static_catalog)
    if any(overlays) and not all(overlays):
        parser.error("A replacement projection requires its plan, map inventory and Static catalog")

    args.output.mkdir(parents=True)
    reference = load_reference()
    pairs = list(reference["pairs"])
    frequency = pooled_counts(FREQUENCY, reference)
    retired = set()
    files = ["static.jsonl.gz", "dynamic.jsonl.gz"]
    if all(overlays):
        files.append("replacement.jsonl.gz")
        pairs, frequency, retired = overlay(frequency, pairs, *overlays, args.output / files[-1])
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(
                freeze_archive, args.static_archive, "Static", args.output / files[0], frequency, retired
            ),
            pool.submit(
                freeze_archive, args.dynamic_archive, "Dynamic", args.output / files[1], frequency, set()
            ),
        ]
        for job in jobs:
            job.result()
    (args.output / "frozen_chunks.json").write_text(
        json.dumps(
            {
                "files": files,
                "token_pairs": pairs,
                "full48_counts": frequency.tolist(),
                "replacement_projection": bool(retired),
                "block_index": "0-based ENU C-order",
                "source_archives": [str(args.static_archive), str(args.dynamic_archive)],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
