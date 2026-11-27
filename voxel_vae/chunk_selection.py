"""Publish and read a fixed list of three VAE chunks per snapshot."""

import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .outer_chunk_rank import CENTER_INDEX, blocks
from .token_anchored_split import SPLITS, snapshots


FORMAT = "voxwm-vae-chunk-selection-v1"


def iter_chunk_selections(path, *, split=None):
    """Yield saved snapshot records in index order, optionally for one split."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    with gzip.open(path, "rt") as stream:
        if json.loads(next(stream))["format"] != FORMAT:
            raise ValueError("Unsupported VAE chunk selection format")
        for line in stream:
            row = json.loads(line)
            if split is None or row["split"] == split:
                yield row


def extract_selected_chunks(snapshot, row):
    """Extract the three 16³ arrays listed in a published row."""
    return blocks(snapshot)[row["chunk_indices"]].reshape(3, 16, 16, 16)


def publish(bundle, output, overrides):
    frozen = json.loads((bundle / "frozen_chunks.json").read_text())
    overlay = json.loads((bundle / "dataset_overlay.json").read_text())
    imports = {row["snapshot_id"]: row for row in overlay["samples"]}
    source_archives = {
        Path(path).name.removesuffix(".tar.zst"): os.path.relpath(path, output.parent)
        for path in frozen["source_archives"]
    }
    with gzip.open(bundle / "split/world_seed_splits.csv.gz", "rt") as stream:
        assignment = {int(row["world_seed"]): row["split"] for row in csv.DictReader(stream)}
    split_summary = json.loads((bundle / "split/summary.json").read_text())["split"]
    header = {
        "format": FORMAT,
        "chunk_edge": 16,
        "region_edge": 48,
        "block_index": "0-based ENU C-order",
        "source_archives": source_archives,
        "replacement_files": {name: os.path.relpath(bundle / row["file"], output.parent) for name, row in imports.items()},
        "vocabulary": os.path.relpath(bundle / "vocabulary.json", output.parent),
        "overrides": overrides,
    }

    found = set()
    seen = set()
    counts = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".chunk-selection-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        with gzip.open(staged, "wt", compresslevel=6) as stream:
            stream.write(json.dumps(header, separators=(",", ":")) + "\n")
            for row in snapshots([bundle / file for file in frozen["files"]]):
                scene = row["scene"]
                seed = row["world_seed"]
                category = row["quota_cell"][0]
                split = assignment[seed]
                indices = [block["index"] for block in row["blocks"]]
                if (
                    scene in seen or len(indices) != 3 or len(set(indices)) != 3
                    or CENTER_INDEX not in indices or any(not 0 <= index < 27 for index in indices)
                ):
                    raise ValueError(f"Invalid frozen selection for {scene}")
                seen.add(scene)
                if scene in overrides:
                    source = imports[scene]
                    same_source = (category, seed, split, indices) == (
                        source["category"], source["world_seed"], source["vae_split"], source["block_indices"]
                    )
                    if category != "Static" or split != "train" or not same_source:
                        raise ValueError(f"An override must match an imported Static Train row: {scene}")
                    indices = overrides[scene]
                    found.add(scene)
                counts[split] += 1
                saved = {"snapshot_id": scene, "category": category, "world_seed": seed, "split": split,
                         "chunk_indices": indices, "region_origin_enu": row["region_origin_enu"]}
                stream.write(json.dumps(saved, separators=(",", ":")) + "\n")
        if found != overrides.keys() or counts != {s: split_summary[s]["snapshots"] for s in SPLITS}:
            raise ValueError("The published snapshot list is incomplete")
        os.link(staged, output)
    print(json.dumps({"snapshots": dict(counts), "chunks": {s: 3 * counts[s] for s in SPLITS}, "overrides": overrides}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--override", action="append", nargs=4, metavar=("SNAPSHOT", "I", "J", "K"), default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    overrides = {}
    for scene, *values in args.override:
        try:
            indices = sorted(map(int, values))
        except ValueError:
            parser.error("Chunk indices must be integers")
        if scene in overrides or len(set(indices)) != 3 or not (0 <= indices[0] and indices[-1] < 27):
            parser.error("Each override needs a unique snapshot and three distinct chunk indices in 0..26")
        overrides[scene] = indices
    publish(args.bundle.resolve(), args.output.resolve(), overrides)


if __name__ == "__main__":
    main()
