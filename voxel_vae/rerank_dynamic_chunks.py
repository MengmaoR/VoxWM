"""Freeze the same three positions across each five-frame Dynamic trajectory."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

import numpy as np

from .chunk_selection import extract_selected_chunks, iter_chunk_selections
from .freeze_split_chunks import freeze_archive
from .static_overlap_repair import load, selected_counts, selection_header, write_json
from .token_anchored_split import SPLITS, snapshots


def rerank(bundle, inventory_dir, output):
    index = bundle / "chunk_selection.jsonl.gz"
    header = selection_header(index)
    overlay = load(bundle / "dataset_overlay.json")
    plan = load(Path(overlay["overlap_repair"]["plan"]))
    source = Path(plan["source_bundle"])
    frozen = load(source / "frozen_chunks.json")
    vocabulary = load(bundle / header["vocabulary"])
    pairs = [(row["node_name"], row["param2"]) for row in vocabulary["tokens"]]
    if pairs[:len(frozen["token_pairs"])] != [tuple(pair) for pair in frozen["token_pairs"]]:
        raise ValueError("The published vocabulary differs from the frozen source IDs")
    size = len(pairs)
    frequency = np.pad(np.array(frozen["full48_counts"], dtype=np.int64), (0, size - len(frozen["token_pairs"])))
    archive = Path(overlay["overlap_repair"]["archive"])
    backups = load(archive / "archive.json")["original_samples"]
    replacements = {row["snapshot_id"]: row for row in overlay["samples"] if row.get("light_observed")}
    for name, row in replacements.items():
        original = archive / backups[row["original_vae_sample"]]["snapshot"]
        with np.load(original, allow_pickle=False) as old, np.load(bundle / header["replacement_files"][name], allow_pickle=False) as new:
            frequency += np.bincount(new["tokens"].ravel(), minlength=size)
            frequency -= np.bincount(old["tokens"].ravel(), minlength=size)
    if frequency.min() < 0 or frequency.sum() != sum(frozen["full48_counts"]):
        raise ValueError("Invalid token counts for the repaired dataset")

    counts = {split: np.zeros(size, dtype=np.int64) for split in SPLITS}
    groups = {}
    sizes = Counter()
    changed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dynamic-selection-", dir=output.parent) as tmp:
        temp = Path(tmp)
        dynamic_archive = bundle / header["source_archives"]["vae_100k_dynamic"]
        dynamic_index = temp / "dynamic.jsonl.gz"
        freeze_archive(dynamic_archive, "Dynamic", dynamic_index, frequency, set())
        dynamic = iter(snapshots([dynamic_index]))
        stage = temp / "dataset"
        stage.mkdir()
        copytree(bundle / "replacements", stage / "replacements")
        copy2(bundle / header["vocabulary"], stage / header["vocabulary"])
        header["source_archives"] = {name: os.path.relpath(bundle / path, output)
                                     for name, path in header["source_archives"].items()}
        header["dynamic_chunk_selection"] = "one selection per five-frame chain using rarest frame per position"
        with gzip.open(stage / index.name, "wt", compresslevel=6) as stream:
            stream.write(json.dumps(header, separators=(",", ":")) + "\n")
            baseline = selected_counts(source, source / index.name, frozen, size)
            for (old, hist), current in zip(baseline, iter_chunk_selections(index), strict=True):
                ident = current["snapshot_id"]
                split = current["split"]
                sizes[split] += 1
                if current["category"] != "Static":
                    new = next(dynamic)
                    if (ident, current["world_seed"], current["region_origin_enu"], current["category"]) != (
                        new["scene"], new["world_seed"], new["region_origin_enu"], new["quota_cell"][0],
                    ):
                        raise ValueError("Dynamic frames disagree with the archive or changed order")
                    picks = [block["index"] for block in new["blocks"]]
                    changed += current["chunk_indices"] != picks
                    current["chunk_indices"] = picks
                    hist = [pair for block in new["blocks"] for pair in block["counts"]]
                    key = (current["world_seed"], tuple(current["region_origin_enu"]), current["category"], ident.split(":")[1])
                    expected, expected_split, seen = groups.get(key, (picks, split, 0))
                    if (picks, split) != (expected, expected_split) or len(picks) != 3 or len(set(picks)) != 3:
                        raise ValueError("A dynamic chain changes chunk coordinates or splits")
                    groups[key] = picks, split, seen + 1
                elif ident in replacements:
                    if replacements[ident]["original_vae_sample"] != old["snapshot_id"].split(":", 1)[1]:
                        raise ValueError("A native replacement is out of order")
                    with np.load(bundle / header["replacement_files"][ident], allow_pickle=False) as file:
                        chosen = extract_selected_chunks(file["tokens"], current)
                    counts[split] += np.bincount(chosen.ravel(), minlength=size)
                    hist = []
                elif ident != old["snapshot_id"] or current["chunk_indices"] != old["chunk_indices"]:
                    raise ValueError("A Static row changed slots or order")
                for token, n in hist:
                    counts[split][token] += n
                stream.write(json.dumps(current, separators=(",", ":")) + "\n")

        if next(dynamic, None) is not None or len(groups) != 20_000 or any(n != 5 for _, _, n in groups.values()):
            raise ValueError("Expected exactly 20,000 five-frame Dynamic chains")
        if any(counts[split].sum() != sizes[split] * 3 * 16**3 for split in SPLITS):
            raise ValueError("The new selected voxel totals disagree with the split sizes")
        train = {pair for pair, n in zip(pairs, counts["train"], strict=True) if n}
        targets = {split: {pair for pair, n in zip(pairs, counts[split], strict=True) if n}
                   for split in ("validation", "test")}
        targets["observed_full48"] = {pair for pair, n in zip(pairs, frequency, strict=True) if n}
        for layer in ("scene", "surface"):
            inv = load(inventory_dir / f"{layer}.json")
            targets[f"vox2pix_{layer}"] = {(inv["names"][int(i)//256], int(i)%256) for i, n in inv["counts"].items() if n}
        coverage = {name: {"covered": len(states & train), "total": len(states), "missing": sorted(states - train)}
                    for name, states in targets.items()}
        report = load(bundle / "verification.json")
        report.update(dynamic_chains=len(groups), dynamic_positions=3 * len(groups), dynamic_chunk_frames=15 * len(groups),
                      dynamic_snapshots_with_changed_selection=changed, train_types=len(train), coverage=coverage,
                      split_snapshots=dict(sizes), split_chunks={split: 3 * n for split, n in sizes.items()},
                      coverage_ok=all(not row["missing"] for row in coverage.values()))
        overlay["dynamic_selection"] = {"source_bundle": str(bundle), "positions_per_trajectory": 3,
                                        "rarity": "rarest frame at each position"}
        write_json(stage / "dataset_overlay.json", overlay)
        write_json(stage / "verification.json", report)
        stage.rename(output)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report["coverage_ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vox2pix-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    return 0 if rerank(args.bundle.resolve(), args.vox2pix_inventory.resolve(), args.output.resolve()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
