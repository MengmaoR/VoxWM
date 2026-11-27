"""Materialize the planned VAE Static replacements from saved Vox2Pix maps."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from .freeze_split_chunks import write_row
from .outer_chunk_rank import blocks
from .token_anchored_split import make_split, snapshots


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def import_snapshots(source, archive_index, inventory, plan, projected, token_pairs, output):
    names = {name: index for index, name in enumerate(inventory["names"])}
    source_code = np.array([256 * names.get(name, -1) + param2 for name, param2 in token_pairs])
    token_for_code = np.full(256 * len(names), -1, dtype=np.int32)
    known = source_code >= 0
    token_for_code[source_code[known]] = np.flatnonzero(known)
    samples = []
    covered = set()
    (output / "replacements").mkdir()
    with np.load(source, allow_pickle=False) as maps, gzip.open(output / "replacement.jsonl.gz", "wt") as index:
        sources = json.loads(maps["metadata"].item())["scenes"]
        for row in plan:
            scene = row["source_vox2pix_scene"]
            ref = sources[scene]
            seed, origin = row["world_seed"], row["region_origin_enu"]
            if (ref["world_seed"], ref["region_origin_enu"]) != (seed, origin):
                raise ValueError(f"{scene}: the saved map location differs from the plan")
            codes = maps[scene]
            if codes.shape != (48, 48, 48) or codes.min() < 0 or codes.max() >= len(token_for_code):
                raise ValueError(f"{scene}: expected a complete 48³ map with valid source codes")
            ids = token_for_code[codes]
            if ids.min() < 0 or not np.array_equal(source_code[ids], codes):
                raise ValueError(f"{scene}: a voxel does not round-trip through the VAE vocabulary")
            tokens = ids.astype(np.uint16)
            slots = sorted(row["block_indices"])
            if len(slots) != 3 or len(set(slots)) != 3 or slots[0] < 0 or slots[-1] >= 27:
                raise ValueError(f"{scene}: expected exactly three distinct aligned blocks")
            old = row["original_static"]
            snapshot_id = f"Vox2Pix:{scene}"
            cell = ["Static", row["expected_static_type"]["group"], row["expected_static_type"]["stratum"]]
            archived = archive_index[old["id"]]
            original = json.loads(archived["metadata"].read_text())
            if cell != ["Static", original["biome_group"], original["stratum"]] or not archived["snapshot"].is_file():
                raise ValueError(f"{scene}: expected an archived original in the same Static cell")
            region = blocks(tokens)
            selected = []
            for slot in slots:
                values, counts = np.unique(region[slot], return_counts=True)
                histogram = list(zip(values.tolist(), counts.tolist(), strict=True))
                selected.append((slot, histogram))
                covered.update(token_pairs[value] for value in values.tolist())
            expected = projected[snapshot_id]
            if (seed, origin, cell, selected) != (
                expected["world_seed"], expected["region_origin_enu"], expected["quota_cell"],
                [(b["index"], [tuple(pair) for pair in b["counts"]]) for b in expected["blocks"]],
            ):
                raise ValueError(f"{scene}: the selected raw voxels differ from the frozen plan")
            write_row(index, snapshot_id, seed, cell, origin, selected)
            relative = Path("replacements") / f"{scene}.npz"
            np.savez_compressed(output / relative, tokens=tokens, light=np.zeros(tokens.shape, dtype=np.uint8))
            samples.append({
                "snapshot_id": snapshot_id,
                "file": relative.as_posix(),
                "category": "Static",
                "world_seed": seed,
                "region_origin_enu": origin,
                "biome_group": cell[1],
                "stratum": cell[2],
                "block_indices": slots,
                "original_vae_sample": old["id"],
                "original_vae_archive": str(archived["snapshot"]),
                "vox2pix_split": row["vox2pix_original_split"],
                "source_tick": ref["first_tick"],
                "light_observed": False,
                "light_fill_value": 0,
            })
    return samples, covered


def materialize(plan_root, map_inventory, output, seed):
    plan = json.loads((plan_root / "recapture_manifest.json").read_text())["targets"]
    source = plan_root / "archived_vox2pix_reference.npz"
    inventory = json.loads(map_inventory.read_text())
    frozen = plan_root / "frozen_chunks_for_token_split"
    manifest = json.loads((frozen / "frozen_chunks.json").read_text())
    projected = {row["scene"]: row for row in snapshots([frozen / "replacement.jsonl.gz"])}
    if len({row["original_static"]["id"] for row in plan}) != len(plan):
        raise ValueError("The plan replaces an original snapshot more than once")
    if set(projected) != {f"Vox2Pix:{row['source_vox2pix_scene']}" for row in plan}:
        raise ValueError("The frozen index and replacement plan refer to different source scenes")
    archive_index = json.loads((plan_root / "archive_index.json").read_text())["originals"]
    archives = {ident: {key: plan_root / value for key, value in row.items()} for ident, row in archive_index.items()}
    token_pairs = [tuple(pair) for pair in manifest["token_pairs"]]
    if len(token_pairs) != len(set(token_pairs)) or len(token_pairs) > np.iinfo(np.uint16).max + 1:
        raise ValueError("The frozen vocabulary must contain unique uint16 tokens")
    wanted = {tuple(state) for row in plan for state in row["observed_new_states"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".static-import-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        samples, covered = import_snapshots(source, archives, inventory, plan, projected, token_pairs, staging)
        if wanted - covered:
            raise ValueError(f"Missing planned states in the imported blocks: {sorted(wanted - covered)}")
        for name in ("static.jsonl.gz", "dynamic.jsonl.gz"):
            copyfile(frozen / name, staging / name)
        manifest.update(replacement_projection=False, replacement_source="archived_vox2pix_map")
        write_json(staging / "frozen_chunks.json", manifest)
        write_json(staging / "vocabulary.json", {
            "K": len(token_pairs),
            "tokens": [{"id": i, "node_name": name, "param2": param2} for i, (name, param2) in enumerate(token_pairs)],
        })
        make_split(staging / "frozen_chunks.json", staging / "split", seed)
        with gzip.open(staging / "split/world_seed_splits.csv.gz", "rt") as stream:
            splits = {int(row["world_seed"]): row["split"] for row in csv.DictReader(stream)}
        for sample in samples:
            sample["vae_split"] = splits[sample["world_seed"]]
        write_json(staging / "dataset_overlay.json", {
            "source": str(source),
            "base_archives": manifest["source_archives"],
            "replacement_count": len(samples),
            "target_states_covered": len(wanted),
            "samples": samples,
        })
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--map-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    materialize(args.plan_root.resolve(), args.map_inventory.resolve(), args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()
