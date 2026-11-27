"""Import saved Vox2Pix Static replacements and rerank the VAE training chunks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .freeze_split_chunks import freeze_archive, write_ranked_row
from .full48_source import load_reference
from .outer_chunk_audit import FREQUENCY, pooled_counts
from .outer_chunk_rank import blocks
from .token_anchored_split import make_split


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def import_snapshots(source, archive_index, names, plan, output):
    reference = load_reference()
    name_ids = {name: index for index, name in enumerate(names)}
    samples = []
    (output / "replacements").mkdir()
    with np.load(source, allow_pickle=False) as maps:
        sources = json.loads(maps["metadata"].item())["scenes"]
        raw = {}
        for row in plan:
            scene = row["source_vox2pix_scene"]
            ref = sources[scene]
            seed, origin = row["world_seed"], row["region_origin_enu"]
            if (ref["world_seed"], ref["region_origin_enu"]) != (seed, origin):
                raise ValueError(f"{scene}: the saved map location differs from the plan")
            codes = maps[scene]
            if (
                codes.shape != (48, 48, 48)
                or not np.issubdtype(codes.dtype, np.integer)
                or codes.min() < 0 or codes.max() >= 256 * len(names)
            ):
                raise ValueError(f"{scene}: expected a complete 48³ map with valid source codes")
            raw[scene] = codes
    if len(raw) != len(plan):
        raise ValueError("The plan repeats a source map")

    incoming = {(names[code // 256], code % 256) for cube in raw.values() for code in np.unique(cube).tolist()}
    base_pairs = reference["pairs"]
    token_pairs = [*base_pairs, *sorted(incoming - set(base_pairs))]
    size = len(token_pairs)
    if size > np.iinfo(np.uint16).max + 1:
        raise ValueError("The combined vocabulary does not fit uint16")
    source_code = np.array([256 * name_ids.get(name, -1) + param2 for name, param2 in token_pairs])
    token_for_code = np.full(256 * len(names), -1, dtype=np.int32)
    known = source_code >= 0
    token_for_code[source_code[known]] = np.flatnonzero(known)
    base_count = pooled_counts(FREQUENCY, reference)
    frequency = np.pad(base_count, (0, size - len(base_count)))

    for row in plan:
        scene = row["source_vox2pix_scene"]
        codes = raw[scene]
        ids = token_for_code[codes]
        if ids.min() < 0 or not np.array_equal(source_code[ids], codes):
            raise ValueError(f"{scene}: a voxel does not round-trip through the VAE vocabulary")
        tokens = ids.astype(np.uint16)
        old = row["original_static"]
        group = row["expected_static_type"]["group"]
        stratum = row["expected_static_type"]["stratum"]
        archived = archive_index[old["id"]]
        original = json.loads(archived["metadata"].read_text())
        planned_original = group, stratum, old["world_seed"], old["region_origin_enu"]
        archived_original = original["biome_group"], original["stratum"], original["world_seed"], original["region_origin"]
        if planned_original != archived_original:
            raise ValueError(f"{scene}: the archived original differs from the replacement plan")
        with np.load(archived["snapshot"], allow_pickle=False) as saved:
            removed = saved["tokens"]
        if removed.shape != tokens.shape or removed.dtype != np.uint16 or removed.max() >= len(base_pairs):
            raise ValueError(f"{scene}: invalid tokens for the archived original")
        frequency += np.bincount(tokens.ravel(), minlength=size) - np.bincount(removed.ravel(), minlength=size)

        relative = Path("replacements") / f"{scene}.npz"
        np.savez_compressed(output / relative, tokens=tokens, light=np.zeros(tokens.shape, dtype=np.uint8))
        samples.append({
            "snapshot_id": f"Vox2Pix:{scene}",
            "file": relative.as_posix(),
            "category": "Static",
            "world_seed": row["world_seed"],
            "region_origin_enu": row["region_origin_enu"],
            "biome_group": group,
            "stratum": stratum,
            "original_vae_sample": old["id"],
            "original_vae_archive": str(archived["snapshot"]),
            "vox2pix_split": row["vox2pix_original_split"],
            "source_tick": sources[scene]["first_tick"],
            "light_observed": False,
            "light_fill_value": 0,
        })
    if frequency.min() < 0 or frequency.sum() != base_count.sum():
        raise ValueError("Static replacements produced an invalid full-window voxel inventory")
    return samples, token_pairs, frequency


def materialize(plan_root, map_inventory, output, seed, base_archives=None):
    plan = json.loads((plan_root / "recapture_manifest.json").read_text())["targets"]
    source = plan_root / "archived_vox2pix_reference.npz"
    names = json.loads(map_inventory.read_text())["names"]
    if base_archives is None:
        frozen = plan_root / "frozen_chunks_for_token_split/frozen_chunks.json"
        base_archives = json.loads(frozen.read_text())["source_archives"]
    static_archive, dynamic_archive = (Path(path).resolve() for path in base_archives)
    retired = {row["original_static"]["id"] for row in plan}
    if len(retired) != len(plan):
        raise ValueError("The plan replaces an original snapshot more than once")
    archive_index = json.loads((plan_root / "archive_index.json").read_text())["originals"]
    archives = {ident: {key: plan_root / value for key, value in row.items()} for ident, row in archive_index.items()}
    wanted = {tuple(state) for row in plan for state in row["observed_new_states"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".static-import-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        samples, token_pairs, frequency = import_snapshots(source, archives, names, plan, staging)
        print("Ranking all VAE snapshots using the updated full-window frequency.", flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(freeze_archive, static_archive, "Static", staging / "static.jsonl.gz", frequency, retired),
                pool.submit(freeze_archive, dynamic_archive, "Dynamic", staging / "dynamic.jsonl.gz", frequency, set()),
            ]
            with gzip.open(staging / "replacement.jsonl.gz", "wt", compresslevel=1) as index:
                for sample in samples:
                    with np.load(staging / sample["file"], allow_pickle=False) as data:
                        frame = blocks(data["tokens"])
                    sample["block_indices"] = write_ranked_row(
                        index, sample["snapshot_id"], sample["world_seed"],
                        ["Static", sample["biome_group"], sample["stratum"]],
                        sample["region_origin_enu"], frame, frequency,
                    )
            for job in jobs:
                job.result()
        manifest = {
            "files": ["static.jsonl.gz", "dynamic.jsonl.gz", "replacement.jsonl.gz"],
            "token_pairs": token_pairs,
            "full48_counts": frequency.tolist(),
            "replacement_projection": False,
            "replacement_source": "archived_vox2pix_map",
            "block_index": "0-based ENU C-order",
            "source_archives": [str(static_archive), str(dynamic_archive)],
            "selection_rule": {
                "Static": "center + two rarest outer blocks",
                "Dynamic": "center + motion-ranked outer block + rarest remaining outer block",
            },
        }
        write_json(staging / "frozen_chunks.json", manifest)
        write_json(staging / "vocabulary.json", {
            "K": len(token_pairs),
            "tokens": [{"id": i, "node_name": name, "param2": param2} for i, (name, param2) in enumerate(token_pairs)],
        })
        make_split(staging / "frozen_chunks.json", staging / "split", seed)
        with gzip.open(staging / "split/world_seed_splits.csv.gz", "rt") as stream:
            splits = {int(row["world_seed"]): row["split"] for row in csv.DictReader(stream)}
        with (staging / "split/token_voxel_counts_by_split.csv").open() as stream:
            train = {(row["node_name"], int(row["raw_param2"])) for row in csv.DictReader(stream) if int(row["train"])}
        for sample in samples:
            sample["vae_split"] = splits[sample["world_seed"]]
        write_json(staging / "dataset_overlay.json", {
            "source": str(source),
            "base_archives": manifest["source_archives"],
            "replacement_count": len(samples),
            "selection_rule": manifest["selection_rule"],
            "target_states_covered": len(wanted & train),
            "target_states_missing_from_train": sorted(wanted - train),
            "samples": samples,
        })
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--map-inventory", type=Path, required=True)
    parser.add_argument("--base-archives", type=Path, nargs=2, metavar=("STATIC", "DYNAMIC"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    materialize(args.plan_root.resolve(), args.map_inventory.resolve(), args.output.resolve(), args.seed, args.base_archives)


if __name__ == "__main__":
    main()
