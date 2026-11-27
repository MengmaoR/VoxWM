"""Plan a Static overlap repair using the published VAE snapshot selections."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import numpy as np

from vox2pix_data.static_replacement import rank

from .chunk_selection import extract_selected_chunks, iter_chunk_selections
from .outer_chunk_rank import CENTER_INDEX, blocks, mixed_outer_indices
from .scene_overlap_geometry import summarize
from .token_anchored_split import SPLITS, snapshots


def load(path):
    return json.loads(path.read_text())


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def static_type(row):
    return row["group"], row["stratum"], row["biome_transition"]


def selection_header(path):
    with gzip.open(path, "rt") as stream:
        return json.loads(next(stream))


def selected_counts(bundle, index, frozen, size):
    """Yield each saved row and a sparse histogram of its three selected blocks."""
    header = selection_header(index)
    for source, row in zip(snapshots([bundle / name for name in frozen["files"]]),
                           iter_chunk_selections(index), strict=True):
        if row["snapshot_id"] != source["scene"]:
            raise ValueError("Frozen histograms and the selected snapshot list disagree")
        if row["chunk_indices"] == [block["index"] for block in source["blocks"]]:
            yield row, [pair for block in source["blocks"] for pair in block["counts"]]
        else:
            path = index.parent / header["replacement_files"][row["snapshot_id"]]
            with np.load(path, allow_pickle=False) as data:
                selected = extract_selected_chunks(data["tokens"], row)
            count = np.bincount(selected.ravel(), minlength=size)
            present = np.flatnonzero(count)
            yield row, list(zip(present.tolist(), count[present].tolist(), strict=True))


def plan_repair(bundle, preflight, static_catalog, output):
    index = bundle / "chunk_selection.jsonl.gz"
    frozen = load(bundle / "frozen_chunks.json")
    frequency = np.asarray(frozen["full48_counts"], dtype=np.int64)
    audit = load(preflight)
    if Path(audit["source_selection"]).resolve() != index.resolve():
        raise ValueError("The overlap audit refers to a different selection file")
    targets = audit["candidates"]
    endpoints = {entry[side]["snapshot_id"] for entry in targets
                 for side in ("original_static", "paired_static_to_keep")}
    retired = {entry["original_static"]["snapshot_id"] for entry in targets}
    if len(endpoints) != 2 * len(targets) or len(retired) != len(targets):
        raise ValueError("Expected disjoint Static overlap pairs")
    to_train = sorted((entry[side] for entry in targets if entry["split"] != "train"
                       for side in ("original_static", "paired_static_to_keep")),
                      key=lambda row: (row["split"], row["snapshot_id"]))
    needed = Counter(static_type(row) for row in to_train)
    static_rows = {}
    used_seeds = set()
    for row in iter_chunk_selections(index):
        used_seeds.add(row["world_seed"])
        if row["category"] == "Static":
            static_rows[row["snapshot_id"]] = row
    for row in to_train:
        if static_rows[row["snapshot_id"]]["split"] != row["split"]:
            raise ValueError("A held-out endpoint changed splits since the audit")
    donors = defaultdict(list)
    with gzip.open(static_catalog, "rt") as stream:
        for line in stream:
            ident, group, stratum, seed, origin, transition, clip, _, _, hist = json.loads(line)
            used_seeds.add(seed)
            ident = f"vae_100k_static:{ident}"
            cell = group, stratum, bool(transition)
            source = static_rows.get(ident)
            if cell not in needed or source is None or source["split"] != "train" or ident in endpoints:
                continue
            if (source["world_seed"], source["region_origin_enu"]) != (seed, origin):
                raise ValueError(f"Static catalog coordinates differ for {ident}")
            rarity = rank({int(i): n for i, n in hist.items()}, frequency)
            donor = dict(source, group=group, stratum=stratum, biome_transition=bool(transition),
                         original_clip=clip)
            donors[cell].append((rarity, ident, donor))
    for cell, n in needed.items():
        donors[cell].sort(reverse=True)
        if len(donors[cell]) < n:
            raise ValueError(f"Insufficient matching Train donors for {cell}")
    swaps = [{"to_train": row, "to_holdout": donors[static_type(row)].pop(0)[2],
              "holdout_split": row["split"]} for row in to_train]
    changed = {row["to_train"]["snapshot_id"]: "train" for row in swaps}
    changed.update({row["to_holdout"]["snapshot_id"]: row["holdout_split"] for row in swaps})
    if len(changed) != 2 * len(swaps):
        raise ValueError("A split swap reuses a snapshot")

    count_before = np.zeros(len(frequency), dtype=np.int64)
    count_after = np.zeros(len(frequency), dtype=np.int64)
    for row, hist in selected_counts(bundle, index, frozen, len(frequency)):
        ident = row["snapshot_id"]
        keep_in_train = ident not in retired and changed.get(ident, row["split"]) == "train"
        for token, n in hist:
            if row["split"] == "train":
                count_before[token] += n
            if keep_in_train:
                count_after[token] += n
    missing = np.flatnonzero((count_before > 0) & (count_after == 0))
    if missing.size:
        raise ValueError(f"Train loses states after swaps and retirement: {[frozen['token_pairs'][i] for i in missing]}")
    if int(count_before.sum() - count_after.sum()) != len(retired) * 3 * 16**3:
        raise ValueError("The split swaps or retirement changed the planned Train size")
    report = {"source_bundle": str(bundle.resolve()), "targets": targets, "split_swaps": swaps,
              "used_world_seeds": sorted(used_seeds), "train_types_before": int((count_before > 0).sum()),
              "train_types_after_swaps_and_removals": int((count_after > 0).sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"Planned {len(swaps)} split swaps and {len(retired)} fresh Static captures; all Train states retained.", flush=True)


def load_replacements(plan, frozen, captures, archive):
    from vae_data.sharded.protocol import Frame, Vocabulary
    from vae_data.sharded.static_height import validate_static
    from vae_data.sharded.validate import validate_native

    old_pairs = [tuple(pair) for pair in frozen["token_pairs"]]
    archived = load(archive / "archive.json")["original_samples"]
    used_seeds = set(plan["used_world_seeds"])
    added_pairs = set()
    accepted = []
    for target in plan["targets"]:
        original = target["original_static"]
        folder = captures / original["id"].replace(":", "_")
        metadata = load(folder / "metadata.json")
        vocabulary = Vocabulary(folder / "vocabulary.json")
        with np.load(folder / "snapshot.npz", allow_pickle=False) as data:
            frame = Frame(data["tokens"], data["light"], metadata)
        validate_native(frame)
        validate_static(frame, vocabulary)
        seed = metadata["world_seed"]
        captured_type = metadata["biome_group"], metadata["stratum"], metadata["biome_transition"]
        if (
            metadata["category"] != "Static" or metadata["original_static_id"] != original["id"]
            or captured_type != static_type(original) or seed in used_seeds
        ):
            raise ValueError(f"Capture does not match the planned Static replacement: {folder}")
        used_seeds.add(seed)
        added_pairs.update(vocabulary.entries[int(i)] for i in np.unique(frame.tokens))
        backup = archived[original["id"]]
        with np.load(archive / backup["snapshot"], allow_pickle=False) as data:
            old_tokens = data["tokens"]
        if old_tokens.shape != frame.tokens.shape or old_tokens.dtype != np.uint16 or old_tokens.max() >= len(old_pairs):
            raise ValueError(f"Invalid archived original {original['id']}")
        accepted.append((original, frame, vocabulary.entries, archive / backup["snapshot"], old_tokens))

    pairs = [*old_pairs, *sorted(added_pairs - set(old_pairs))]
    if len(pairs) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Combined tokens no longer fit uint16")
    ids = {pair: index for index, pair in enumerate(pairs)}
    frequency = np.pad(np.asarray(frozen["full48_counts"], dtype=np.int64), (0, len(pairs) - len(old_pairs)))
    replacements = {}
    for original, frame, local_pairs, backup_path, old_tokens in accepted:
        mapping = np.array([ids.get(pair, -1) for pair in local_pairs], dtype=np.int32)
        mapped = mapping[frame.tokens]
        if mapped.min() < 0:
            raise ValueError(f"Unresolved captured token in {original['id']}")
        tokens = mapped.astype(np.uint16)
        frequency += np.bincount(tokens.ravel(), minlength=len(pairs)) - np.bincount(old_tokens.ravel(), minlength=len(pairs))
        replacements[original["snapshot_id"]] = (tokens, frame.light, frame.meta, backup_path)
    if frequency.min() < 0 or frequency.sum() != sum(frozen["full48_counts"]):
        raise ValueError("Replacing Static maps produced an invalid full-cube frequency")
    return replacements, pairs, frequency


def publish_repair(plan_path, captures, archive, inventory_dir, output):
    plan = load(plan_path)
    bundle = Path(plan["source_bundle"])
    index = bundle / "chunk_selection.jsonl.gz"
    header = selection_header(index)
    frozen = load(bundle / "frozen_chunks.json")
    overlay = load(bundle / "dataset_overlay.json")
    replacements, pairs, frequency = load_replacements(plan, frozen, captures, archive)
    swaps = {row["to_train"]["snapshot_id"]: "train" for row in plan["split_swaps"]}
    for row in plan["split_swaps"]:
        if static_type(row["to_train"]) != static_type(row["to_holdout"]):
            raise ValueError("Mismatched Static type in split swap")
        swaps[row["to_holdout"]["snapshot_id"]] = row["holdout_split"]
    counts = {split: np.zeros(len(pairs), dtype=np.int64) for split in SPLITS}
    before = np.zeros(len(pairs), dtype=np.int64)
    sizes_before = Counter()
    sizes_after = Counter()
    static = []
    dynamic = {}
    replacement_rows = {}
    replacement_histograms = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".overlap-repair-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        (stage / "replacements").mkdir(parents=True)
        for ident, path in header["replacement_files"].items():
            new_path = Path("replacements") / Path(path).name
            copy2(index.parent / path, stage / new_path)
            header["replacement_files"][ident] = new_path.as_posix()
        header["source_archives"] = {name: os.path.relpath(index.parent / path, output)
                                     for name, path in header["source_archives"].items()}
        header["vocabulary"] = "vocabulary.json"
        header["split_unit"] = "snapshot"
        for ident, (tokens, light, metadata, backup_path) in replacements.items():
            original_id = metadata["original_static_id"]
            name = original_id.replace(":", "_")
            snapshot_id = f"StaticResample:{name}"
            file = Path("replacements") / f"static_resample_{name}.npz"
            np.savez_compressed(stage / file, tokens=tokens, light=light)
            write_json(stage / file.with_suffix(".json"), metadata)
            frame = blocks(tokens)
            slots = sorted([CENTER_INDEX, *mixed_outer_indices(frame, frequency, limits=(2,))[2]])
            count = np.bincount(frame[slots].ravel(), minlength=len(pairs))
            replacement_histograms[ident] = [(int(i), int(count[i])) for i in np.flatnonzero(count)]
            row = {"snapshot_id": snapshot_id, "category": "Static", "world_seed": metadata["world_seed"],
                   "split": "train", "chunk_indices": slots, "region_origin_enu": metadata["region_origin"]}
            replacement_rows[ident] = row
            header["replacement_files"][snapshot_id] = file.as_posix()
            overlay["samples"].append(dict(row, file=file.as_posix(), vae_split="train", light_observed=True,
                                           original_vae_sample=original_id, original_vae_archive=str(backup_path),
                                           biome_group=metadata["biome_group"], stratum=metadata["stratum"],
                                           biome_transition=metadata["biome_transition"]))

        seen = set()
        seen_changes = set()
        with gzip.open(stage / index.name, "wt", compresslevel=6) as stream:
            stream.write(json.dumps(header, separators=(",", ":")) + "\n")
            for row, hist in selected_counts(bundle, index, frozen, len(pairs)):
                ident = row["snapshot_id"]
                sizes_before[(row["split"], row["category"])] += 1
                if row["split"] == "train":
                    for token, n in hist:
                        before[token] += n
                row["split"] = swaps.get(ident, row["split"])
                if ident in swaps or ident in replacement_rows:
                    seen_changes.add(ident)
                if ident in replacement_rows:
                    if row["split"] != "train" or row["category"] != "Static":
                        raise ValueError(f"Retired sample was not in Static Train after swaps: {ident}")
                    row = replacement_rows[ident]
                    hist = replacement_histograms[ident]
                if row["snapshot_id"] in seen:
                    raise ValueError(f"Duplicate final snapshot {row['snapshot_id']}")
                seen.add(row["snapshot_id"])
                sizes_after[(row["split"], row["category"])] += 1
                tally = counts[row["split"]]
                for token, n in hist:
                    tally[token] += n
                center = tuple(x + 24 for x in row["region_origin_enu"])
                scene = (row["world_seed"], center, row["category"], "", row["snapshot_id"])
                if row["category"] == "Static":
                    static.append(scene)
                else:
                    key = (row["world_seed"], center, row["category"], row["snapshot_id"].split(":")[1])
                    first, n = dynamic.get(key, (scene, 0))
                    dynamic[key] = first, n + 1
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

        if sizes_before != sizes_after or seen_changes != (swaps.keys() | replacement_rows.keys()):
            raise ValueError("The final selection is missing a swap or changed a split quota")
        if any(n != 5 for _, n in dynamic.values()):
            raise ValueError("An existing dynamic chain is incomplete")
        dynamic_scenes = [scene for scene, _ in dynamic.values()]
        overlaps = {"Static": summarize(static)["overlap_pairs"], "Dynamic": summarize(dynamic_scenes)["overlap_pairs"],
                    "Static_vs_Dynamic": summarize(static + dynamic_scenes, cross=True)["overlap_pairs"]}
        if any(overlaps.values()):
            raise ValueError(f"The final dataset still has overlapping scenes: {overlaps}")
        if np.any((before > 0) & (counts["train"] == 0)):
            raise ValueError("A previously covered Train state was lost")
        train = {pair for pair, n in zip(pairs, counts["train"], strict=True) if n}
        coverage = {split: {pair for pair, n in zip(pairs, counts[split], strict=True) if n}
                    for split in SPLITS if split != "train"}
        for layer in ("scene", "surface"):
            inventory = load(inventory_dir / f"{layer}.json")
            coverage[f"vox2pix_{layer}"] = {(inventory["names"][int(i)//256], int(i)%256)
                                             for i, n in inventory["counts"].items() if n}
        for name, states in coverage.items():
            if states - train:
                raise ValueError(f"Train fails required coverage for {name}: {sorted(states - train)}")
        report = {"split_swaps": len(plan["split_swaps"]), "new_native_snapshots": len(replacement_rows),
                  "overlaps": overlaps, "train_types": len(train),
                  "coverage": {name: len(states) for name, states in coverage.items()},
                  "split_snapshots": {split: sum(n for (s, _), n in sizes_after.items() if s == split) for split in SPLITS},
                  "split_chunks": {split: int(counts[split].sum()) // 16**3 for split in SPLITS}}
        write_json(stage / "verification.json", report)
        write_json(stage / "vocabulary.json", {"K": len(pairs),
                   "tokens": [{"id": i, "node_name": name, "param2": param2} for i, (name, param2) in enumerate(pairs)]})
        overlay["target_states_covered"] += len(overlay["target_states_missing_from_train"])
        overlay.update(replacement_count=len(overlay["samples"]), target_states_missing_from_train=[],
                       split_swaps=plan["split_swaps"], overlap_repair=dict(archive=str(archive), plan=str(plan_path)))
        write_json(stage / "dataset_overlay.json", overlay)
        stage.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--bundle", type=Path, required=True)
    plan.add_argument("--preflight", type=Path, required=True)
    plan.add_argument("--static-catalog", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--captures", type=Path, required=True)
    publish.add_argument("--archive", type=Path, required=True)
    publish.add_argument("--vox2pix-inventory", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    if args.command == "plan":
        plan_repair(args.bundle, args.preflight, args.static_catalog, args.output)
    else:
        publish_repair(args.plan.resolve(), args.captures.resolve(), args.archive.resolve(),
                       args.vox2pix_inventory.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
