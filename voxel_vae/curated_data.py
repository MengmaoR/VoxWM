"""Materialize the three published 16³ chunks for each frozen VAE snapshot."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np

from .change_masks import compute_chain_masks
from .chunk_selection import FORMAT, extract_selected_chunks
from .dataset import SampleRef, VALID_CATEGORIES
from .full48_scan import _tokens_array
from .full48_source import SPLIT_COUNTS, iter_archive_shards, validate_canonical
from .outer_chunk_audit import chain_indices
from .vocabulary import CanonicalVocabulary, sha256_file


PACK_FORMAT = "voxwm-vae-curated-chunks-v1"
CHUNK_SHAPE = (16, 16, 16)
CHUNK_VOLUME = 16 ** 3
ARCHIVES = ("vae_100k_static", "vae_100k_dynamic")


def _read_selection(path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (header.get("format"), header.get("chunk_edge"), header.get("region_edge")) != (FORMAT, 16, 48):
            raise ValueError("Expected the published 48³ → 16³ chunk-selection file")
        overrides = header["overrides"]
        for line in stream:
            row = json.loads(line)
            name, slots = row["snapshot_id"], row["chunk_indices"]
            if (name in rows or len(slots) != 3 or len(set(slots)) != 3
                    or any(type(slot) is not int or not 0 <= slot < 27 for slot in slots)
                    or (slots != overrides[name] if name in overrides else 13 not in slots)):
                raise ValueError(f"Invalid fixed chunk selection for {name}")
            if row["split"] not in SPLIT_COUNTS or row["category"] not in VALID_CATEGORIES:
                raise ValueError(f"Invalid split or category for {name}")
            rows[name] = row
    if set(overrides) - rows.keys() or any(
        (rows[name]["split"], rows[name]["category"]) != ("train", "Static") for name in overrides
    ):
        raise ValueError("The explicit center exceptions must belong to Static Train")
    if Counter(row["split"] for row in rows.values()) != SPLIT_COUNTS:
        raise ValueError("The fixed list does not contain the published split sizes")
    return header, rows


def _frozen_vocabulary(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = CanonicalVocabulary(
        ((row["node_name"], row["param2"]) for row in raw["tokens"]), preserve_ids=True)
    if raw["K"] != vocabulary.K or raw["tokens"] != vocabulary.semantic_payload():
        raise ValueError("Invalid exact token IDs in the published vocabulary")
    return vocabulary


def _journal(files):
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(files["journal.sqlite"])
        rows = [(index, category, json.loads(raw)) for index, category, raw in db.execute(
            "SELECT idx,category,metadata FROM samples ORDER BY idx")]
        subtypes = {name: json.loads(raw).get("subtype") or None
                    for name, raw in db.execute("SELECT id,data FROM chains")}
    return rows, subtypes


def _chunks(voxels, row, k):
    chosen = extract_selected_chunks(voxels, row)
    if chosen.dtype != np.uint16 or int(chosen.max()) >= k:
        raise ValueError(f"Invalid canonical token in {row['snapshot_id']}")
    return chosen


def _chain_masks(states, rows):
    first = rows[0]
    fields = ("category", "split", "world_seed", "region_origin_enu", "chunk_indices")
    if any(any(row[key] != first[key] for key in fields) for row in rows[1:]):
        raise ValueError(f"Dynamic chain changes split or spatial selection: {first['snapshot_id']}")
    return np.stack([compute_chain_masks(states[:, slot], allow_stagnant=True)
                     for slot in range(3)], axis=1)


def prepare(selection, output):
    selection, output = Path(selection).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Prepared output already exists: {output}")
    header, pending = _read_selection(selection)
    parent = selection.parent
    vocabulary = _frozen_vocabulary(parent / header["vocabulary"])
    source_paths = [(parent / header["source_archives"][name]).resolve() for name in ARCHIVES]
    replacements = {row["snapshot_id"]: row for row in
                    json.loads((parent / "dataset_overlay.json").read_text())["samples"]}
    if replacements.keys() != header["replacement_files"].keys():
        raise ValueError("Replacement metadata and fixed list refer to different snapshots")
    seen = {split: np.zeros(vocabulary.K, dtype=bool) for split in SPLIT_COUNTS}
    categories = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".curated_chunks_", dir=output.parent) as temporary:
        stage = Path(temporary)
        count = 3 * len(pending)
        tokens = np.lib.format.open_memmap(stage / "tokens.npy", mode="w+", dtype=np.uint16,
                                          shape=(count, *CHUNK_SHAPE))
        packed = np.lib.format.open_memmap(stage / "changed.npy", mode="w+", dtype=np.uint8,
                                          shape=(count, CHUNK_VOLUME // 8))
        offset = 0
        with gzip.open(stage / "snapshots.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
            def store(row, chunks, meta, root, shard, changed=None, subtype=None):
                nonlocal offset
                if (row["category"], row["world_seed"], row["region_origin_enu"]) != (
                        meta["category"], meta["world_seed"],
                        meta.get("region_origin", meta.get("region_origin_enu"))):
                    raise ValueError(f"Source and fixed list differ for {row['snapshot_id']}")
                batch = slice(offset, offset + 3)
                tokens[batch] = chunks
                if changed is not None:
                    packed[batch] = np.packbits(changed.reshape(3, -1), axis=1)
                seen[row["split"]][chunks.ravel()] = True
                categories[row["category"]] += 1
                record = dict(row, root=root, shard=shard, chain_id=meta.get("chain_id"),
                              chain_index=meta.get("chain_index"), biome_group=meta.get("biome_group"),
                              stratum=meta.get("stratum"), d4_subtype=subtype)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                offset += 3

            for root, (name, source) in enumerate(zip(ARCHIVES, source_paths, strict=True)):
                for number, item in enumerate(iter_archive_shards(
                    source, select_file=lambda relative: relative in {"READY", "journal.sqlite"}
                    or relative.startswith("snapshots.zarr/tokens/")
                ), start=1):
                    if number == 1:
                        base = validate_canonical(item.header["json"]["vocabulary.json"])
                        if vocabulary.pairs[:base["K"]] != base["pairs"] or item.root_name != name:
                            raise ValueError("Archive canonical IDs must be the frozen vocabulary prefix")
                    size = json.loads(item.files["READY"])["samples"]
                    array = _tokens_array(item.files, "snapshots", size, item.shard_name)
                    journal, subtypes = _journal(item.files)
                    if [i for i, _, _ in journal] != list(range(size)):
                        raise ValueError(f"Incomplete shard journal: {item.shard_name}")
                    prefix = f"{name}:{item.shard_name}"
                    shard = int(item.shard_name.removeprefix("shard_"))
                    if root == 0:
                        for i, _, meta in journal:
                            row = pending.pop(f"{prefix}:{i}", None)
                            if row is not None:
                                store(row, _chunks(array[i], row, base["K"]), meta, root, shard)
                    else:
                        for indices in chain_indices(item.files, size):
                            rows = [pending.pop(f"{prefix}:{i}") for i in indices]
                            states = np.stack([_chunks(array[i], row, base["K"])
                                               for i, row in zip(indices, rows, strict=True)])
                            masks = _chain_masks(states, rows)
                            chain_id = journal[indices[0]][2]["chain_id"]
                            for position, i in enumerate(indices):
                                meta = journal[i][2]
                                if meta["chain_id"] != chain_id or meta["chain_index"] != position:
                                    raise ValueError(f"Unexpected journal chain order: {prefix}:{i}")
                                store(rows[position], states[position], meta, root, shard,
                                      changed=masks[position], subtype=subtypes[chain_id])
                    if number % 25 == 0:
                        print(f"{name}: {number} shards, {offset:,}/{count:,} chunk frames", flush=True)

            for name, filename in header["replacement_files"].items():
                row = pending.pop(name)
                with np.load(parent / filename, allow_pickle=False) as data:
                    chosen = _chunks(data["tokens"], row, vocabulary.K)
                store(row, chosen, replacements[name], 2, 0)

        expected_categories = dict(Static=100_000, D1=25_000, D2=25_000, D3=25_000, D4=25_000)
        if pending or offset != count or categories != expected_categories:
            raise ValueError(f"Incomplete published dataset: {offset:,} chunk frames, {dict(categories)}")
        for split in ("validation", "test"):
            missing = np.flatnonzero(seen[split] & ~seen["train"])
            if missing.size:
                raise ValueError(f"Train lacks {split} canonical IDs {missing.tolist()}")
        tokens.flush()
        packed.flush()
        vocabulary.save(stage / "training_vocabulary.json")
        manifest = {
            "format": PACK_FORMAT, "source_names": [*map(str, source_paths), str(selection)],
            "chunk_frames": count, "by_snapshot_category": dict(categories),
            "chunk_frames_by_split": {split: 3 * size for split, size in SPLIT_COUNTS.items()},
            "token_types_by_split": {split: int(flags.sum()) for split, flags in seen.items()},
            "train_token_ids": np.flatnonzero(seen["train"]).tolist(),
            "vocabulary_sha256": vocabulary.sha256,
            "selection_sha256": sha256_file(selection),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "train_token_ids"}),
          flush=True)


def read_manifest(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    vocabulary = CanonicalVocabulary.load(path / "training_vocabulary.json")
    if manifest["format"] != PACK_FORMAT or vocabulary.sha256 != manifest["vocabulary_sha256"]:
        raise ValueError("Prepared chunk format or vocabulary mismatch")
    return manifest, vocabulary


class PreparedChunks:
    """Read fixed canonical tokens and their packed changed-voxel masks by sample index."""

    def __init__(self, path):
        path = Path(path).resolve()
        self.manifest, self.vocabulary = read_manifest(path)
        self.roots = tuple(self.manifest["source_names"])
        self.manifest_hashes = (sha256_file(path / "manifest.json"),)
        self.read_path = str(path / "snapshots.jsonl.gz")
        self.records = []
        chain_ids = {}
        with gzip.open(self.read_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for slot in row["chunk_indices"]:
                    chain = row["chain_id"]
                    if chain is not None:
                        key = (chain, slot)
                        if key not in chain_ids:
                            chain_ids[key] = f"{chain}/chunk{slot:02d}"
                        chain = chain_ids[key]
                    self.records.append(SampleRef(
                        root_index=row["root"], shard=row["shard"], shard_path=self.read_path,
                        snapshot_index=len(self.records), category=row["category"], chain_id=chain,
                        chain_index=row["chain_index"], world_seed=row["world_seed"], split=row["split"],
                        biome_group=row["biome_group"], stratum=row["stratum"], d4_subtype=row["d4_subtype"]))
        self.tokens = np.load(path / "tokens.npy", mmap_mode="r")
        self.masks = np.load(path / "changed.npy", mmap_mode="r")
        n = len(self.records)
        if (n != self.manifest["chunk_frames"] or self.tokens.shape != (n, *CHUNK_SHAPE)
                or self.tokens.dtype != np.uint16 or self.masks.shape != (n, CHUNK_VOLUME // 8)
                or self.masks.dtype != np.uint8):
            raise ValueError("Prepared array shapes or dtypes do not match the snapshot records")

    def indices(self, split=None, categories=None):
        return [i for i, row in enumerate(self.records)
                if (split is None or row.split == split)
                and (categories is None or row.category in categories)]

    def read_numpy(self, sample_index):
        return self.tokens[sample_index].astype(np.int64), sample_index

    def __getitem__(self, sample_index):
        return self.changed(sample_index)

    def changed(self, sample_index):
        return np.unpackbits(self.masks[sample_index]).view(np.bool_).reshape(CHUNK_SHAPE)

    def statistics(self, indices):
        groups = defaultdict(list)
        for item in indices:
            category = self.records[item].category
            if category != "Static":
                groups[category].append(int(self.changed(item).sum()))
        return {"by_category": {
            category: {"states": len(counts), "changed_voxels": sum(counts),
                       "empty_changed_states": counts.count(0),
                       "empty_unchanged_states": counts.count(CHUNK_VOLUME)}
            for category, counts in sorted(groups.items())
        }}
