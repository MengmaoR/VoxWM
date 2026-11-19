"""Read committed VAE snapshots without touching raw datasets or D4 anchors."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr

from .vocabulary import CanonicalVocabulary, mapping_array

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Index/vocabulary tooling remains usable in the CPU collection venv.
    torch = None

    class Dataset:  # pragma: no cover - exercised only in environments without Torch.
        pass


CENTER = (slice(16, 32),) * 3
VALID_SPLITS = ("train", "validation", "test")
VALID_CATEGORIES = ("Static", "D1", "D2", "D3", "D4")


@dataclass(frozen=True)
class SampleRef:
    root_index: int
    shard: int
    shard_path: str
    snapshot_index: int
    category: str
    chain_id: str | None
    chain_index: int | None
    world_seed: int
    split: str
    biome_group: str | None
    stratum: str | None
    d4_subtype: str | None

    @property
    def key(self):
        return f"{self.root_index}:{self.shard:04d}:{self.snapshot_index}"


class DatasetIndex:
    VERSION = 1

    def __init__(self, roots, records, manifest_hashes, vocabulary_hash):
        self.roots = tuple(map(str, roots))
        self.records = tuple(records)
        self.manifest_hashes = tuple(manifest_hashes)
        self.vocabulary_hash = str(vocabulary_hash)
        self._validate()

    def _validate(self):
        if len(self.roots) != len(self.manifest_hashes):
            raise ValueError("Dataset roots/manifest hashes length mismatch")
        seed_split = {}
        keys = set()
        for record in self.records:
            if not 0 <= record.root_index < len(self.roots):
                raise ValueError("Sample root index is out of range")
            if record.key in keys:
                raise ValueError(f"Duplicate indexed sample: {record.key}")
            keys.add(record.key)
            if record.split not in VALID_SPLITS:
                raise ValueError(f"Unknown split: {record.split}")
            if record.category not in VALID_CATEGORIES:
                raise ValueError(f"Unknown category: {record.category}")
            previous = seed_split.setdefault(record.world_seed, record.split)
            if previous != record.split:
                raise ValueError(
                    f"World seed {record.world_seed} crosses {previous}/{record.split}")

    def indices(self, split=None, categories=None):
        categories = set(categories) if categories is not None else None
        return [index for index, row in enumerate(self.records)
                if (split is None or row.split == split)
                and (categories is None or row.category in categories)]

    def summary(self):
        by_split = Counter(row.split for row in self.records)
        by_category = Counter(row.category for row in self.records)
        by_root = Counter(row.root_index for row in self.records)
        split_category = Counter((row.split, row.category) for row in self.records)
        return {
            "samples": len(self.records),
            "roots": {self.roots[index]: by_root[index] for index in range(len(self.roots))},
            "splits": dict(sorted(by_split.items())),
            "categories": dict(sorted(by_category.items())),
            "split_categories": {
                split: {category: split_category[split, category]
                        for category in VALID_CATEGORIES}
                for split in VALID_SPLITS
            },
            "unique_world_seeds": len({row.world_seed for row in self.records}),
        }

    def to_dict(self):
        return {
            "version": self.VERSION,
            "roots": list(self.roots),
            "manifest_hashes": list(self.manifest_hashes),
            "canonical_vocabulary_sha256": self.vocabulary_hash,
            "summary": self.summary(),
            "samples": [asdict(row) for row in self.records],
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), separators=(",", ":"),
                                        ensure_ascii=False) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        if raw.get("version") != cls.VERSION:
            raise ValueError("Unsupported dataset index version")
        return cls(raw["roots"], [SampleRef(**row) for row in raw["samples"]],
                   raw["manifest_hashes"], raw["canonical_vocabulary_sha256"])


def _shard_directories(root):
    shards = Path(root) / "shards"
    if not shards.is_dir():
        raise ValueError(f"Missing shards directory: {shards}")
    selected = {}
    for path in shards.iterdir():
        name = path.name
        if path.is_dir() and name.startswith("shard_") and len(name) == 10:
            selected[int(name[-4:])] = path
    for path in shards.iterdir():
        name = path.name
        if path.is_dir() and name.startswith(".shard_") and name.endswith(".partial"):
            shard = int(name[7:11])
            selected.setdefault(shard, path)
    return [selected[key] for key in sorted(selected)]


def _shard_number(shard_path):
    name = Path(shard_path).name
    if name.startswith("shard_") and len(name) == 10:
        return int(name[-4:])
    if name.startswith(".shard_") and name.endswith(".partial"):
        return int(name[7:11])
    raise ValueError(f"Unrecognized shard directory: {shard_path}")


def _read_shard(root_index, shard_path):
    database = shard_path / "journal.sqlite"
    if not database.is_file():
        raise ValueError(f"Missing authoritative journal: {database}")
    shard = _shard_number(shard_path)
    manifest = shard_path / "manifest.json"
    if manifest.is_file():
        recorded_shard = int(json.loads(manifest.read_text())["shard"])
        if recorded_shard != shard:
            raise ValueError(f"Shard manifest/directory mismatch: {shard_path}")
    db = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        chain_subtypes = {}
        chain_indices = {}
        for (raw,) in db.execute("SELECT data FROM chains"):
            chain = json.loads(raw)
            chain_id = str(chain["id"])
            indices = list(map(int, chain["snapshot_indices"]))
            if len(indices) != 5:
                raise ValueError(f"Dynamic chain is not five states: {chain_id}")
            chain_subtypes[chain_id] = chain.get("subtype") or None
            chain_indices[chain_id] = indices
        records = []
        for snapshot_index, category, raw in db.execute(
                "SELECT idx,category,metadata FROM samples ORDER BY idx"):
            if int(snapshot_index) != len(records):
                raise ValueError(f"Non-contiguous committed snapshot IDs in {database}")
            metadata = json.loads(raw)
            if int(metadata.get("snapshot_id", -1)) != int(snapshot_index):
                raise ValueError(f"Snapshot index mismatch in {database}")
            if metadata.get("category") != category:
                raise ValueError(f"Category mismatch in {database}")
            chain_id = metadata.get("chain_id")
            chain_index = metadata.get("chain_index")
            if category == "Static":
                if chain_id is not None or chain_index is not None:
                    raise ValueError("Static row unexpectedly belongs to a chain")
            else:
                if chain_id not in chain_indices:
                    raise ValueError(f"Missing dynamic chain {chain_id}")
                expected = chain_indices[chain_id]
                if not 0 <= int(chain_index) < 5 or expected[int(chain_index)] != snapshot_index:
                    raise ValueError(f"Broken dynamic chain order: {chain_id}")
            records.append(SampleRef(
                root_index=int(root_index),
                shard=shard,
                shard_path=str(shard_path.resolve()), snapshot_index=int(snapshot_index),
                category=str(category), chain_id=chain_id,
                chain_index=None if chain_index is None else int(chain_index),
                world_seed=int(metadata["world_seed"]), split=str(metadata["split"]),
                biome_group=metadata.get("biome_group"), stratum=metadata.get("stratum"),
                d4_subtype=chain_subtypes.get(chain_id) if category == "D4" else None,
            ))
        arrays = zarr.open_group(str(shard_path / "snapshots.zarr"), mode="r")
        committed = len(records)
        if "tokens" not in arrays or len(arrays["tokens"]) < committed:
            raise ValueError(f"Committed Zarr rows are missing in {shard_path}")
        return records
    finally:
        db.close()


def build_dataset_index(vocabulary):
    if not isinstance(vocabulary, CanonicalVocabulary):
        raise TypeError("Expected CanonicalVocabulary")
    records = []
    for root_index, root_record in enumerate(vocabulary.roots):
        root = Path(root_record.root)
        for shard_path in _shard_directories(root):
            records.extend(_read_shard(root_index, shard_path))
    records.sort(key=lambda row: (row.root_index, row.shard, row.snapshot_index))
    return DatasetIndex([root.root for root in vocabulary.roots], records,
                        [root.manifest_sha256 for root in vocabulary.roots],
                        vocabulary.sha256)


def crop_center(tokens):
    array = np.asarray(tokens)
    if array.shape != (48, 48, 48):
        raise ValueError(f"Expected [48,48,48], got {array.shape}")
    return array[CENTER]


class VoxelSnapshotDataset(Dataset):
    """Returns only canonical center-token IDs and an index; never opens light."""

    def __init__(self, index, vocabulary, sample_indices=None):
        if index.vocabulary_hash != vocabulary.sha256:
            raise ValueError("Dataset index/canonical vocabulary mismatch")
        self.index = index
        self.vocabulary = vocabulary
        self.sample_indices = tuple(index.indices() if sample_indices is None else
                                    map(int, sample_indices))
        self._mappings = tuple(mapping_array(vocabulary, root)
                               for root in range(len(vocabulary.roots)))
        self._arrays = {}

    def __len__(self):
        return len(self.sample_indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _tokens_array(self, record):
        key = record.shard_path
        if key not in self._arrays:
            group = zarr.open_group(str(Path(key) / "snapshots.zarr"), mode="r")
            self._arrays[key] = group["tokens"]
        return self._arrays[key]

    def read_numpy(self, item):
        global_index = self.sample_indices[int(item)]
        record = self.index.records[global_index]
        local = crop_center(self._tokens_array(record)[record.snapshot_index])
        mapping = self._mappings[record.root_index]
        if local.dtype != np.uint16 or int(local.max(initial=0)) >= len(mapping):
            raise ValueError(f"Unknown local token in {record.key}")
        canonical = mapping[local]
        return canonical.astype(np.int64, copy=False), global_index

    def __getitem__(self, item):
        if torch is None:
            raise RuntimeError("PyTorch is required to return training tensors")
        tokens, global_index = self.read_numpy(item)
        return {"tokens": torch.from_numpy(np.ascontiguousarray(tokens)),
                "sample_index": global_index}
