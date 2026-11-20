import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from voxel_vae.dataset import (DatasetIndex, SampleRef, VoxelSnapshotDataset,
                               build_dataset_index, crop_center)
from voxel_vae.change_masks import ChangeMaskCache, compute_chain_masks
from voxel_vae.e00 import _balanced_subset, _tiny_a
from voxel_vae.sampler import BALANCE, BalancedBatchSampler
from voxel_vae.vocabulary import build_canonical_vocabulary


def _write_root(path, pairs, second=False):
    path = Path(path)
    path.mkdir()
    (path / "dataset_manifest.json").write_text(json.dumps({
        "version": "vae-region48-fixed24-v2", "source_mode": "native"}))
    (path / "vocabulary.json").write_text(json.dumps({
        "version": 1, "K": len(pairs),
        "tokens": [{"id": i, "node_name": name, "param2": param2}
                   for i, (name, param2) in enumerate(pairs)]}))
    shard = path / "shards/shard_0000"
    shard.mkdir(parents=True)
    (shard / "manifest.json").write_text(json.dumps({"shard": 0}))
    rows = []
    if second:
        rows.append((0, "Static", None, None, 200, "train"))
    else:
        rows.append((0, "Static", None, None, 100, "train"))
        for index in range(1, 6):
            rows.append((index, "D4", "chain/d4", index - 1, 101, "train"))
        rows.append((6, "Static", None, None, 102, "test"))
    db = sqlite3.connect(shard / "journal.sqlite")
    try:
        db.executescript("""
            CREATE TABLE samples(idx INTEGER PRIMARY KEY,category TEXT,source_key TEXT,metadata TEXT);
            CREATE TABLE chains(id TEXT PRIMARY KEY,data TEXT,anchor_idx INTEGER);
            CREATE TABLE attempts(id INTEGER PRIMARY KEY,data TEXT);
        """)
        for snapshot, category, chain_id, chain_index, seed, split in rows:
            metadata = {
                "snapshot_id": snapshot, "category": category,
                "chain_id": chain_id, "chain_index": chain_index,
                "world_seed": seed, "split": split,
                "biome_group": "Plains" if category == "Static" else None,
                "stratum": "surface" if category == "Static" else None,
            }
            db.execute("INSERT INTO samples VALUES(?,?,?,?)",
                       (snapshot, category, f"key-{snapshot}", json.dumps(metadata)))
        if not second:
            chain = {"id": "chain/d4", "category": "D4",
                     "snapshot_indices": [1, 2, 3, 4, 5], "subtype": "B"}
            db.execute("INSERT INTO chains VALUES(?,?,?)",
                       (chain["id"], json.dumps(chain), 0))
        db.commit()
    finally:
        db.close()
    group = zarr.open_group(str(shard / "snapshots.zarr"), mode="w")
    tokens = group.create_dataset("tokens", shape=(len(rows), 48, 48, 48),
                                  chunks=(1, 48, 48, 48), dtype="<u2")
    for index in range(len(rows)):
        outside = 0 if not second else 1
        stone = 1 if not second else 0
        tokens[index] = outside
        tokens[index, 16:32, 16:32, 16:32] = stone
        if not second and 1 <= index <= 5:
            center = np.full((16, 16, 16), stone, dtype=np.uint16)
            center.reshape(-1)[:index] = 2
            tokens[index, 16:32, 16:32, 16:32] = center
            tokens[index, 0, 0, 0] = index % 3
    anchors = zarr.open_group(str(shard / "anchors.zarr"), mode="w")
    anchor_tokens = anchors.create_dataset("tokens", shape=(3, 48, 48, 48),
                                           chunks=(1, 48, 48, 48), dtype="<u2")
    anchor_tokens[:] = 2
    return path


class DataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        base = Path(self.folder.name)
        self.root1 = _write_root(base / "one", [
            ("air", 0), ("mcl_core:stone", 0), ("mcl_core:water_flowing", 3)])
        self.root2 = _write_root(base / "two", [
            ("mcl_core:stone", 0), ("air", 0), ("mcl_core:lava_flowing", 4)], True)

    def tearDown(self):
        self.folder.cleanup()

    def test_center_crop_and_semantic_multiroot_remap_are_exact(self):
        vocabulary = build_canonical_vocabulary([self.root1, self.root2])
        index = build_dataset_index(vocabulary)
        dataset = VoxelSnapshotDataset(index, vocabulary)
        first, _ = dataset.read_numpy(0)
        second_global = next(i for i, row in enumerate(index.records) if row.root_index == 1)
        second, _ = dataset.read_numpy(second_global)
        stone = vocabulary.index[("mcl_core:stone", 0)]
        self.assertEqual(first.shape, (16, 16, 16))
        self.assertTrue(np.all(first == stone))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(vocabulary.roots[0].identity_mapping)
        self.assertFalse(vocabulary.roots[1].identity_mapping)
        raw = np.zeros((48, 48, 48), np.uint16)
        raw[16:32, 16:32, 16:32] = 9
        self.assertTrue(np.all(crop_center(raw) == 9))

    def test_reader_uses_committed_samples_not_light_or_anchors_and_is_read_only(self):
        before = {str(path.relative_to(self.root1)): (path.stat().st_size, path.stat().st_mtime_ns)
                  for path in self.root1.rglob("*") if path.is_file()}
        vocabulary = build_canonical_vocabulary([self.root1])
        index = build_dataset_index(vocabulary)
        self.assertEqual(len(index.records), 7)
        d4 = [row for row in index.records if row.category == "D4"]
        self.assertEqual([row.chain_index for row in d4], list(range(5)))
        self.assertTrue(all(row.d4_subtype == "B" for row in d4))
        dataset = VoxelSnapshotDataset(index, vocabulary)
        dataset.read_numpy(0)  # snapshots.zarr has deliberately no light array.
        after = {str(path.relative_to(self.root1)): (path.stat().st_size, path.stat().st_mtime_ns)
                 for path in self.root1.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_partial_shard_uses_committed_journal_without_export_manifest(self):
        shard = self.root1 / "shards/shard_0000"
        partial = self.root1 / "shards/.shard_0000.partial"
        shard.rename(partial)
        (partial / "manifest.json").unlink()
        vocabulary = build_canonical_vocabulary([self.root1])
        index = build_dataset_index(vocabulary)
        self.assertEqual(len(index.records), 7)
        self.assertTrue(all(Path(row.shard_path) == partial.resolve()
                            for row in index.records))

    def test_world_seed_cannot_cross_splits(self):
        row = dict(root_index=0, shard=0, shard_path="/tmp/x", snapshot_index=0,
                   category="Static", chain_id=None, chain_index=None,
                   world_seed=7, biome_group=None, stratum=None, d4_subtype=None)
        with self.assertRaises(ValueError):
            DatasetIndex(["x"], [SampleRef(split="train", **row),
                                  SampleRef(split="test", **row)], ["hash"], "vocab")

    def test_balanced_batch_is_train_only_32_8_8_8_8(self):
        records = []
        index_value = 0
        for category, count in {"Static": 40, "D1": 12, "D2": 12,
                                "D3": 12, "D4": 12}.items():
            for _ in range(count):
                records.append(SampleRef(0, 0, "/tmp/x", index_value, category,
                    None, None, 1000 + index_value, "train", None, None, None))
                index_value += 1
        for _ in range(10):
            records.append(SampleRef(0, 0, "/tmp/x", index_value, "Static",
                None, None, 1000 + index_value, "test", None, None, None))
            index_value += 1
        index = DatasetIndex(["x"], records, ["hash"], "vocab")
        sampler = BalancedBatchSampler(index, seed=3)
        batch = sampler.batch(0)
        self.assertEqual(len(batch), 64)
        self.assertEqual(sampler.counts(batch), Counter(BALANCE))
        self.assertTrue(all(index.records[item].split == "train" for item in batch))

    def test_fixed_e00_subsets_have_required_real_state_composition(self):
        records = []
        snapshot = 0
        for split in ("train", "validation"):
            for static_index in range(100):
                records.append(SampleRef(
                    0, 0, "/tmp/x", snapshot, "Static", None, None,
                    10_000 + snapshot, split, f"biome-{static_index % 25}",
                    ("surface", "elevated_complex", "shallow_underground", "deep_cave")[static_index % 4],
                    None))
                snapshot += 1
            for category in ("D1", "D2", "D3", "D4"):
                for chain_number in range(5):
                    chain = f"{split}/{category}/{chain_number}"
                    for chain_index in range(5):
                        records.append(SampleRef(
                            0, 0, "/tmp/x", snapshot, category, chain, chain_index,
                            10_000 + snapshot, split, None, None,
                            "A" if category == "D4" else None))
                        snapshot += 1
        index = DatasetIndex(["x"], records, ["hash"], "vocab")
        tiny = _tiny_a(index)
        self.assertEqual(len(tiny), 8)
        self.assertEqual(set(index.records[item].category for item in tiny),
                         {"Static", "D1", "D2", "D3", "D4"})
        balanced = _balanced_subset(index, "train")
        self.assertEqual(Counter(index.records[item].category for item in balanced),
                         Counter({"Static": 80, "D1": 20, "D2": 20,
                                  "D3": 20, "D4": 20}))
        for category in ("D1", "D2", "D3", "D4"):
            chains = Counter(index.records[item].chain_id for item in balanced
                             if index.records[item].category == category)
            self.assertEqual(sorted(chains.values()), [5, 5, 5, 5])

    def test_change_masks_use_end_neighbors_and_middle_union(self):
        states = np.zeros((5, 16, 16, 16), dtype=np.uint16)
        states[1:, 1, 1, 1] = 3
        states[2:, 2, 2, 2] = 4
        states[3:, 3, 3, 3] = 5
        states[4, 4, 4, 4] = 6
        masks = compute_chain_masks(states)
        self.assertEqual(int(masks[0].sum()), 1)
        self.assertEqual(int(masks[4].sum()), 1)
        self.assertTrue(masks[2, 2, 2, 2])
        self.assertTrue(masks[2, 3, 3, 3])
        self.assertFalse(masks[2, 1, 1, 1])

    def test_canonical_param2_token_change_is_changed_and_bad_chain_rejected(self):
        states = np.zeros((5, 16, 16, 16), dtype=np.uint16)
        for state in range(1, 5):
            states[state:, state, 0, 0] = state  # distinct canonical token IDs
        masks = compute_chain_masks(states)
        self.assertTrue(masks[0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "chain_index"):
            compute_chain_masks(states, [0, 1, 3, 2, 4])
        with self.assertRaisesRegex(ValueError, "adjacent"):
            compute_chain_masks(np.zeros_like(states))

    def test_mask_cache_reads_complete_center_chain_and_ignores_anchor(self):
        vocabulary = build_canonical_vocabulary([self.root1])
        index = build_dataset_index(vocabulary)
        dynamic = [item for item, row in enumerate(index.records)
                   if row.category == "D4"]
        cache = ChangeMaskCache(index, vocabulary, [dynamic[2]])
        self.assertEqual(int(cache.changed(dynamic[0]).sum()), 1)
        self.assertEqual(int(cache.changed(dynamic[2]).sum()), 2)
        self.assertEqual(int(cache.changed(dynamic[4]).sum()), 1)
        stats = cache.statistics(dynamic)
        self.assertEqual(stats["chains"], 1)
        self.assertEqual(stats["dynamic_states"], 5)
        self.assertEqual(stats["d4_chain_subtypes"], {"B": 1})
        with patch.object(VoxelSnapshotDataset, "read_numpy", autospec=True,
                          side_effect=VoxelSnapshotDataset.read_numpy) as read:
            lazy = ChangeMaskCache(index, vocabulary, dynamic, lazy=True)
            read.assert_not_called()
            for item in reversed(dynamic):
                self.assertTrue(np.array_equal(lazy.changed(item), cache.changed(item)))
            self.assertEqual(read.call_count, 5)
            self.assertEqual(lazy.statistics(dynamic), stats)
            self.assertEqual(read.call_count, 5)


if __name__ == "__main__":
    unittest.main()
