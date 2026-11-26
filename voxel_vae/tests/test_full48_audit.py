"""Small in-memory audit fixtures, never native data or dataset generation."""
import json
import sqlite3
from unittest import TestCase, main

import numpy as np
import zarr

from voxel_vae.full48_scan import analyze_shard


def fixture():
    store = {}
    group = zarr.group(store=store)
    states = np.zeros((11, 48, 48, 48), dtype=np.uint16)
    states[0, 0, 0, 0] = 4          # Static outer-only unseen token.
    states[1, 0, 0, 0] = 2          # Present in anchor 0's own chain.
    states[6, 0, 0, 0] = 3          # Token 3 is present in an unrelated chain.
    anchors = np.zeros((2, 48, 48, 48), dtype=np.uint16)
    anchors[0, 0, 0, :2] = [2, 3]
    anchors[1, 0, 0, 0] = 3
    for pool, array in (("snapshots", states), ("anchors", anchors)):
        group.create_dataset(pool + ".zarr/tokens", data=array, chunks=(1, 48, 48, 48))
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE samples(idx INTEGER PRIMARY KEY,category TEXT,source_key TEXT,metadata TEXT);"
                     "CREATE TABLE chains(id TEXT PRIMARY KEY,data TEXT,anchor_idx INTEGER);")
    for index in range(11):
        static, first_chain = index == 0, index <= 5
        seed = 13 if first_chain else 17
        source = dict(world_seed=seed, split="train" if first_chain else "test",
                      center_origin=[0, 0, 0], region_origin=[-16, -16, -16],
                      rollout_id="r1" if first_chain else "r2")
        meta = dict(source, snapshot_id=index, category="Static" if static else "D4",
                    chain_id=None if static else source["rollout_id"] + "/D4",
                    chain_index=None if static else (index - 1 if first_chain else index - 6),
                    repack_source=dict(clip="fixture_clip", shard=0, snapshot_id=index))
        db.execute("INSERT INTO samples VALUES(?,?,?,?)",
                   (index, meta["category"], str(index), json.dumps(meta)))
    for anchor_index, first in enumerate((1, 6)):
        name = "r1" if anchor_index == 0 else "r2"
        source = dict(world_seed=13 if anchor_index == 0 else 17,
                      center_origin=[0, 0, 0], region_origin=[-16, -16, -16], rollout_id=name)
        record = dict(id=name + "/D4", category="D4", subtype="A" if anchor_index == 0 else "C",
                      snapshot_indices=list(range(first, first + 5)), anchor_idx=anchor_index,
                      anchor_metadata=source)
        db.execute("INSERT INTO chains VALUES(?,?,?)", (record["id"], json.dumps(record), anchor_index))
    db.commit()
    files = dict(store)
    files["journal.sqlite"] = db.serialize()
    db.close()
    files["READY"] = json.dumps(dict(samples=11)).encode()
    return files


def analyze(files):
    return analyze_shard(files, "fixture", "shard_0000", np.arange(5), 5,
                         {17: "validation"}, {2, 3, 4}, np.array([0, 0, 7, 255, 14], np.uint8))


class Full48AuditTests(TestCase):
    def test_anchor_exclusive_uses_own_chain_not_global_dynamic_union(self):
        result = analyze(fixture())
        self.assertEqual(result["anchor_total"], 2)
        self.assertEqual(result["anchors_with_unseen"], 2)
        self.assertEqual(result["anchor_sample_count"][3], 2)
        self.assertEqual(result["exclusive_voxel_count"][3], 1)
        self.assertEqual(result["exclusive_anchor_count"][3], 1)
        self.assertEqual(result["paired_chain_voxel_count"][3], 1)
        self.assertEqual([link["exclusive"] for link in result["anchor_links"][3]], [True, False])
        self.assertFalse(result["anchor_links"][2][0]["exclusive"])

    def test_full48_outer_tokens_are_separate_from_center_and_anchor_pools(self):
        result = analyze(fixture())
        static = result["frequency"]["train"]["Static"]
        dynamic = result["frequency"]["train"]["Dynamic"]
        self.assertEqual((static["snapshot_count"], dynamic["snapshot_count"]), (1, 5))
        self.assertEqual((static["voxel_count"][4], static["sample_count"][4]), (1, 1))
        self.assertEqual(dynamic["voxel_count"][2], 1)
        self.assertEqual(dynamic["voxel_count"][3], 0)  # Anchor content is not a snapshot.
        self.assertEqual(result["center_counts"]["train"].sum(), 6 * 16 ** 3)
        self.assertEqual(result["center_counts"]["train"][4], 0)
        validation = result["frequency"]["validation"]["Dynamic"]
        self.assertEqual(validation["snapshot_count"], 5)
        self.assertEqual(validation["voxel_count"].sum(), 5 * 48 ** 3)
        self.assertEqual(result["frequency"]["test"]["Dynamic"]["snapshot_count"], 0)
        self.assertEqual(len(result["samples"]), 11)
        self.assertEqual(result["samples"][6]["original_split"], "test")
        self.assertEqual(result["samples"][6]["split"], "validation")

    def test_missing_chunk_and_unmapped_id_stop_instead_of_filling_or_repairing(self):
        files = fixture()
        del files["snapshots.zarr/tokens/0.0.0.0"]
        with self.assertRaisesRegex(ValueError, "missing committed snapshots token chunk"):
            analyze(files)
        files = fixture()
        array = zarr.open_array(store=files, path="snapshots.zarr/tokens", mode="r+")
        array[0, 0, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "unmapped local token 5"):
            analyze(files)


if __name__ == "__main__":
    main()
