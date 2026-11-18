"""Small metadata/vocabulary tests only; no production dataset is modified."""
import contextlib
import io
import tempfile
import unittest

import numpy as np
import zarr

from vae_p0.codec import Vocabulary
from vae_data.sharded.merge_static import (_copy_selected, merge_vocabularies, parser,
                                            reindex_chains, reindex_metadata)


class MergeStaticTests(unittest.TestCase):
    def test_dynamic_ids_stay_fixed_and_static_ids_are_remapped_by_pair(self):
        dynamic = Vocabulary()
        dynamic.add("air", 0)
        dynamic.add("mcl_core:stone", 0)
        static = Vocabulary()
        static.add("mcl_core:stone", 0)
        static.add("air", 0)
        static.add("mcl_core:dirt", 3)
        merged, mapping = merge_vocabularies(dynamic, static)
        self.assertEqual(merged.entries, [
            ("air", 0), ("mcl_core:stone", 0), ("mcl_core:dirt", 3)])
        self.assertEqual(mapping.tolist(), [1, 0, 2])

    def test_rows_and_chain_indices_are_rebuilt_without_mixing_sources(self):
        dynamic = [
            (4, "D1", "a", {"snapshot_id": 4, "vocabulary_size_at_commit": 2}),
            (5, "D1", "b", {"snapshot_id": 5, "vocabulary_size_at_commit": 2}),
        ]
        static = [(8, "Static", "c", {"snapshot_id": 8, "vocabulary_size_at_commit": 3})]
        rows, mapping = reindex_metadata(dynamic, static, 7)
        self.assertEqual([row[0] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0][3]["vocabulary_size_at_commit"], 2)
        self.assertEqual(rows[2][3]["vocabulary_size_at_commit"], 7)
        self.assertEqual(rows[2][3]["merge_source"]["snapshot_id"], 8)
        chains = reindex_chains([{"id": "x", "snapshot_indices": [4, 5]}], mapping)
        self.assertEqual(chains[0]["snapshot_indices"], [0, 1])

    def test_cli_is_read_only_unless_apply_is_explicit(self):
        args = parser([])
        self.assertFalse(args.apply)
        self.assertFalse(args.resume_staging)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser(["--resume-staging"])

    def test_selected_zarr_rows_are_reordered_and_token_remapped(self):
        with tempfile.TemporaryDirectory() as folder:
            source = zarr.open_array(folder + "/source", mode="w", shape=(3, 2, 2, 2),
                                     chunks=(1, 2, 2, 2), dtype="<u2")
            destination = zarr.open_array(folder + "/destination", mode="w",
                                          shape=(2, 2, 2, 2), chunks=(1, 2, 2, 2),
                                          dtype="<u2")
            source[0] = 0
            source[1] = 1
            source[2] = 2
            _copy_selected(source, destination, [2, 0], 0, 2,
                           np.asarray([4, 5, 6], dtype=np.uint16))
            np.testing.assert_array_equal(destination[:, 0, 0, 0], [6, 4])


if __name__ == "__main__":
    unittest.main()
