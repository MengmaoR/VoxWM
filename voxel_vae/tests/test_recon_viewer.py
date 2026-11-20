import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from voxel_vae.recon_viewer import ReconstructionArchive, select_validation_gallery
from voxel_vae.vocabulary import CanonicalVocabulary


class ReconstructionViewerTests(unittest.TestCase):
    def test_gallery_contains_static_and_complete_dynamic_chains(self):
        records = []
        for index in range(3):
            records.append(SimpleNamespace(
                split="validation", category="Static", chain_id=None,
                chain_index=None, root_index=0, shard=0))
        for category in ("D1", "D2", "D3", "D4"):
            for chain_index in range(5):
                records.append(SimpleNamespace(
                    split="validation", category=category,
                    chain_id=f"{category}/chain", chain_index=chain_index,
                    root_index=0, shard=0))
        index = SimpleNamespace(records=records)
        selected = select_validation_gallery(
            index, range(len(records)), static_count=2, chains_per_dynamic=1)
        self.assertEqual(len(selected), 22)
        self.assertEqual(sum(records[item].category == "Static" for item in selected), 2)
        for category in ("D1", "D2", "D3", "D4"):
            members = [records[item].chain_index for item in selected
                       if records[item].category == category]
            self.assertEqual(members, list(range(5)))

    def test_archive_preserves_exact_tokens_and_builds_static_dynamic_groups(self):
        vocabulary = CanonicalVocabulary((
            ("air", 0), ("mcl_core:stone", 0), ("mcl_core:water_flowing", 3)))
        gt = np.zeros((6, 16, 16, 16), dtype=np.uint16)
        gt[:, 1, 2, 3] = 1
        predicted = gt.copy()
        predicted[0, 1, 2, 3] = 2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vocabulary.save(root / "training_vocabulary.json")
            np.savez_compressed(
                root / "examples.npz", gt_tokens=gt, predicted_tokens=predicted,
                category=np.asarray(["Static"] + ["D1"] * 5),
                chain_id=np.asarray([""] + ["chain/d1"] * 5),
                chain_index=np.asarray([-1, 0, 1, 2, 3, 4], dtype=np.int16),
                snapshot_id=np.arange(6, dtype=np.int32),
                world_seed=np.arange(100, 106, dtype=np.int64),
                split=np.asarray(["validation"] * 6),
                biome_group=np.asarray(["Plains"] + [""] * 5),
                stratum=np.asarray(["surface"] + [""] * 5),
            )
            archive = ReconstructionArchive(
                root / "examples.npz", root / "training_vocabulary.json")
            self.assertEqual(archive.groups["Static"][0]["indices"], [0])
            self.assertEqual(archive.groups["D1"][0]["indices"], [1, 2, 3, 4, 5])
            sample = archive.sample(0)
            self.assertEqual(sample["metrics"]["errors"], 1)
            self.assertAlmostEqual(sample["metrics"]["exact_accuracy"], 4095 / 4096)
            self.assertEqual(sample["gt"][1 * 256 + 2 * 16 + 3], 1)
            self.assertEqual(sample["predicted"][1 * 256 + 2 * 16 + 3], 2)

    def test_comparison_archive_keeps_plain_masked_and_true_change_mask(self):
        vocabulary = CanonicalVocabulary((
            ("air", 0), ("mcl_core:stone", 0), ("mcl_core:water_flowing", 3)))
        gt = np.zeros((1, 16, 16, 16), dtype=np.uint16)
        plain = gt.copy()
        masked = gt.copy()
        plain[0, 1, 1, 1] = 1
        changed = np.zeros_like(gt, dtype=bool)
        changed[0, 1, 1, 1] = True
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vocabulary.save(root / "training_vocabulary.json")
            np.savez_compressed(
                root / "comparison.npz", gt_tokens=gt,
                predicted_tokens=masked,
                predicted_plain_tokens=plain,
                predicted_masked_tokens=masked,
                changed_mask=changed,
                category=np.asarray(["Static"]), chain_id=np.asarray([""]),
                chain_index=np.asarray([-1], dtype=np.int16),
                snapshot_id=np.asarray([0], dtype=np.int32),
                world_seed=np.asarray([42], dtype=np.int64),
            )
            archive = ReconstructionArchive(
                root / "comparison.npz", root / "training_vocabulary.json")
            self.assertEqual(list(archive.predictions), ["Plain", "Masked"])
            sample = archive.sample(0)
            offset = 1 * 256 + 1 * 16 + 1
            self.assertEqual(sample["predictions"]["Plain"][offset], 1)
            self.assertEqual(sample["predictions"]["Masked"][offset], 0)
            self.assertTrue(sample["changed_mask"][offset])


if __name__ == "__main__":
    unittest.main()
