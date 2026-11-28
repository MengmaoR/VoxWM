import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vox2pix_data.vae_embeddings import BACKGROUND, build_lookup, canonical_ids, embed
from voxel_vae.curated_data import _chain_masks
from voxel_vae.curated_vae import copy_expanded_state
from voxel_vae.losses import masked_batch_normalization, masked_vae_loss
from voxel_vae.vocabulary import CanonicalVocabulary


class CuratedVAETests(unittest.TestCase):
    def test_append_order_and_expanded_head(self):
        vocabulary = CanonicalVocabulary([("dirt", 0), ("stone", 0), ("air", 0)], preserve_ids=True)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "vocabulary.json"
            vocabulary.save(path)
            self.assertEqual(CanonicalVocabulary.load(path).pairs, vocabulary.pairs)

        def model(size):
            return nn.ModuleDict({"embedding": nn.Embedding(size, 32),
                                  "decoder_logits": nn.Conv3d(32, size, 1)})

        old, new = model(2).state_dict(), model(3)
        appended = {key: value[2:].clone() for key, value in new.state_dict().items()}
        copy_expanded_state(new, old, 2)
        for key, parameter in new.state_dict().items():
            torch.testing.assert_close(parameter[:2], old[key])
            torch.testing.assert_close(parameter[2:], appended[key])

    def test_five_frame_alignment_and_empty_change_loss(self):
        states = np.zeros((5, 3, 16, 16, 16), dtype=np.uint16)
        states[2:, 0, 1, 2, 3] = 1
        source = {"category": "D1", "split": "train", "world_seed": 8,
                  "region_origin_enu": [0, 0, 0], "chunk_indices": [2, 13, 25], "snapshot_id": "fixture"}
        rows = [dict(source) for _ in range(5)]
        masks = _chain_masks(states, rows)
        self.assertEqual(masks[:, 0, 1, 2, 3].tolist(), [False, True, True, False, False])
        self.assertFalse(masks[:, 1:].any())
        rows[4]["chunk_indices"] = [1, 13, 25]
        with self.assertRaisesRegex(ValueError, "spatial selection"):
            _chain_masks(states, rows)

        categories, indices = ["Static", "D1", "D2"], [0, 1, 2]
        changed = {1: masks[1, 0], 2: masks[1, 1]}
        counts = masked_batch_normalization(indices, categories, changed, allow_empty_changes=True)
        self.assertEqual(counts["changed"], {"D1": 1, "D2": 0, "D3": 0, "D4": 0})
        self.assertEqual(counts["unchanged"]["D2"], 1)
        config = dict(objective="static_dynamic_masked_v1", lambda_static=0.5,
                      lambda_dynamic=0.5, alpha_changed=0.5, beta_kl=1e-6)
        target = torch.zeros((3, 16, 16, 16), dtype=torch.long)
        logits = torch.zeros((3, 2, 16, 16, 16))
        latent = torch.zeros((3, 1, 1, 1, 1))
        result = masked_vae_loss(logits, target, latent, latent, indices, categories,
                                 changed, counts, config)
        torch.testing.assert_close(result["loss_total"], torch.tensor(np.log(2), dtype=torch.float32))

    def test_vox2pix_raw_param2_and_background(self):
        lookup = build_lookup(["water", "air"], [("air", 0), ("water", 0), ("water", 8)], [0, 2])
        node = np.array([1, 0, BACKGROUND], dtype=np.uint16)
        param = np.array([0, 8, 0], dtype=np.uint8)
        weight = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
        vectors, valid = embed(node, param, lookup, weight)
        np.testing.assert_array_equal(vectors[:2], weight[[0, 2]])
        self.assertFalse(vectors[2].any())
        self.assertEqual(valid.tolist(), [True, True, False])
        param[1] = 0
        with self.assertRaisesRegex(ValueError, "node ID 0 and param2 0"):
            canonical_ids(node, param, lookup)


if __name__ == "__main__":
    unittest.main()
