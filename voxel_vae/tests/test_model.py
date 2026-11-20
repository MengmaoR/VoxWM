import tempfile
import unittest
from pathlib import Path

import torch

from voxel_vae.checkpoint import load_checkpoint, save_checkpoint
from voxel_vae.losses import vae_loss
from voxel_vae.model import VoxelVAE
from voxel_vae.vocabulary import CanonicalVocabulary


class _Index:
    roots = ("/dataset/a",)
    manifest_hashes = ("manifest-hash",)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = VoxelVAE(7)
        self.tokens = torch.randint(0, 7, (1, 16, 16, 16))

    def test_embedding_encoder_decoder_and_sampling_shapes(self):
        self.assertEqual(self.model.embed(self.tokens).shape, (1, 32, 16, 16, 16))
        shapes = []
        hooks = [self.model.downsample16_8.register_forward_hook(
                    lambda _m, _i, output: shapes.append(tuple(output.shape))),
                 self.model.downsample8_4.register_forward_hook(
                    lambda _m, _i, output: shapes.append(tuple(output.shape)))]
        mu, logvar = self.model.encode(self.tokens)
        for hook in hooks:
            hook.remove()
        self.assertEqual(shapes, [(1, 192, 8, 8, 8), (1, 192, 4, 4, 4)])
        self.assertEqual(mu.shape, (1, 16, 4, 4, 4))
        self.assertEqual(logvar.shape, mu.shape)
        self.assertTrue(torch.all(logvar >= -10) and torch.all(logvar <= 10))
        self.assertEqual(self.model.decode(mu).shape, (1, 7, 16, 16, 16))
        self.model.train()
        self.assertFalse(torch.equal(self.model.sample(mu, logvar), mu))
        self.model.eval()
        self.assertTrue(torch.equal(self.model.sample(mu, logvar), mu))

    def test_plain_ce_kl_backward_has_finite_key_gradients(self):
        self.model.train()
        output = self.model(self.tokens)
        losses = vae_loss(output["logits"], self.tokens, output["mu"],
                          output["logvar"], beta=1e-6)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total_loss"].backward()
        for parameter in (self.model.embedding.weight,
                          self.model.encoder_input.weight,
                          self.model.decoder_logits.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_checkpoint_roundtrip_and_vocabulary_mismatch_rejection(self):
        vocabulary = CanonicalVocabulary((
            ("air", 0), ("mcl_core:stone", 0), ("node:a", 0),
            ("node:b", 0), ("node:c", 0), ("node:d", 0), ("node:e", 0)))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model.eval()
        expected = self.model.encode_mean(self.tokens).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 9, {"beta": 1e-6},
                            vocabulary, _Index(), 1.0, "test")
            restored = VoxelVAE(7)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            payload = load_checkpoint(path, restored, restored_optimizer,
                                      vocabulary, _Index(), restore_rng=False)
            restored.eval()
            self.assertEqual(payload["global_step"], 9)
            self.assertTrue(torch.equal(expected, restored.encode_mean(self.tokens)))
            wrong = CanonicalVocabulary((
                ("air", 0), ("mcl_core:dirt", 0), ("node:a", 0),
                ("node:b", 0), ("node:c", 0), ("node:d", 0), ("node:e", 0)))
            with self.assertRaisesRegex(ValueError, "vocabulary_sha256"):
                load_checkpoint(path, restored, None, wrong, _Index(), restore_rng=False)

    def test_same_seed_reproduces_embedding_initialization(self):
        torch.manual_seed(42)
        first = VoxelVAE(7).embedding.weight.detach().clone()
        torch.manual_seed(42)
        second = VoxelVAE(7).embedding.weight.detach().clone()
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
