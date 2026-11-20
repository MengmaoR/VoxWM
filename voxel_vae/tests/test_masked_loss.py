import unittest

import torch
import torch.nn.functional as F

from voxel_vae.losses import (DYNAMIC_CATEGORIES, masked_batch_normalization,
                              masked_vae_loss, vae_loss)


CONFIG = {
    "objective": "static_dynamic_masked_v1",
    "lambda_static": 0.5,
    "lambda_dynamic": 0.5,
    "alpha_changed": 0.5,
    "beta_kl": 1e-6,
}


def fixture():
    categories = ["Static"] * 32
    for category in DYNAMIC_CATEGORIES:
        categories += [category] * 8
    indices = list(range(64))
    masks = {}
    for item, category in zip(indices, categories):
        if category != "Static":
            mask = torch.zeros((16, 16, 16), dtype=torch.bool)
            mask.reshape(-1)[item % 17] = True
            masks[item] = mask
    normalization = masked_batch_normalization(indices, categories, masks)
    return indices, categories, masks, normalization


class MaskedLossTests(unittest.TestCase):
    def test_effective_batch_denominators_are_32_and_8_per_dynamic_category(self):
        _, _, _, normalization = fixture()
        self.assertEqual(normalization["static"], 32)
        self.assertEqual(normalization["batch_size"], 64)
        self.assertEqual(normalization["changed"], {category: 8 for category in DYNAMIC_CATEGORIES})
        self.assertEqual(normalization["unchanged"], {category: 8 for category in DYNAMIC_CATEGORIES})

    def test_static_component_is_all_voxel_mean(self):
        indices, categories, masks, normalization = fixture()
        target = torch.zeros((32, 16, 16, 16), dtype=torch.long)
        logits = torch.randn((32, 3, 16, 16, 16), requires_grad=True)
        mu = torch.zeros((32, 1, 1, 1, 1))
        logvar = torch.zeros_like(mu)
        result = masked_vae_loss(
            logits, target, mu, logvar, indices[:32], categories[:32],
            masks, normalization, CONFIG)
        self.assertTrue(torch.allclose(
            result["loss_static"], F.cross_entropy(logits, target), atol=1e-6))

    def test_each_dynamic_sample_is_normalized_before_category_average(self):
        indices, categories, masks, normalization = fixture()
        selected = indices[32:]
        selected_categories = categories[32:]
        target = torch.zeros((32, 16, 16, 16), dtype=torch.long)
        logits = torch.zeros((32, 2, 16, 16, 16), requires_grad=True)
        for offset in range(32):
            logits.data[offset, 1] = float(offset) / 10.0
        mu = torch.zeros((32, 1, 1, 1, 1))
        result = masked_vae_loss(
            logits, target, mu, torch.zeros_like(mu), selected,
            selected_categories, masks, normalization, CONFIG)
        per_sample = F.cross_entropy(logits, target, reduction="none").flatten(1)
        expected = []
        for category in DYNAMIC_CATEGORIES:
            values = []
            for offset, item in enumerate(selected):
                if selected_categories[offset] == category:
                    values.append(per_sample[offset][masks[item].flatten()].mean())
            expected.append(torch.stack(values).mean())
        self.assertTrue(torch.allclose(
            result["loss_dynamic_changed"], torch.stack(expected).mean(), atol=1e-6))

    def test_one_empty_unchanged_mask_is_skipped_without_nan(self):
        indices, categories, masks, _ = fixture()
        masks[32] = torch.ones((16, 16, 16), dtype=torch.bool)
        normalization = masked_batch_normalization(indices, categories, masks)
        self.assertEqual(normalization["unchanged"]["D1"], 7)
        target = torch.zeros((8, 16, 16, 16), dtype=torch.long)
        logits = torch.randn((8, 2, 16, 16, 16), requires_grad=True)
        mu = torch.zeros((8, 1, 1, 1, 1))
        result = masked_vae_loss(
            logits, target, mu, torch.zeros_like(mu), indices[32:40],
            categories[32:40], masks, normalization, CONFIG)
        self.assertTrue(torch.isfinite(result["loss_dynamic_unchanged"]))

    def test_full_batch_and_microbatch_two_have_equal_loss_and_gradients(self):
        indices, categories, masks, normalization = fixture()
        torch.manual_seed(9)
        target = torch.randint(0, 3, (64, 16, 16, 16))
        initial = torch.randn((64, 3, 16, 16, 16))
        mu = torch.randn((64, 2, 1, 1, 1))
        logvar = torch.randn_like(mu).clamp(-2, 2)

        full_logits = initial.clone().requires_grad_(True)
        full = masked_vae_loss(
            full_logits, target, mu, logvar, indices, categories,
            masks, normalization, CONFIG)
        full["loss_total"].backward()

        accumulated_logits = initial.clone().requires_grad_(True)
        total = 0.0
        for start in range(0, 64, 2):
            chunk = masked_vae_loss(
                accumulated_logits[start:start + 2], target[start:start + 2],
                mu[start:start + 2], logvar[start:start + 2],
                indices[start:start + 2], categories[start:start + 2],
                masks, normalization, CONFIG)
            total = total + chunk["loss_total"].detach()
            chunk["loss_total"].backward()
        self.assertTrue(torch.allclose(total, full["loss_total"].detach(), atol=2e-6))
        self.assertTrue(torch.allclose(
            accumulated_logits.grad, full_logits.grad, atol=2e-7, rtol=2e-5))

    def test_kl_sum_matches_plain_e00_definition(self):
        indices, categories, masks, normalization = fixture()
        target = torch.zeros((64, 16, 16, 16), dtype=torch.long)
        logits = torch.zeros((64, 2, 16, 16, 16))
        torch.manual_seed(5)
        mu = torch.randn((64, 16, 4, 4, 4))
        logvar = torch.randn_like(mu).clamp(-3, 3)
        masked = masked_vae_loss(
            logits, target, mu, logvar, indices, categories,
            masks, normalization, CONFIG)
        plain = vae_loss(logits, target, mu, logvar, beta=1e-6)
        self.assertTrue(torch.equal(masked["loss_kl_raw"], plain["kl_raw"]))
        self.assertTrue(torch.equal(masked["loss_beta_kl"], plain["beta_kl"]))


if __name__ == "__main__":
    unittest.main()
