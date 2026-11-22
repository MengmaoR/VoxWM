"""Bounded test inference with the E01 reconstruction and pair metrics."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .change_masks import compute_chain_masks
from .evaluate import _autocast, _dynamic_metrics, _ratio
from .freeze_d3 import D3Audit
from .freeze_statistics import Moments, channel_observations
from .losses import DYNAMIC_CATEGORIES, validate_masked_loss_config


@torch.inference_mode()
def posterior_mean(model, tokens, device, precision):
    model.eval()
    with _autocast(device, precision):
        mu = model.encode_mean(tokens.to(device, non_blocking=True))
    return mu.float().cpu().numpy()


@torch.inference_mode()
def reconstruct(model, tokens, device, precision):
    model.eval()
    target = tokens.to(device, non_blocking=True)
    with _autocast(device, precision):
        mu, logvar = model.encode(target)
        logits = model.decode(mu)
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Non-finite reconstruction logits during freeze audit")
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    return (logits.argmax(dim=1).cpu().numpy().astype(np.uint16),
            ce.cpu().numpy(), mu.float().cpu().numpy(), logvar.float().cpu().numpy())


def _accuracy(counts):
    return {
        "exact_token_accuracy": _ratio(counts["correct"], counts["voxels"]),
        "non_air_exact_token_accuracy": _ratio(counts["non_air_correct"], counts["non_air_voxels"]),
    }


def _masked_accuracy(counts):
    return {**_accuracy(counts),
            "changed_voxel_exact_accuracy": _ratio(counts["changed_correct"], counts["changed"]),
            "unchanged_voxel_exact_accuracy": _ratio(counts["unchanged_correct"], counts["unchanged"]),
            "changed_voxels": counts["changed"], "unchanged_voxels": counts["unchanged"]}


class ReconstructionAudit:
    """Keep only the current reconstruction batch and complete chain pairs."""

    def __init__(self, vocabulary, loss):
        self.loss = validate_masked_loss_config(loss)
        self.non_air = np.ones(vocabulary.K, dtype=bool)
        self.non_air[list(vocabulary.air_ids)] = False
        self.categories = defaultdict(Counter)
        self.by_biome = defaultdict(Counter)
        self.by_stratum = defaultdict(Counter)
        self.masked_ce = defaultdict(Counter)
        self.pairs = defaultdict(Counter)
        self.chains = Counter()
        self.ce_sum = self.kl_sum = self.posterior_std_sum = 0.0
        self.latents = Moments(16)
        self.distances = Moments(1)
        self.zero_distances = Counter()
        self.d3 = D3Audit(vocabulary)

    def observe(self, groups, gt, pred, ce, mu, logvar):
        self.ce_sum += float(ce.sum(dtype=np.float64))
        self.kl_sum += float((-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar))).sum(dtype=np.float64))
        self.posterior_std_sum += float(np.exp(0.5 * logvar).sum(dtype=np.float64))
        self.latents.update(channel_observations(mu))
        if not np.isfinite(logvar).all() or not np.isfinite(ce).all():
            raise FloatingPointError("Non-finite evaluation loss or posterior")

        start = 0
        for rows in groups:
            stop = start + len(rows)
            self._observe_group(rows, gt[start:stop], pred[start:stop], ce[start:stop], mu[start:stop])
            start = stop
        if start != len(gt):
            raise ValueError("Evaluation batches must contain complete sample groups")

    def _observe_group(self, rows, gt, pred, ce, mu):
        row = rows[0]
        category = row.category
        exact = gt == pred
        non_air = self.non_air[gt]
        counts = {"samples": len(rows), "voxels": gt.size, "correct": int(exact.sum()),
                  "non_air_correct": int((exact & non_air).sum()), "non_air_voxels": int(non_air.sum())}
        self.categories[category].update(counts)
        if category == "Static":
            self.by_biome[row.biome_group or "unspecified"].update(counts)
            self.by_stratum[row.stratum or "unspecified"].update(counts)
            self.masked_ce[category]["sum"] += float(ce.mean(dtype=np.float64))
            return

        masks = compute_chain_masks(gt, [r.chain_index for r in rows])
        self.categories[category].update(
            changed=int(masks.sum()), changed_correct=int((exact & masks).sum()),
            unchanged=int((~masks).sum()), unchanged_correct=int((exact & ~masks).sum()))
        for name, selected in (("changed", masks), ("unchanged", ~masks)):
            counts = selected.reshape(len(rows), -1).sum(axis=1)
            sums = (ce * selected).reshape(len(rows), -1).sum(axis=1, dtype=np.float64)
            self.masked_ce[category][name] += float((sums[counts > 0] / counts[counts > 0]).sum())
            self.masked_ce[category][f"{name}_states"] += int((counts > 0).sum())

        paired_changed = gt[1:] != gt[:-1]
        predicted_changed = pred[1:] != pred[:-1]
        self.pairs[category].update(
            pairs=len(rows) - 1, changed=int(paired_changed.sum()),
            unchanged=int((~paired_changed).sum()),
            predicted_changed=int((paired_changed & predicted_changed).sum()),
            pair_exact=int((paired_changed & exact[1:] & exact[:-1]).sum()),
            unchanged_stable=int((~paired_changed & ~predicted_changed).sum()))
        self.chains[category] += 1
        distances = np.linalg.norm((mu[1:] - mu[:-1]).reshape(len(rows) - 1, -1), axis=1)
        self.distances.update(distances[:, None])
        self.zero_distances[category] += int((distances == 0).sum())
        if category == "D3":
            self.d3.observe_states(gt, pred, masks)
            for a, b, c, d in zip(gt[:-1], gt[1:], pred[:-1], pred[1:]):
                self.d3.observe_transition(a, b, c, d)

    def _loss_components(self, samples):
        static = self.masked_ce["Static"]["sum"] / self.categories["Static"]["samples"]
        components = {"loss_static": static}
        for name in ("changed", "unchanged"):
            by_category = {category: self.masked_ce[category][name]
                           / self.masked_ce[category][f"{name}_states"]
                           for category in DYNAMIC_CATEGORIES}
            components[f"dynamic_{name}_by_category"] = by_category
            components[f"loss_dynamic_{name}"] = sum(by_category.values()) / len(by_category)
        alpha = self.loss["alpha_changed"]
        reconstruction = (self.loss["lambda_static"] * static + self.loss["lambda_dynamic"] * (
            (1 - alpha) * components["loss_dynamic_unchanged"] + alpha * components["loss_dynamic_changed"]))
        kl = self.kl_sum / samples
        weighted_kl = self.loss["beta_kl"] * kl
        return {**components, "loss_reconstruction": reconstruction, "loss_kl_raw": kl,
                "loss_beta_kl": weighted_kl, "loss_total": reconstruction + weighted_kl}

    def result(self):
        total = sum(self.categories.values(), Counter())
        masks = sum((self.categories[c] for c in DYNAMIC_CATEGORIES), Counter())
        variance_threshold = 1e-4
        latent = self.latents.summary()
        variance = np.asarray(latent["variance"])
        distances = self.distances.summary()
        components = self._loss_components(total["samples"])
        pair_metrics = {category: {
            **_dynamic_metrics(self.pairs[category], self.chains[category]),
            **_masked_accuracy(self.categories[category]),
            "changed_ce": components["dynamic_changed_by_category"][category],
            "unchanged_ce": components["dynamic_unchanged_by_category"][category],
        } for category in DYNAMIC_CATEGORIES}
        return {
            "samples": total["samples"], "cross_entropy": self.ce_sum / total["voxels"],
            **_accuracy(total), "non_finite_prediction": False, "loss_components": components,
            "categories": {k: {**_accuracy(c), "samples": c["samples"]}
                           for k, c in sorted(self.categories.items())},
            "static_preservation": {**_accuracy(self.categories["Static"]),
                "by_biome_group": {k: {**_accuracy(c), "voxels": c["voxels"]}
                                   for k, c in sorted(self.by_biome.items())},
                "by_stratum": {k: {**_accuracy(c), "voxels": c["voxels"]}
                               for k, c in sorted(self.by_stratum.items())}},
            "dynamic_reconstruction": {**_masked_accuracy(masks),
                "by_category": {k: _masked_accuracy(self.categories[k]) for k in DYNAMIC_CATEGORIES}},
            "dynamic": _dynamic_metrics(sum(self.pairs.values(), Counter()), sum(self.chains.values())),
            "dynamic_by_category": pair_metrics,
            "latent": {
                "mean_mu": latent["global_mean"], "std_mu": latent["global_std"],
                "mean_posterior_std": self.posterior_std_sum / (total["samples"] * 16 * 4 ** 3),
                "kl_raw": components["loss_kl_raw"],
                "latent_variance_per_channel": latent["variance"],
                "active_channel_variance_threshold": variance_threshold,
                "active_latent_channels": int((variance > variance_threshold).sum()),
                "adjacent_dynamic_latent_l2": {
                    "count": self.distances.count, "mean": distances["global_mean"],
                    "std": distances["global_std"], "min": distances["global_min"],
                    "max": distances["global_max"], "zero_count": sum(self.zero_distances.values()),
                    "zero_count_by_category": dict(self.zero_distances)},
            },
        }
