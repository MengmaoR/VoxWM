"""Reconstruction, dynamic-change, and latent diagnostics for E00."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import VoxelSnapshotDataset
from .losses import (DYNAMIC_CATEGORIES, masked_batch_normalization,
                     masked_vae_loss, validate_masked_loss_config)


def _autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _ratio(numerator, denominator):
    return None if not denominator else float(numerator / denominator)


def _dynamic_metrics(values, complete_chains):
    return {
        "complete_chains": int(complete_chains),
        "adjacent_pairs": values["pairs"],
        "predicted_change_recall": _ratio(
            values["predicted_changed"], values["changed"]),
        "changed_pair_exact_reconstruction": _ratio(
            values["pair_exact"], values["changed"]),
        "unchanged_stability": _ratio(
            values["unchanged_stable"], values["unchanged"]),
        "gt_changed_voxels": values["changed"],
    }


def evaluate_indices(model, index, vocabulary, indices, device, precision="fp32",
                     batch_size=2, save_examples=None, change_masks=None,
                     masked_loss_config=None, *, dataset=None, allow_empty_changes=False):
    indices = list(map(int, indices))
    if dataset is None:
        dataset = VoxelSnapshotDataset(index, vocabulary, indices)
        dataset_items = range(len(indices))
    else:
        dataset_items = indices
    air_ids = set(vocabulary.air_ids)
    predictions = {}
    ground_truth = {}
    mus = {}
    logvars = {}
    category_counts = defaultdict(lambda: Counter(
        correct=0, voxels=0, non_air_correct=0, non_air_voxels=0))
    static_biome_counts = defaultdict(lambda: Counter(
        correct=0, voxels=0, non_air_correct=0, non_air_voxels=0))
    static_stratum_counts = defaultdict(lambda: Counter(
        correct=0, voxels=0, non_air_correct=0, non_air_voxels=0))
    dynamic_mask_counts = defaultdict(lambda: Counter(
        changed=0, changed_correct=0, unchanged=0, unchanged_correct=0,
        voxels=0, correct=0))
    ce_sum = 0.0
    non_finite_prediction = False
    masked_config = (None if masked_loss_config is None else
                     validate_masked_loss_config(masked_loss_config))
    if masked_config is not None and change_masks is None:
        raise ValueError("Masked evaluation requires ChangeMaskCache")
    normalization = None
    masked_totals = None
    if masked_config is not None:
        categories = [index.records[item].category for item in indices]
        normalization = masked_batch_normalization(
            indices, categories, change_masks, allow_empty_changes=allow_empty_changes)
        masked_totals = {
            key: 0.0 for key in (
                "loss_static", "loss_dynamic_unchanged", "loss_dynamic_changed",
                "loss_reconstruction", "loss_kl_raw", "loss_beta_kl", "loss_total")}
        masked_totals["dynamic_unchanged_by_category"] = {
            category: 0.0 for category in DYNAMIC_CATEGORIES}
        masked_totals["dynamic_changed_by_category"] = {
            category: 0.0 for category in DYNAMIC_CATEGORIES}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset_items), int(batch_size)):
            batch_rows = [dataset.read_numpy(item) for item in dataset_items[start:start + batch_size]]
            target = torch.from_numpy(np.stack([row[0] for row in batch_rows])).to(device)
            sample_ids = [row[1] for row in batch_rows]
            sample_categories = [index.records[item].category for item in sample_ids]
            with _autocast(device, precision):
                mu, logvar = model.encode(target)
                logits = model.decode(mu)
                non_finite_prediction |= not bool(torch.isfinite(logits).all().item())
                if non_finite_prediction:
                    raise FloatingPointError("Non-finite reconstruction logits during evaluation")
                ce_sum += float(F.cross_entropy(logits.float(), target.long(),
                                                reduction="sum").item())
                if masked_config is not None:
                    components = masked_vae_loss(
                        logits.float(), target, mu.float(), logvar.float(),
                        sample_ids, sample_categories, change_masks,
                        normalization, masked_config)
                    for key in masked_totals:
                        if key.startswith("dynamic_") and isinstance(masked_totals[key], dict):
                            for category in DYNAMIC_CATEGORIES:
                                masked_totals[key][category] += float(
                                    components[key][category].item())
                        else:
                            masked_totals[key] += float(components[key].item())
            predicted = logits.argmax(dim=1)
            for offset, sample_index in enumerate(sample_ids):
                gt = target[offset].detach().cpu().numpy().astype(np.uint16)
                pred = predicted[offset].detach().cpu().numpy().astype(np.uint16)
                ground_truth[sample_index] = gt
                predictions[sample_index] = pred
                mus[sample_index] = mu[offset].float().cpu().numpy()
                logvars[sample_index] = logvar[offset].float().cpu().numpy()
                category = index.records[sample_index].category
                correct = pred == gt
                non_air = np.ones(gt.shape, dtype=bool)
                for air_id in air_ids:
                    non_air &= gt != air_id
                counter = category_counts[category]
                counter["correct"] += int(correct.sum())
                counter["voxels"] += int(gt.size)
                counter["non_air_correct"] += int((correct & non_air).sum())
                counter["non_air_voxels"] += int(non_air.sum())
                if category == "Static":
                    for group, key in ((static_biome_counts, index.records[sample_index].biome_group
                                        or "unspecified"),
                                       (static_stratum_counts, index.records[sample_index].stratum
                                        or "unspecified")):
                        values = group[key]
                        values["correct"] += int(correct.sum())
                        values["voxels"] += int(gt.size)
                        values["non_air_correct"] += int((correct & non_air).sum())
                        values["non_air_voxels"] += int(non_air.sum())
                elif change_masks is not None:
                    mask = change_masks.changed(sample_index)
                    unchanged_mask = ~mask
                    values = dynamic_mask_counts[category]
                    values["changed"] += int(mask.sum())
                    values["changed_correct"] += int((correct & mask).sum())
                    values["unchanged"] += int(unchanged_mask.sum())
                    values["unchanged_correct"] += int((correct & unchanged_mask).sum())
                    values["voxels"] += int(gt.size)
                    values["correct"] += int(correct.sum())

    total = sum(category_counts.values(), Counter())
    category_metrics = {}
    for category, values in sorted(category_counts.items()):
        category_metrics[category] = {
            "exact_token_accuracy": _ratio(values["correct"], values["voxels"]),
            "non_air_exact_token_accuracy": _ratio(
                values["non_air_correct"], values["non_air_voxels"]),
            "samples": sum(index.records[i].category == category for i in indices),
        }

    def grouped_accuracy(groups):
        return {
            name: {
                "exact_token_accuracy": _ratio(values["correct"], values["voxels"]),
                "non_air_exact_token_accuracy": _ratio(
                    values["non_air_correct"], values["non_air_voxels"]),
                "voxels": values["voxels"],
            }
            for name, values in sorted(groups.items())
        }

    dynamic_total = sum(dynamic_mask_counts.values(), Counter())
    dynamic_reconstruction = None
    if dynamic_mask_counts:
        dynamic_reconstruction = {
            "exact_token_accuracy": _ratio(dynamic_total["correct"], dynamic_total["voxels"]),
            "changed_voxel_exact_accuracy": _ratio(
                dynamic_total["changed_correct"], dynamic_total["changed"]),
            "unchanged_voxel_exact_accuracy": _ratio(
                dynamic_total["unchanged_correct"], dynamic_total["unchanged"]),
            "changed_voxels": dynamic_total["changed"],
            "unchanged_voxels": dynamic_total["unchanged"],
            "by_category": {
                category: {
                    "exact_token_accuracy": _ratio(values["correct"], values["voxels"]),
                    "changed_voxel_exact_accuracy": _ratio(
                        values["changed_correct"], values["changed"]),
                    "unchanged_voxel_exact_accuracy": _ratio(
                        values["unchanged_correct"], values["unchanged"]),
                    "changed_voxels": values["changed"],
                    "unchanged_voxels": values["unchanged"],
                }
                for category, values in sorted(dynamic_mask_counts.items())
            },
        }

    change = Counter(changed=0, predicted_changed=0, pair_exact=0,
                     unchanged=0, unchanged_stable=0, pairs=0)
    change_by_category = defaultdict(lambda: Counter(
        changed=0, predicted_changed=0, pair_exact=0,
        unchanged=0, unchanged_stable=0, pairs=0))
    chains_by_category = Counter()
    latent_distances = []
    groups = defaultdict(list)
    for sample_index in indices:
        row = index.records[sample_index]
        if row.chain_id is not None:
            groups[(row.root_index, row.shard, row.chain_id)].append(sample_index)
    complete_chains = 0
    for members in groups.values():
        members.sort(key=lambda i: index.records[i].chain_index)
        if len(members) != 5 or [index.records[i].chain_index for i in members] != list(range(5)):
            continue
        complete_chains += 1
        category = index.records[members[0]].category
        chains_by_category[category] += 1
        for before_index, after_index in zip(members, members[1:]):
            before, after = ground_truth[before_index], ground_truth[after_index]
            pred_before, pred_after = predictions[before_index], predictions[after_index]
            changed = before != after
            unchanged = ~changed
            increments = {
                "changed": int(changed.sum()),
                "predicted_changed": int((changed & (pred_before != pred_after)).sum()),
                "pair_exact": int((changed & (pred_before == before)
                                    & (pred_after == after)).sum()),
                "unchanged": int(unchanged.sum()),
                "unchanged_stable": int((unchanged &
                                          (pred_before == pred_after)).sum()),
                "pairs": 1,
            }
            change.update(increments)
            change_by_category[category].update(increments)
            latent_distances.append(float(np.linalg.norm(
                mus[before_index] - mus[after_index])))

    mu_stack = np.stack([mus[index_] for index_ in indices])
    logvar_stack = np.stack([logvars[index_] for index_ in indices])
    channel_variance = mu_stack.var(axis=(0, 2, 3, 4))
    threshold = 1e-4
    latent = {
        "mean_mu": float(mu_stack.mean()),
        "std_mu": float(mu_stack.std()),
        "mean_posterior_std": float(np.exp(0.5 * logvar_stack).mean()),
        "kl_raw": float((-0.5 * (1 + logvar_stack - mu_stack ** 2
                                  - np.exp(logvar_stack))).reshape(len(indices), -1)
                        .sum(axis=1).mean()),
        "latent_variance_per_channel": channel_variance.tolist(),
        "active_channel_variance_threshold": threshold,
        "active_latent_channels": int((channel_variance > threshold).sum()),
        "adjacent_dynamic_latent_l2": {
            "count": len(latent_distances),
            "mean": None if not latent_distances else float(np.mean(latent_distances)),
            "std": None if not latent_distances else float(np.std(latent_distances)),
            "min": None if not latent_distances else float(np.min(latent_distances)),
            "max": None if not latent_distances else float(np.max(latent_distances)),
        },
    }
    if allow_empty_changes and masked_totals is not None:
        for kind in ("changed", "unchanged"):
            for category, count in normalization[kind].items():
                if not count:
                    masked_totals[f"dynamic_{kind}_by_category"][category] = None
    result = {
        "samples": len(indices),
        "cross_entropy": ce_sum / (len(indices) * 16 ** 3),
        "exact_token_accuracy": _ratio(total["correct"], total["voxels"]),
        "non_air_exact_token_accuracy": _ratio(
            total["non_air_correct"], total["non_air_voxels"]),
        "categories": category_metrics,
        "static_preservation": {
            "exact_token_accuracy": category_metrics.get("Static", {}).get(
                "exact_token_accuracy"),
            "non_air_exact_token_accuracy": category_metrics.get("Static", {}).get(
                "non_air_exact_token_accuracy"),
            "by_biome_group": grouped_accuracy(static_biome_counts),
            "by_stratum": grouped_accuracy(static_stratum_counts),
        },
        "dynamic_reconstruction": dynamic_reconstruction,
        "dynamic": _dynamic_metrics(change, complete_chains),
        "dynamic_by_category": {
            category: _dynamic_metrics(change_by_category[category], count)
            for category, count in sorted(chains_by_category.items())
        },
        "latent": latent,
        "loss_components": masked_totals,
        "change_masks": (None if change_masks is None else
                         change_masks.statistics(indices)),
        "non_finite_prediction": non_finite_prediction,
    }
    if save_examples is not None:
        path = Path(save_examples)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [index.records[i] for i in indices]
        np.savez_compressed(
            path,
            gt_tokens=np.stack([ground_truth[i] for i in indices]),
            predicted_tokens=np.stack([predictions[i] for i in indices]),
            category=np.asarray([row.category for row in rows]),
            chain_id=np.asarray([row.chain_id or "" for row in rows]),
            chain_index=np.asarray([-1 if row.chain_index is None else row.chain_index
                                    for row in rows], dtype=np.int16),
            snapshot_id=np.asarray([row.snapshot_index for row in rows], dtype=np.int32),
            shard=np.asarray([row.shard for row in rows], dtype=np.int16),
            dataset_root=np.asarray([index.roots[row.root_index] for row in rows]),
            world_seed=np.asarray([row.world_seed for row in rows], dtype=np.int64),
            split=np.asarray([row.split for row in rows]),
            biome_group=np.asarray([row.biome_group or "" for row in rows]),
            stratum=np.asarray([row.stratum or "" for row in rows]),
            changed_mask=np.stack([
                (change_masks.changed(i) if change_masks is not None
                 and index.records[i].category in DYNAMIC_CATEGORIES
                 else np.zeros((16, 16, 16), dtype=bool))
                for i in indices]),
        )
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)
