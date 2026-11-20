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


def _autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _ratio(numerator, denominator):
    return None if not denominator else float(numerator / denominator)


def evaluate_indices(model, index, vocabulary, indices, device, precision="fp32",
                     batch_size=2, save_examples=None):
    indices = list(map(int, indices))
    dataset = VoxelSnapshotDataset(index, vocabulary, indices)
    air_ids = set(vocabulary.air_ids)
    predictions = {}
    ground_truth = {}
    mus = {}
    logvars = {}
    category_counts = defaultdict(lambda: Counter(
        correct=0, voxels=0, non_air_correct=0, non_air_voxels=0))
    ce_sum = 0.0
    non_finite_prediction = False
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            batch_rows = [dataset.read_numpy(item)
                          for item in range(start, min(len(dataset), start + batch_size))]
            target = torch.from_numpy(np.stack([row[0] for row in batch_rows])).to(device)
            sample_ids = [row[1] for row in batch_rows]
            with _autocast(device, precision):
                mu, logvar = model.encode(target)
                logits = model.decode(mu)
                non_finite_prediction |= not bool(torch.isfinite(logits).all().item())
                if non_finite_prediction:
                    raise FloatingPointError("Non-finite reconstruction logits during evaluation")
                ce_sum += float(F.cross_entropy(logits.float(), target.long(),
                                                reduction="sum").item())
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

    total = sum(category_counts.values(), Counter())
    category_metrics = {}
    for category, values in sorted(category_counts.items()):
        category_metrics[category] = {
            "exact_token_accuracy": _ratio(values["correct"], values["voxels"]),
            "non_air_exact_token_accuracy": _ratio(
                values["non_air_correct"], values["non_air_voxels"]),
            "samples": sum(index.records[i].category == category for i in indices),
        }

    change = Counter(changed=0, predicted_changed=0, pair_exact=0,
                     unchanged=0, unchanged_stable=0, pairs=0)
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
        for before_index, after_index in zip(members, members[1:]):
            before, after = ground_truth[before_index], ground_truth[after_index]
            pred_before, pred_after = predictions[before_index], predictions[after_index]
            changed = before != after
            unchanged = ~changed
            change["changed"] += int(changed.sum())
            change["predicted_changed"] += int((changed & (pred_before != pred_after)).sum())
            change["pair_exact"] += int((changed & (pred_before == before)
                                         & (pred_after == after)).sum())
            change["unchanged"] += int(unchanged.sum())
            change["unchanged_stable"] += int((unchanged &
                                                (pred_before == pred_after)).sum())
            change["pairs"] += 1
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
    result = {
        "samples": len(indices),
        "cross_entropy": ce_sum / (len(indices) * 16 ** 3),
        "exact_token_accuracy": _ratio(total["correct"], total["voxels"]),
        "non_air_exact_token_accuracy": _ratio(
            total["non_air_correct"], total["non_air_voxels"]),
        "categories": category_metrics,
        "dynamic": {
            "complete_chains": complete_chains,
            "adjacent_pairs": change["pairs"],
            "predicted_change_recall": _ratio(
                change["predicted_changed"], change["changed"]),
            "changed_pair_exact_reconstruction": _ratio(
                change["pair_exact"], change["changed"]),
            "unchanged_stability": _ratio(
                change["unchanged_stable"], change["unchanged"]),
            "gt_changed_voxels": change["changed"],
        },
        "latent": latent,
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
        )
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)
