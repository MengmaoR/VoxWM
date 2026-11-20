"""Plain and static/dynamic-masked VAE reconstruction objectives."""
import torch
import torch.nn.functional as F


DYNAMIC_CATEGORIES = ("D1", "D2", "D3", "D4")


def vae_loss(logits, target, mu, logvar, beta=1e-6):
    reconstruction = F.cross_entropy(logits, target.long())
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.flatten(1).sum(1).mean()
    beta_kl = float(beta) * kl
    total = reconstruction + beta_kl
    return {"total_loss": total, "recon_ce": reconstruction,
            "kl_raw": kl, "beta_kl": beta_kl}


def validate_masked_loss_config(config):
    required = {
        "objective": "static_dynamic_masked_v1",
        "lambda_static": 0.5,
        "lambda_dynamic": 0.5,
        "alpha_changed": 0.5,
        "beta_kl": 1e-6,
    }
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Masked loss config is missing: " + ", ".join(missing))
    if str(config["objective"]) != required["objective"]:
        raise ValueError("Unsupported masked loss objective")
    for key in required.keys() - {"objective"}:
        if abs(float(config[key]) - required[key]) > 1e-15:
            raise ValueError(f"Masked objective fixes {key}={required[key]}")
    return {key: (str(value) if key == "objective" else float(value))
            for key, value in config.items() if key in required}


def masked_batch_normalization(sample_indices, categories, changed_masks):
    """Compute full-effective-batch denominators before micro-batching."""
    sample_indices = list(map(int, sample_indices))
    categories = list(categories)
    if len(sample_indices) != len(categories):
        raise ValueError("sample_indices/categories length mismatch")
    static = sum(category == "Static" for category in categories)
    changed = {category: 0 for category in DYNAMIC_CATEGORIES}
    unchanged = {category: 0 for category in DYNAMIC_CATEGORIES}
    for sample_index, category in zip(sample_indices, categories):
        if category == "Static":
            continue
        if category not in changed:
            raise ValueError(f"Unknown masked-loss category: {category}")
        mask = torch.as_tensor(changed_masks[sample_index], dtype=torch.bool)
        if tuple(mask.shape) != (16, 16, 16) or not bool(mask.any()):
            raise ValueError(f"Invalid/empty changed mask for sample {sample_index}")
        changed[category] += 1
        if bool((~mask).any()):
            unchanged[category] += 1
    if static <= 0:
        raise ValueError("Masked loss effective batch has no Static samples")
    if any(changed[category] <= 0 for category in DYNAMIC_CATEGORIES):
        raise ValueError("Masked loss requires D1-D4 in every effective batch")
    if any(unchanged[category] <= 0 for category in DYNAMIC_CATEGORIES):
        raise ValueError("A dynamic category has no valid Unchanged sample")
    return {
        "batch_size": len(sample_indices),
        "static": static,
        "changed": changed,
        "unchanged": unchanged,
    }


def masked_vae_loss(logits, target, mu, logvar, sample_indices, categories,
                    changed_masks, normalization, config):
    """Return additive contributions normalized over the full effective batch."""
    config = validate_masked_loss_config(config)
    sample_indices = list(map(int, sample_indices))
    categories = list(categories)
    if len(sample_indices) != len(categories) or len(sample_indices) != len(target):
        raise ValueError("Masked loss micro-batch metadata mismatch")
    per_voxel = F.cross_entropy(logits, target.long(), reduction="none").flatten(1)
    kl_per_sample = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).flatten(1).sum(1)
    zero = per_voxel.sum() * 0.0
    loss_static = zero
    unchanged_by_category = {category: zero for category in DYNAMIC_CATEGORIES}
    changed_by_category = {category: zero for category in DYNAMIC_CATEGORIES}

    for offset, (sample_index, category) in enumerate(zip(sample_indices, categories)):
        if category == "Static":
            loss_static = loss_static + per_voxel[offset].mean() / normalization["static"]
            continue
        if category not in DYNAMIC_CATEGORIES:
            raise ValueError(f"Unknown masked-loss category: {category}")
        mask = torch.as_tensor(changed_masks[sample_index], dtype=torch.bool,
                               device=per_voxel.device).flatten()
        if not bool(mask.any()):
            raise ValueError(f"Empty Changed mask for sample {sample_index}")
        changed_by_category[category] = (
            changed_by_category[category]
            + per_voxel[offset][mask].mean() / normalization["changed"][category])
        if bool((~mask).any()):
            unchanged_by_category[category] = (
                unchanged_by_category[category]
                + per_voxel[offset][~mask].mean() / normalization["unchanged"][category])

    loss_dynamic_unchanged = sum(unchanged_by_category.values(), zero) / 4.0
    loss_dynamic_changed = sum(changed_by_category.values(), zero) / 4.0
    loss_reconstruction = (
        config["lambda_static"] * loss_static
        + config["lambda_dynamic"] * (
            (1.0 - config["alpha_changed"]) * loss_dynamic_unchanged
            + config["alpha_changed"] * loss_dynamic_changed))
    loss_kl_raw = kl_per_sample.sum() / normalization["batch_size"]
    loss_beta_kl = config["beta_kl"] * loss_kl_raw
    return {
        "loss_static": loss_static,
        "loss_dynamic_unchanged": loss_dynamic_unchanged,
        "loss_dynamic_changed": loss_dynamic_changed,
        "loss_reconstruction": loss_reconstruction,
        "loss_kl_raw": loss_kl_raw,
        "loss_beta_kl": loss_beta_kl,
        "loss_total": loss_reconstruction + loss_beta_kl,
        "dynamic_unchanged_by_category": unchanged_by_category,
        "dynamic_changed_by_category": changed_by_category,
    }


def accuracy_counts(logits, target, air_ids):
    prediction = logits.argmax(dim=1)
    correct = prediction.eq(target)
    non_air = torch.ones_like(target, dtype=torch.bool)
    for air_id in air_ids:
        non_air &= target.ne(int(air_id))
    return {
        "correct": int(correct.sum().item()),
        "voxels": int(target.numel()),
        "non_air_correct": int((correct & non_air).sum().item()),
        "non_air_voxels": int(non_air.sum().item()),
    }
