"""Historical E00-Plain and E00-Masked comparison."""
from __future__ import annotations

import json
from pathlib import Path


DYNAMIC_CATEGORIES = ("D1", "D2", "D3", "D4")


def _read_curve(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "validation":
            rows.append({
                "global_step": row["global_step"],
                "cross_entropy": row["cross_entropy"],
                "exact_token_accuracy": row["exact_token_accuracy"],
                "non_air_exact_token_accuracy": row["non_air_exact_token_accuracy"],
                "predicted_change_recall": row["dynamic"]["predicted_change_recall"],
                "changed_pair_exact_reconstruction": row["dynamic"][
                    "changed_pair_exact_reconstruction"],
                "loss_components": row.get("loss_components"),
                "dynamic_reconstruction": row.get("dynamic_reconstruction"),
            })
    return rows


def _metric_table(plain, masked):
    paths = {
        "Static Exact Accuracy": lambda row: row["static_preservation"]["exact_token_accuracy"],
        "Static Non-Air Accuracy": lambda row: row["static_preservation"][
            "non_air_exact_token_accuracy"],
        "Dynamic Exact Accuracy": lambda row: row["dynamic_reconstruction"][
            "exact_token_accuracy"],
        "Dynamic Changed Accuracy": lambda row: row["dynamic_reconstruction"][
            "changed_voxel_exact_accuracy"],
        "Dynamic Unchanged Accuracy": lambda row: row["dynamic_reconstruction"][
            "unchanged_voxel_exact_accuracy"],
        "Changed-Pair Exact": lambda row: row["dynamic"][
            "changed_pair_exact_reconstruction"],
        "Unchanged Stability": lambda row: row["dynamic"]["unchanged_stability"],
        "KL Raw": lambda row: row["latent"]["kl_raw"],
        "Active Latent Channels": lambda row: row["latent"]["active_latent_channels"],
    }
    for category in DYNAMIC_CATEGORIES:
        paths[f"{category} Changed Accuracy"] = (
            lambda row, category=category: row["dynamic_reconstruction"][
                "by_category"][category]["changed_voxel_exact_accuracy"])
    result = {}
    for label, getter in paths.items():
        plain_value = getter(plain)
        masked_value = getter(masked)
        result[label] = {
            "E00-Plain": plain_value,
            "E00-Masked": masked_value,
            "Difference": (None if plain_value is None or masked_value is None
                           else masked_value - plain_value),
        }
    return result


def compare(output_root, device_name="cuda:0", batch_size=2):
    from .change_masks import ChangeMaskCache
    from .checkpoint import load_checkpoint
    from .config import load_config
    from .e00 import _load_artifacts, _model, _subset_indices, render_report
    from .evaluate import evaluate_indices, write_json
    from .losses import validate_masked_loss_config
    from .train import resolve_device, set_seed

    output = Path(output_root).expanduser().resolve()
    vocabulary, index, subsets = _load_artifacts(output)
    config = load_config(output / "config.yaml")
    masked_config = load_config(output / "phase_masked/config.yaml")
    loss_config = validate_masked_loss_config(masked_config["loss"])
    validation = _subset_indices(subsets, "e00_b_validation")
    train = _subset_indices(subsets, "e00_b_train")
    masks = ChangeMaskCache(index, vocabulary, train + validation)
    device = resolve_device(device_name)

    def load_and_evaluate(checkpoint, expected_phase=None, expected_loss=None):
        set_seed(int(config["training"]["seed"]))
        model = _model(config, vocabulary).to(device)
        payload = load_checkpoint(
            checkpoint, model, None, vocabulary, index, restore_rng=False,
            map_location=device, expected_phase=expected_phase,
            expected_loss=expected_loss)
        metrics = evaluate_indices(
            model, index, vocabulary, validation, device,
            precision=config["training"].get("precision", "fp32"),
            batch_size=batch_size, change_masks=masks,
            masked_loss_config=loss_config)
        return payload, metrics

    plain_payload, plain = load_and_evaluate(output / "checkpoint_best.pt")
    masked_payload, masked = load_and_evaluate(
        output / "phase_masked/checkpoint_best.pt",
        expected_phase="E00-Masked", expected_loss=loss_config)
    result = {
        "scope": "fixed 160-state E00 validation implementation sanity comparison",
        "warning": "Not a large-scale generalization result.",
        "subset": "e00_b_validation",
        "plain": {
            "checkpoint": str(output / "checkpoint_best.pt"),
            "global_step": int(plain_payload["global_step"]),
            "metrics": plain,
        },
        "masked": {
            "checkpoint": str(output / "phase_masked/checkpoint_best.pt"),
            "global_step": int(masked_payload["global_step"]),
            "metrics": masked,
        },
        "table": _metric_table(plain, masked),
        "initial_metrics": {
            "plain": json.loads((output / "initial_metrics.json").read_text()),
            "masked": json.loads((output / "phase_masked/initial_metrics.json").read_text()),
        },
        "validation_curves": {
            "plain": _read_curve(output / "metrics.jsonl"),
            "masked": _read_curve(output / "phase_masked/metrics.jsonl"),
        },
    }
    write_json(output / "loss_comparison.json", result)
    render_report(output)
    return result


if __name__ == "__main__":
    raise SystemExit("E00-Plain is historical. Use voxel_vae.recon_viewer export for Masked.")
