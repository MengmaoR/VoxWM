"""Step-based E00 trainer with balanced effective batches and strict resume."""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import VoxelSnapshotDataset
from .early_stopping import (EarlyStoppingState, state_from_metrics,
                             validate_early_stopping_config)
from .evaluate import evaluate_indices, write_json
from .losses import (DYNAMIC_CATEGORIES, accuracy_counts,
                     masked_batch_normalization, masked_vae_loss,
                     vae_loss, validate_masked_loss_config)
from .sampler import BalancedBatchSampler
from .training_curves import write_training_curves


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(value="auto"):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        prefix = "\n" if value.get("event") == "resume" else ""
        stream.write(prefix + json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _tokens(dataset, global_indices, device):
    rows = [dataset.read_numpy(index)[0] for index in global_indices]
    return torch.from_numpy(np.stack(rows)).to(device)


def _gradient_status(model):
    groups = {
        "embedding": [model.embedding.weight],
        "encoder": list(model.encoder_input.parameters()) + list(model.moments_output.parameters()),
        "decoder": list(model.decoder_hidden.parameters()) + list(model.decoder_logits.parameters()),
    }
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "present": bool(gradients),
            "finite": bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients),
            "nonzero": bool(gradients) and any(torch.count_nonzero(g).item() > 0 for g in gradients),
        }
    return result


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def save_initial_embedding(path, model, vocabulary, seed):
    """Persist the small step-0 codebook needed for learned-drift analysis."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        weight=model.embedding.weight.detach().float().cpu().numpy(),
        token_id=np.arange(vocabulary.K, dtype=np.int32),
        vocabulary_sha256=np.asarray(vocabulary.sha256),
        seed=np.asarray(int(seed), dtype=np.int64),
    )
    temporary.replace(path)


def embedding_drift(path, model):
    """Summarize how far each learned token row moved from step 0."""
    with np.load(path, allow_pickle=False) as archive:
        initial = np.asarray(archive["weight"], dtype=np.float32)
    current = model.embedding.weight.detach().float().cpu().numpy()
    if initial.shape != current.shape:
        raise ValueError(
            f"Initial embedding shape {initial.shape} != current {current.shape}")
    row_l2 = np.linalg.norm(current - initial, axis=1)
    denominator = np.linalg.norm(initial, axis=1) * np.linalg.norm(current, axis=1)
    cosine = np.divide(
        np.sum(initial * current, axis=1), denominator,
        out=np.zeros_like(denominator), where=denominator > 0)
    return {
        "rows": int(row_l2.size),
        "mean_row_l2": float(row_l2.mean()),
        "median_row_l2": float(np.median(row_l2)),
        "max_row_l2": float(row_l2.max()),
        "mean_initial_to_final_cosine": float(cosine.mean()),
        "rows_l2_above_1e-6": int((row_l2 > 1e-6).sum()),
    }


def fit(model, index, vocabulary, train_indices, evaluation_indices, output_dir,
        config, phase, balanced=True, resume=None, validation_indices=None):
    """Historical Plain trainer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    model.to(device)
    initial_embedding_path = output_dir / "embedding_initial.npz"
    if not resume:
        save_initial_embedding(initial_embedding_path, model, vocabulary, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    global_step = 0
    best = math.inf
    if resume:
        payload = load_checkpoint(resume, model, optimizer, vocabulary, index,
                                  map_location=device)
        global_step = int(payload["global_step"])
        best = float(payload.get("best_metric", math.inf))

    dataset = VoxelSnapshotDataset(index, vocabulary)
    sampler = (BalancedBatchSampler(index, train_indices, seed=seed)
               if balanced else None)
    if not balanced and not train_indices:
        raise ValueError("Empty fixed overfit subset")
    max_steps = int(config["max_steps"])
    micro_batch = int(config.get("micro_batch_size", 2))
    if not 1 <= micro_batch <= 64 or 64 % micro_batch:
        raise ValueError("micro_batch_size must divide effective batch size 64")
    beta = float(config.get("beta", 1e-6))
    log_every = int(config.get("log_every", 25))
    validate_every = int(config.get("validate_every", 250))
    evaluation_batch = int(config.get("evaluation_batch_size", micro_batch))
    metrics_path = output_dir / "metrics.jsonl"

    initial = evaluate_indices(model, index, vocabulary, evaluation_indices, device,
                               precision=precision, batch_size=evaluation_batch)
    initial_validation = (None if validation_indices is None else evaluate_indices(
        model, index, vocabulary, validation_indices, device,
        precision=precision, batch_size=evaluation_batch))
    write_json(output_dir / "initial_metrics.json", {
        "overfit": initial, "validation": initial_validation})
    started = time.monotonic()
    gradient_check = None
    while global_step < max_steps:
        if balanced:
            effective = sampler.batch(global_step)
        else:
            repeats = (64 + len(train_indices) - 1) // len(train_indices)
            effective = (list(train_indices) * repeats)[:64]
            np.random.default_rng(seed + global_step).shuffle(effective)
        optimizer.zero_grad(set_to_none=True)
        aggregate = {"recon_ce": 0.0, "kl_raw": 0.0, "beta_kl": 0.0,
                     "total_loss": 0.0, "correct": 0, "voxels": 0,
                     "non_air_correct": 0, "non_air_voxels": 0}
        chunks = [effective[start:start + micro_batch]
                  for start in range(0, len(effective), micro_batch)]
        model.train()
        for selected in chunks:
            target = _tokens(dataset, selected, device)
            with _autocast(device, precision):
                output = model(target)
                losses = vae_loss(output["logits"].float(), target, output["mu"].float(),
                                  output["logvar"].float(), beta=beta)
            if not all(torch.isfinite(losses[key]).item()
                       for key in ("total_loss", "recon_ce", "kl_raw", "beta_kl")):
                raise FloatingPointError(f"Non-finite loss at step {global_step}")
            if not torch.isfinite(output["mu"]).all() or not torch.isfinite(output["logvar"]).all():
                raise FloatingPointError(f"Non-finite posterior at step {global_step}")
            (losses["total_loss"] / len(chunks)).backward()
            weight = len(selected) / 64
            for key in ("recon_ce", "kl_raw", "beta_kl", "total_loss"):
                aggregate[key] += float(losses[key].detach().item()) * weight
            counts = accuracy_counts(output["logits"], target, vocabulary.air_ids)
            for key, value in counts.items():
                aggregate[key] += value
        if gradient_check is None:
            gradient_check = _gradient_status(model)
            if not all(row["present"] and row["finite"] and row["nonzero"]
                       for row in gradient_check.values()):
                raise FloatingPointError(f"Key gradient check failed: {gradient_check}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {global_step}")
        optimizer.step()
        global_step += 1

        if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
            row = {
                "phase": phase, "global_step": global_step,
                "recon_ce": aggregate["recon_ce"], "kl_raw": aggregate["kl_raw"],
                "beta_kl": aggregate["beta_kl"], "total_loss": aggregate["total_loss"],
                "exact_token_accuracy": aggregate["correct"] / aggregate["voxels"],
                "non_air_exact_token_accuracy": (
                    None if not aggregate["non_air_voxels"] else
                    aggregate["non_air_correct"] / aggregate["non_air_voxels"]),
                "gradient_norm_before_clip": float(gradient_norm.item()),
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_jsonl(metrics_path, row)

        if global_step % validate_every == 0 or global_step == max_steps:
            evaluation = evaluate_indices(
                model, index, vocabulary, evaluation_indices, device,
                precision=precision, batch_size=evaluation_batch)
            _append_jsonl(metrics_path, {
                "phase": phase, "global_step": global_step,
                "event": "overfit_evaluation", **evaluation})
            validation = (None if validation_indices is None else evaluate_indices(
                model, index, vocabulary, validation_indices, device,
                precision=precision, batch_size=evaluation_batch))
            if validation is not None:
                _append_jsonl(metrics_path, {
                    "phase": phase, "global_step": global_step,
                    "event": "validation", **validation})
            score = float((validation or evaluation)["cross_entropy"])
            save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer,
                            global_step, config, vocabulary, index, best, phase)
            if score < best:
                best = score
                save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer,
                                global_step, config, vocabulary, index, best, phase)

    final = evaluate_indices(
        model, index, vocabulary, evaluation_indices, device,
        precision=precision, batch_size=evaluation_batch,
        save_examples=output_dir / "recon_examples.npz")
    final_validation = (None if validation_indices is None else evaluate_indices(
        model, index, vocabulary, validation_indices, device,
        precision=precision, batch_size=evaluation_batch))
    write_json(output_dir / "latent_statistics.json", final["latent"])
    result = {
        "phase": phase,
        "global_step": global_step,
        "training_seconds": time.monotonic() - started,
        "device": str(device),
        "precision": precision,
        "parameter_count": parameter_count(model),
        "beta": beta,
        "initial_embedding": (str(initial_embedding_path)
                              if initial_embedding_path.is_file() else None),
        "embedding_drift": (embedding_drift(initial_embedding_path, model)
                            if initial_embedding_path.is_file() else None),
        "gradient_check": gradient_check,
        "initial": initial,
        "final": final,
        "initial_validation": initial_validation,
        "validation": final_validation,
    }
    write_json(output_dir / "final_metrics.json", result)
    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer,
                    global_step, config, vocabulary, index, best, phase)
    return result


def fit_masked(model, index, vocabulary, train_indices, evaluation_indices,
               validation_indices, change_masks, output_dir, config,
               resume=None, run_metadata=None):
    """Joint Static/Dynamic masked E00 training with exact batch denominators."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    loss_config = validate_masked_loss_config(config["loss"])
    early_config = validate_early_stopping_config(config.get("early_stopping", {}))
    if abs(float(config.get("beta", loss_config["beta_kl"]))
           - loss_config["beta_kl"]) > 1e-15:
        raise ValueError("Training beta and masked beta_kl must remain identical")
    model.to(device)
    initial_embedding_path = output_dir / "embedding_initial.npz"
    if not resume:
        save_initial_embedding(initial_embedding_path, model, vocabulary, seed)
    elif not initial_embedding_path.is_file():
        raise ValueError("Masked resume requires its original embedding_initial.npz")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)))
    global_step = 0
    payload = None
    if resume:
        # Keep CPU RNG state on CPU, including when the model/optimizer are on CUDA.
        payload = load_checkpoint(
            resume, model, optimizer, vocabulary, index,
            expected_phase="E00-Masked", expected_loss=loss_config)
        global_step = int(payload["global_step"])

    dataset = VoxelSnapshotDataset(index, vocabulary)
    sampler = BalancedBatchSampler(index, train_indices, seed=seed)
    max_steps = int(config["max_steps"])
    micro_batch = int(config.get("micro_batch_size", 2))
    if not 1 <= micro_batch <= 64 or 64 % micro_batch:
        raise ValueError("micro_batch_size must divide effective batch size 64")
    log_every = int(config.get("log_every", 25))
    validate_every = int(config.get("validate_every", 500))
    evaluation_batch = int(config.get("evaluation_batch_size", micro_batch))
    metrics_path = output_dir / "metrics.jsonl"
    early_path = output_dir / "early_stopping.json"
    early_state = EarlyStoppingState()
    if payload is not None:
        saved_state = payload.get("early_stopping")
        if saved_state and saved_state["config"] == early_config:
            early_state = EarlyStoppingState.from_dict(saved_state)
        else:
            # Old checkpoints did not store patience. Apply it to the next live check.
            early_state = state_from_metrics(
                metrics_path, {**early_config, "enabled": False}, global_step)
            saved_best = payload.get("best_metric")
            if (saved_best is not None and math.isfinite(saved_best)
                    and (early_state.best_metric is None or saved_best < early_state.best_metric)):
                early_state.best_metric = saved_best
                early_state.best_step = None
                early_state.patience_reference = saved_best
        _append_jsonl(metrics_path, {
            "phase": "E00-Masked", "global_step": global_step, "event": "resume"})
        del payload
    metadata_path = output_dir / "run_metadata.json"
    metadata = dict(run_metadata or {})
    metadata.update({
        "status": "RUNNING", "phase": "E00-Masked", "seed": seed,
        "device": str(device), "precision": precision, "loss": loss_config,
        "early_stopping": early_config,
        "global_step": global_step, "max_steps": max_steps,
    })
    write_json(metadata_path, metadata)
    write_json(early_path, early_state.to_dict(early_config))
    write_training_curves(metrics_path, output_dir)

    if global_step == 0:
        initial = evaluate_indices(
            model, index, vocabulary, evaluation_indices, device,
            precision=precision, batch_size=evaluation_batch,
            change_masks=change_masks, masked_loss_config=loss_config)
        initial_validation = evaluate_indices(
            model, index, vocabulary, validation_indices, device,
            precision=precision, batch_size=evaluation_batch,
            change_masks=change_masks, masked_loss_config=loss_config)
        write_json(output_dir / "initial_metrics.json", {
            "overfit": initial, "validation": initial_validation})
    else:
        saved_initial = json.loads((output_dir / "initial_metrics.json").read_text())
        initial = saved_initial["overfit"]
        initial_validation = saved_initial["validation"]

    loss_statistics_path = output_dir / "loss_statistics.json"
    loss_statistics = {
        "objective": loss_config,
        "train_masks": change_masks.statistics(evaluation_indices),
        "validation_masks": change_masks.statistics(validation_indices),
        "gradient_accumulation": {
            "effective_batch_size": 64,
            "micro_batch_size": micro_batch,
            "micro_batches": 64 // micro_batch,
            "normalization": "full-effective-batch denominators; no micro-batch divisor",
        },
        "early_stopping": early_config,
    }
    write_json(loss_statistics_path, loss_statistics)

    started = time.monotonic()
    gradient_check = None
    last_validation = None
    while global_step < max_steps and not early_state.stopped:
        effective = sampler.batch(global_step)
        categories = [index.records[item].category for item in effective]
        normalization = masked_batch_normalization(
            effective, categories, change_masks)
        optimizer.zero_grad(set_to_none=True)
        scalar_keys = (
            "loss_static", "loss_dynamic_unchanged", "loss_dynamic_changed",
            "loss_reconstruction", "loss_kl_raw", "loss_beta_kl", "loss_total")
        aggregate = {key: 0.0 for key in scalar_keys}
        aggregate["dynamic_unchanged_by_category"] = {
            category: 0.0 for category in DYNAMIC_CATEGORIES}
        aggregate["dynamic_changed_by_category"] = {
            category: 0.0 for category in DYNAMIC_CATEGORIES}
        aggregate.update(correct=0, voxels=0, non_air_correct=0, non_air_voxels=0)
        chunks = [effective[start:start + micro_batch]
                  for start in range(0, len(effective), micro_batch)]
        model.train()
        for selected in chunks:
            selected_categories = [index.records[item].category for item in selected]
            target = _tokens(dataset, selected, device)
            with _autocast(device, precision):
                output = model(target)
                losses = masked_vae_loss(
                    output["logits"].float(), target, output["mu"].float(),
                    output["logvar"].float(), selected, selected_categories,
                    change_masks, normalization, loss_config)
            if not all(torch.isfinite(losses[key]).item() for key in scalar_keys):
                raise FloatingPointError(f"Non-finite masked loss at step {global_step}")
            if (not torch.isfinite(output["mu"]).all()
                    or not torch.isfinite(output["logvar"]).all()):
                raise FloatingPointError(f"Non-finite posterior at step {global_step}")
            losses["loss_total"].backward()
            for key in scalar_keys:
                aggregate[key] += float(losses[key].detach().item())
            for group in ("dynamic_unchanged_by_category",
                          "dynamic_changed_by_category"):
                for category in DYNAMIC_CATEGORIES:
                    aggregate[group][category] += float(
                        losses[group][category].detach().item())
            counts = accuracy_counts(output["logits"], target, vocabulary.air_ids)
            for key, value in counts.items():
                aggregate[key] += value
        if gradient_check is None:
            gradient_check = _gradient_status(model)
            if not all(row["present"] and row["finite"] and row["nonzero"]
                       for row in gradient_check.values()):
                raise FloatingPointError(f"Key gradient check failed: {gradient_check}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {global_step}")
        optimizer.step()
        global_step += 1

        validate_now = global_step % validate_every == 0 or global_step == max_steps
        log_now = global_step == 1 or global_step % log_every == 0 or validate_now
        if log_now:
            row = {
                "phase": "E00-Masked", "global_step": global_step,
                **{key: aggregate[key] for key in scalar_keys},
                "dynamic_unchanged_by_category": aggregate[
                    "dynamic_unchanged_by_category"],
                "dynamic_changed_by_category": aggregate[
                    "dynamic_changed_by_category"],
                "exact_token_accuracy": aggregate["correct"] / aggregate["voxels"],
                "non_air_exact_token_accuracy": (
                    None if not aggregate["non_air_voxels"] else
                    aggregate["non_air_correct"] / aggregate["non_air_voxels"]),
                "gradient_norm_before_clip": float(gradient_norm.item()),
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_jsonl(metrics_path, row)

        if validate_now:
            evaluation = evaluate_indices(
                model, index, vocabulary, evaluation_indices, device,
                precision=precision, batch_size=evaluation_batch,
                change_masks=change_masks, masked_loss_config=loss_config)
            _append_jsonl(metrics_path, {
                "phase": "E00-Masked", "global_step": global_step,
                "event": "overfit_evaluation", **evaluation})
            last_validation = evaluate_indices(
                model, index, vocabulary, validation_indices, device,
                precision=precision, batch_size=evaluation_batch,
                change_masks=change_masks, masked_loss_config=loss_config)
            _append_jsonl(metrics_path, {
                "phase": "E00-Masked", "global_step": global_step,
                "event": "validation", **last_validation})
            score = last_validation["loss_components"]["loss_total"]
            new_best, _ = early_state.observe(global_step, score, early_config)
            snapshot = early_state.to_dict(early_config)
            if new_best:
                save_checkpoint(
                    output_dir / "checkpoint_best.pt", model, optimizer,
                    global_step, config, vocabulary, index, early_state.best_metric,
                    "E00-Masked", early_stopping=snapshot)
            save_checkpoint(
                output_dir / "checkpoint_last.pt", model, optimizer,
                global_step, config, vocabulary, index, early_state.best_metric,
                "E00-Masked", early_stopping=snapshot)
            write_json(early_path, snapshot)
        if log_now:
            write_training_curves(metrics_path, output_dir)

    final = evaluate_indices(
        model, index, vocabulary, evaluation_indices, device,
        precision=precision, batch_size=evaluation_batch,
        save_examples=output_dir / "recon_examples.npz",
        change_masks=change_masks, masked_loss_config=loss_config)
    final_validation = last_validation or evaluate_indices(
        model, index, vocabulary, validation_indices, device,
        precision=precision, batch_size=evaluation_batch,
        change_masks=change_masks, masked_loss_config=loss_config)
    write_json(output_dir / "latent_statistics.json", final["latent"])
    loss_statistics.update({
        "final_train": final["loss_components"],
        "final_validation": final_validation["loss_components"],
    })
    write_json(loss_statistics_path, loss_statistics)
    result = {
        "phase": "E00-Masked", "global_step": global_step,
        "training_seconds": time.monotonic() - started,
        "device": str(device), "precision": precision,
        "parameter_count": parameter_count(model), "loss": loss_config,
        "initial_embedding": str(initial_embedding_path),
        "embedding_drift": embedding_drift(initial_embedding_path, model),
        "stopped_early": early_state.stopped,
        "early_stopping": early_state.to_dict(early_config),
        "gradient_check": gradient_check,
        "initial": initial, "final": final,
        "initial_validation": initial_validation,
        "validation": final_validation,
    }
    write_json(output_dir / "final_metrics.json", result)
    metadata.update({
        "status": ("EARLY_STOPPED" if early_state.stopped else "COMPLETE"),
        "global_step": global_step,
        "training_seconds": result["training_seconds"],
        "best_validation_loss": early_state.best_metric,
        "best_step": early_state.best_step,
        "stop_reason": early_state.stop_reason,
    })
    write_json(metadata_path, metadata)
    return result
