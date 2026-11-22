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

from .checkpoint import load_checkpoint, restore_training_rng, rng_state, save_checkpoint
from .config import save_config
from .dataset import VoxelSnapshotDataset
from .early_stopping import (EarlyStoppingState, state_from_metrics,
                             validate_early_stopping_config)
from .evaluate import evaluate_indices, write_json
from .losses import (DYNAMIC_CATEGORIES, accuracy_counts,
                     masked_batch_normalization, masked_vae_loss,
                     vae_loss, validate_masked_loss_config)
from .sampler import BalancedBatchSampler, BalancedEpochSampler
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
               resume=None, run_metadata=None, *, phase="E00-Masked", group=None, epochs=None,
               previous_prepared_data=None, no_early_stop=False):
    """Joint Static/Dynamic masked training with exact batch denominators."""
    output_dir = Path(output_dir)
    primary = group is None or group.primary
    rank = 0 if group is None else group.rank
    world_size = 1 if group is None else group.world_size
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed + rank)
    device = resolve_device(config.get("device", "auto")) if group is None else group.device
    scoped_rng = phase == "E01-Masked"
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    loss_config = validate_masked_loss_config(config["loss"])
    early_config = validate_early_stopping_config(config.get("early_stopping", {}))
    if no_early_stop:
        early_config["enabled"] = False
    if abs(float(config.get("beta", loss_config["beta_kl"]))
           - loss_config["beta_kl"]) > 1e-15:
        raise ValueError("Training beta and masked beta_kl must remain identical")
    model.to(device)
    initial_embedding_path = output_dir / "embedding_initial.npz"
    if not resume and primary:
        save_initial_embedding(initial_embedding_path, model, vocabulary, seed)
    if resume and not initial_embedding_path.is_file():
        raise ValueError("Masked resume requires its original embedding_initial.npz")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)))
    global_step = 0
    payload = None
    split_changed = False
    if resume:
        # Keep CPU RNG state on CPU, including when the model/optimizer are on CUDA.
        expected_config = None
        if phase == "E01-Masked":
            expected_config = {key: value for key, value in config.items() if key != "device"}
        payload = load_checkpoint(
            resume, model, optimizer, vocabulary, index,
            restore_rng=not scoped_rng, expected_phase=phase, expected_loss=loss_config,
            expected_config=expected_config, previous_prepared_data=previous_prepared_data)
        global_step = int(payload["global_step"])
        split_changed = (previous_prepared_data is not None
                         and payload["config"]["prepared_data"] == previous_prepared_data)
        if split_changed and epochs is None:
            raise ValueError("Use --epochs to start a new epoch schedule after rebalancing E01")

    dataset = VoxelSnapshotDataset(index, vocabulary)
    max_steps = int(config["max_steps"])
    saved_plan = payload.get("epoch_plan") if payload is not None and not split_changed else None
    epoch_plan = None
    epoch_start = 0
    if epochs is not None or saved_plan is not None:
        sampler = BalancedEpochSampler(index, train_indices, seed=seed)
        epoch_steps = sampler.steps_per_epoch
        if sampler.samples // epoch_steps < world_size:
            raise ValueError("The epoch sampler needs at least one state per worker in every batch")
        if saved_plan is not None and saved_plan["steps_per_epoch"] != epoch_steps:
            raise ValueError("Training pool changed the saved steps per epoch")
        epoch_start = global_step if saved_plan is None else saved_plan["start_step"]
        if epochs is None:
            epoch_limit = saved_plan["epochs"]
        else:
            completed_epochs = (global_step - epoch_start) // epoch_steps
            epoch_limit = completed_epochs + epochs
        max_steps = epoch_start + epoch_limit * epoch_steps
        if epoch_limit < 1 or not epoch_start <= global_step <= max_steps:
            raise ValueError("The epoch target must include the checkpoint's saved step")
        epoch_plan = {"start_step": epoch_start, "steps_per_epoch": epoch_steps,
                      "epochs": epoch_limit}
    else:
        sampler = BalancedBatchSampler(index, train_indices, seed=seed)
    micro_batch = int(config.get("micro_batch_size", 2))
    if not 1 <= micro_batch <= 64 or 64 % (micro_batch * world_size):
        raise ValueError("micro_batch_size × world_size must divide effective batch size 64")
    training_model = model if group is None else group.wrap(model)
    log_every = int(config.get("log_every", 25))
    validate_every = int(config.get("validate_every", 500))
    evaluation_batch = int(config.get("evaluation_batch_size", micro_batch))
    metrics_path = output_dir / "metrics.jsonl"
    early_path = output_dir / "early_stopping.json"
    early_state = EarlyStoppingState()
    if payload is not None:
        if scoped_rng:
            restore_training_rng(payload, device, rank)
        saved_state = payload.get("early_stopping")
        saved_policy = saved_state["config"] if saved_state else None
        if saved_policy and dict(saved_policy, enabled=early_config["enabled"]) == early_config:
            early_state = EarlyStoppingState.from_dict(saved_state)
            # A run override changes the stopping policy while keeping the best metric.
            if saved_policy != early_config:
                early_state.reset_patience()
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
        if epoch_plan is not None and (saved_plan is None
                or (early_state.stopped and epochs is not None)):
            early_state.reset_patience()
        if primary:
            _append_jsonl(metrics_path, {
                "phase": phase, "global_step": global_step, "event": "resume",
                "world_size": world_size,
                "previous_world_size": len(payload.get("rng_by_rank", [None])),
                "epoch_plan": epoch_plan,
                "split_rebalanced": split_changed,
            })
        del payload
    if primary:
        save_config(output_dir / "config.yaml", dict(
            config, max_steps=max_steps, epoch_plan=epoch_plan, early_stopping=early_config))
        metadata_path = output_dir / "run_metadata.json"
        metadata = dict(run_metadata or {})
        metadata.update({
            "status": "RUNNING", "phase": phase, "seed": seed,
            "device": str(device), "world_size": world_size,
            "precision": precision, "loss": loss_config, "early_stopping": early_config,
            "global_step": global_step, "max_steps": max_steps, "epoch_plan": epoch_plan,
        })
        write_json(metadata_path, metadata)
        write_json(early_path, early_state.to_dict(early_config))
        write_training_curves(metrics_path, output_dir, phase=phase)
        if epoch_plan is not None:
            print(f"Epoch schedule: {epoch_steps:,} steps per epoch, "
                  f"resuming at step {global_step:,}, target step {max_steps:,}.", flush=True)

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
                "world_size": world_size,
                "micro_batch_size": micro_batch,
                "micro_batches": 64 // micro_batch,
                "micro_batches_per_rank": 64 // (micro_batch * world_size),
                "normalization": "global effective batch with DDP gradient scaling",
            },
            "early_stopping": early_config,
        }
        write_json(loss_statistics_path, loss_statistics)

    started = time.monotonic()
    gradient_check = None
    last_validation = None
    while global_step < max_steps and not early_state.stopped:
        effective = sampler.batch(global_step - epoch_start)
        if group is not None:
            effective = group.split_batch(effective)
        categories = [index.records[item].category for item in effective]
        normalization = masked_batch_normalization(
            effective, categories, change_masks,
            sum_counts=None if group is None else group.sum_counts)
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
        training_model.train()
        for chunk_index, selected in enumerate(chunks):
            selected_categories = [index.records[item].category for item in selected]
            target = _tokens(dataset, selected, device)
            sync = (nullcontext() if group is None else group.accumulation_context(
                training_model, synchronize=chunk_index == len(chunks) - 1))
            with sync:
                with _autocast(device, precision):
                    output = training_model(target)
                    losses = masked_vae_loss(
                        output["logits"].float(), target, output["mu"].float(),
                        output["logvar"].float(), selected, selected_categories,
                        change_masks, normalization, loss_config)
                if not all(torch.isfinite(losses[key]).item() for key in scalar_keys):
                    raise FloatingPointError(f"Non-finite masked loss at step {global_step}")
                if (not torch.isfinite(output["mu"]).all()
                        or not torch.isfinite(output["logvar"]).all()):
                    raise FloatingPointError(f"Non-finite posterior at step {global_step}")
                if group is None:
                    losses["loss_total"].backward()
                else:
                    group.backward(losses["loss_total"])
            for key in scalar_keys:
                aggregate[key] += float(losses[key].detach().item())
            for loss_group in ("dynamic_unchanged_by_category", "dynamic_changed_by_category"):
                for category in DYNAMIC_CATEGORIES:
                    aggregate[loss_group][category] += float(
                        losses[loss_group][category].detach().item())
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

        epoch_end = epoch_plan is not None and (global_step - epoch_start) % epoch_steps == 0
        validate_now = global_step % validate_every == 0 or global_step == max_steps or epoch_end
        log_now = global_step == 1 or global_step % log_every == 0 or validate_now
        if log_now and group is not None:
            group.sum_metrics(aggregate)
        if log_now and primary:
            row = {
                "phase": phase, "global_step": global_step,
                "batch_size": normalization["batch_size"],
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
            if epoch_plan is not None:
                row["completed_epochs"] = (global_step - epoch_start) // epoch_steps
            _append_jsonl(metrics_path, row)

        if validate_now and primary:
            evaluation = evaluate_indices(
                model, index, vocabulary, evaluation_indices, device,
                precision=precision, batch_size=evaluation_batch,
                change_masks=change_masks, masked_loss_config=loss_config)
            _append_jsonl(metrics_path, {
                "phase": phase, "global_step": global_step,
                "event": "overfit_evaluation", **evaluation})
            last_validation = evaluate_indices(
                model, index, vocabulary, validation_indices, device,
                precision=precision, batch_size=evaluation_batch,
                change_masks=change_masks, masked_loss_config=loss_config)
            _append_jsonl(metrics_path, {
                "phase": phase, "global_step": global_step,
                "event": "validation", **last_validation})
            score = last_validation["loss_components"]["loss_total"]
            new_best, _ = early_state.observe(
                global_step, score, early_config, can_stop=epoch_plan is None or epoch_end)
        if validate_now:
            rng_by_rank = None
            if group is not None:
                rng_by_rank = group.gather_rng()
            elif scoped_rng:
                rng_by_rank = [rng_state(device)]
            if primary:
                snapshot = early_state.to_dict(early_config)
                if new_best:
                    save_checkpoint(
                        output_dir / "checkpoint_best.pt", model, optimizer,
                        global_step, config, vocabulary, index, early_state.best_metric,
                        phase, early_stopping=snapshot, rng_by_rank=rng_by_rank,
                        epoch_plan=epoch_plan)
                save_checkpoint(
                    output_dir / "checkpoint_last.pt", model, optimizer,
                    global_step, config, vocabulary, index, early_state.best_metric,
                    phase, early_stopping=snapshot, rng_by_rank=rng_by_rank,
                    epoch_plan=epoch_plan)
                write_json(early_path, snapshot)
                if epoch_end:
                    completed = (global_step - epoch_start) // epoch_steps
                    print(f"Epoch {completed}/{epoch_limit} complete at step {global_step:,}. "
                          f"Validation loss: {score:.6f}", flush=True)
            if group is not None:
                early_state.stopped = group.broadcast_stop(early_state.stopped)
        if log_now and primary:
            write_training_curves(metrics_path, output_dir, phase=phase)

    if not primary:
        return None
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
        "phase": phase, "global_step": global_step,
        "training_seconds": time.monotonic() - started,
        "device": str(device), "world_size": world_size, "precision": precision,
        "parameter_count": parameter_count(model), "loss": loss_config,
        "initial_embedding": str(initial_embedding_path),
        "embedding_drift": embedding_drift(initial_embedding_path, model),
        "stopped_early": early_state.stopped,
        "early_stopping": early_state.to_dict(early_config),
        "epoch_plan": epoch_plan,
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
