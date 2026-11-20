"""Strict VAE checkpoint provenance and compatibility checks."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, global_step, config, vocabulary,
                    dataset_index, best_metric=None, phase=None, *, early_stopping=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "phase": phase,
        "config": config,
        "architecture": model.architecture_config(),
        "latent_shape": [model.latent_channels, 4, 4, 4],
        "vocabulary_K": vocabulary.K,
        "vocabulary_sha256": vocabulary.sha256,
        "canonical_vocabulary": vocabulary.semantic_payload(),
        "dataset_roots": list(dataset_index.roots),
        "dataset_manifest_hashes": list(dataset_index.manifest_hashes),
        "git_commit": git_commit(),
        "rng": _rng_state(),
    }
    if early_stopping is not None:
        payload["early_stopping"] = early_stopping
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, vocabulary, dataset_index,
                    restore_rng=True, map_location="cpu", expected_phase=None,
                    expected_loss=None, expected_config=None):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    checks = {
        "vocabulary_K": (payload.get("vocabulary_K"), vocabulary.K),
        "vocabulary_sha256": (payload.get("vocabulary_sha256"), vocabulary.sha256),
        "architecture": (payload.get("architecture"), model.architecture_config()),
        "latent_shape": (payload.get("latent_shape"), [model.latent_channels, 4, 4, 4]),
        "dataset_roots": (payload.get("dataset_roots"), list(dataset_index.roots)),
        "dataset_manifest_hashes": (
            payload.get("dataset_manifest_hashes"), list(dataset_index.manifest_hashes)),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if expected_phase is not None and payload.get("phase") != expected_phase:
        mismatches.append("phase")
    if expected_loss is not None and payload.get("config", {}).get("loss") != expected_loss:
        mismatches.append("loss_objective")
    if expected_config is not None:
        saved = payload.get("config", {})
        mismatches.extend(f"config.{key}" for key, value in expected_config.items()
                          if saved.get(key) != value)
    if mismatches:
        raise ValueError("Checkpoint compatibility failure: " + ", ".join(mismatches))
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return payload
