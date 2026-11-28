"""Prepare the fixed three-chunk dataset or explicitly launch its Masked VAE."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .curated_data import PreparedChunks, prepare
from .e00 import _balanced_subset, _model, _write_json
from .e01 import DEFAULT_CONFIG, training_pool_summary
from .full48_source import CANONICAL_SHA256
from .vocabulary import sha256_file


PHASE = "E01-Masked-Curated"
INITIAL_STEP = 27_000
INITIAL_K = 1580
INITIAL_SHA256 = "543d728d20f3164f73d4e703ce341fd4ec2ad5ef0ea79c6ec10299168ec765a1"
EXPANDED_PARAMETERS = {"embedding.weight", "decoder_logits.weight", "decoder_logits.bias"}
DEFAULT_EPOCHS = 10


def copy_expanded_state(model, source, old_k):
    """Copy all existing model rows, leaving only the appended rows initialized."""
    target = model.state_dict()
    if source.keys() != target.keys():
        raise ValueError("Warm-start parameter names differ")
    for key, parameter in target.items():
        weight = source[key]
        if key in EXPANDED_PARAMETERS:
            if weight.shape != parameter[:old_k].shape or parameter.shape[0] < old_k:
                raise ValueError(f"Invalid expanded parameter shape: {key}")
            parameter[:old_k].copy_(weight)
        else:
            if weight.shape != parameter.shape:
                raise ValueError(f"Warm-start parameter shape differs: {key}")
            parameter.copy_(weight)


def warm_start(model, vocabulary, checkpoint):
    import torch

    observed = sha256_file(checkpoint)
    if observed != INITIAL_SHA256:
        raise ValueError(f"Expected the published step-{INITIAL_STEP} best checkpoint, got {observed}")
    # E01 checkpoints contain NumPy RNG state. Only deserialize the authenticated artifact.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if ((saved.get("phase"), saved.get("global_step"), saved.get("vocabulary_K"),
         saved.get("vocabulary_sha256")) != ("E01-Masked", INITIAL_STEP, INITIAL_K, CANONICAL_SHA256)
            or saved.get("canonical_vocabulary") != vocabulary.semantic_payload()[:INITIAL_K]
            or saved.get("architecture") != dict(model.architecture_config(), vocabulary_size=INITIAL_K)):
        raise ValueError("Warm-start architecture or semantic prefix differs from E01 best")
    copy_expanded_state(model, saved["model"], INITIAL_K)
    return {"path": str(checkpoint), "sha256": observed, "step": INITIAL_STEP,
            "copied_token_rows": INITIAL_K, "new_token_rows": vocabulary.K - INITIAL_K}


def _run(args, group=None):
    from .train import fit_masked, set_seed

    data = PreparedChunks(args.prepared)
    vocabulary = data.vocabulary
    base = load_config(DEFAULT_CONFIG)
    train_indices = data.indices(split="train")
    settings = base["training"]
    seed = settings["seed"]
    validation = base["validation"]
    train_diagnostics = _balanced_subset(data, "train", seed=seed)
    validation_indices = _balanced_subset(
        data, "validation", static_count=validation["static_states"],
        chains_per_category=validation["chains_per_dynamic_category"], seed=seed)
    device = (args.device or "cuda:0") if group is None else str(group.device)
    output = args.output_dir.resolve()
    if args.resume:
        resume = (output / "checkpoint_last.pt" if args.resume == "auto"
                  else Path(args.resume).resolve())
        if not resume.is_file():
            raise FileNotFoundError(resume)
    else:
        resume = None
    primary = group is None or group.primary
    if primary and not resume:
        output.mkdir(parents=True, exist_ok=False)
    epochs = args.epochs
    if epochs is None and not resume:
        epochs = DEFAULT_EPOCHS
    set_seed(seed)
    model = _model(base, vocabulary)
    prepared_id = {"manifest_sha256": data.manifest_hashes[0],
                   "vocabulary_sha256": vocabulary.sha256}
    config = {
        **settings, "device": device,
        "loss": base["loss"], "prepared_data": prepared_id, "warm_start_sha256": INITIAL_SHA256,
        "early_stopping": {**base["early_stopping"], "patience_validations": 5,
                           "min_steps": (len(train_indices) + 63) // 64},
    }
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if resume and primary else {}
    metadata.update({
        "phase": PHASE, "prepared_dir": str(args.prepared.resolve()),
        "training_pool": training_pool_summary(data, train_indices),
        "validation_states": len(validation_indices),
        "canonical_vocabulary_sha256": vocabulary.sha256,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    })
    try:
        if not resume:
            metadata["warm_start"] = warm_start(model, vocabulary, args.init_from.resolve())
        if primary and not resume:
            vocabulary.save(output / "training_vocabulary.json")
        return fit_masked(
            model, data, vocabulary, train_indices, train_diagnostics, validation_indices,
            data, output, config, resume=resume, run_metadata=metadata, phase=PHASE, group=group,
            epochs=epochs, no_early_stop=args.no_early_stop, dataset=data, allow_empty_changes=True)
    except (Exception, KeyboardInterrupt) as exc:
        if primary:
            if metadata_path.is_file():
                metadata.update(json.loads(metadata_path.read_text()))
            metadata.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                            error=f"{type(exc).__name__}: {exc}")
            _write_json(metadata_path, metadata)
        raise


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if args.device:
            raise ValueError("With torchrun, use CUDA_VISIBLE_DEVICES and omit --device")
        from .distributed_training import cuda_training_group

        with cuda_training_group() as group:
            return _run(args, group)
    return _run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("prepare", help="CPU-only materialization of the fixed index")
    pack.add_argument("--selection", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    train = commands.add_parser("run", help="Start or resume this Masked VAE run")
    train.add_argument("--prepared", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-from", type=Path)
    source.add_argument("--resume", nargs="?", const="auto")
    train.add_argument("--device")
    train.add_argument("--epochs", type=int, help="Target additional whole epochs")
    train.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.selection, args.output)
    else:
        if args.epochs is not None and args.epochs < 1:
            parser.error("--epochs must be positive")
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
