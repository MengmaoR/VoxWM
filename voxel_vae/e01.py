"""Prepare and manually run E01-Masked on both completed VAE clips."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .config import load_config, save_config
from .dataset import build_dataset_index
from .e00 import (ROOT, _balanced_subset, _fit_prepared_masked, _load_artifacts,
                  _record, _sha256, _subset_indices, _write_json, render_report)
from .e01_split import SPLIT_FILE, apply_test_seeds, select_test_seeds
from .vocabulary import build_canonical_vocabulary


PHASE = "E01-Masked"
DEFAULT_CONFIG = ROOT / "voxel_vae/configs/e01.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/voxel_vae/e01"
TRAIN_DIAGNOSTICS = "e01_train_diagnostics"
VALIDATION = "e01_validation"


def training_pool_summary(index, train_indices):
    by_category = Counter(index.records[item].category for item in train_indices)
    by_root = Counter(index.roots[index.records[item].root_index] for item in train_indices)
    return {"split": "train", "samples": len(train_indices),
            "by_category": dict(by_category), "by_root": dict(by_root)}


def build_subset_manifest(index, vocabulary, config):
    seed = int(config["training"]["seed"])
    validation = config["validation"]
    static_count = int(validation["static_states"])
    chains = int(validation["chains_per_dynamic_category"])
    if static_count < 8 or chains < 2:
        raise ValueError("E01 validation needs at least 8 Static states and 2 chains per dynamic category")
    subsets = {
        TRAIN_DIAGNOSTICS: _balanced_subset(index, "train", seed=seed),
        VALIDATION: _balanced_subset(index, "validation", static_count=static_count,
                                     chains_per_category=chains, seed=seed),
    }
    return {
        "version": 1,
        "phase": PHASE,
        "canonical_vocabulary_sha256": vocabulary.sha256,
        "training_pool": training_pool_summary(index, index.indices(split="train")),
        "validation_subset": VALIDATION,
        "subsets": {name: [_record(index, item) for item in items]
                    for name, items in subsets.items()},
    }


def prepared_data_identity(output):
    return {name: _sha256(output / name) for name in ("dataset_index.json", "subset_manifest.json")}


def load_prepared_e01(output, artifacts=None):
    vocabulary, index, manifest = _load_artifacts(output) if artifacts is None else artifacts
    if manifest.get("phase") != PHASE:
        raise ValueError("Run voxel_vae.e01 prepare for this E01 output root first")
    if training_pool_summary(index, index.indices(split="train")) != manifest["training_pool"]:
        raise ValueError("E01 original training pool changed since prepare")
    source = prepared_data_identity(output)
    path = output / SPLIT_FILE
    if not path.exists():
        return (vocabulary, index, manifest), source, None

    plan = json.loads(path.read_text())
    if plan.get("version") != 1 or plan.get("source") != source:
        raise ValueError(f"{path.name} does not match the original prepared E01 files")
    index = apply_test_seeds(index, plan["world_seeds"])
    identity = source | {SPLIT_FILE: _sha256(path)}
    manifest = dict(manifest, training_pool=training_pool_summary(index, index.indices(split="train")))
    return (vocabulary, index, manifest), identity, source


def prepare(args):
    output = args.output_root.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Prepare E01 in a new or empty output directory: {output}")
    if len(args.dataset_root) != 2:
        raise ValueError("E01 expects both completed 100K clips as separate --dataset-root arguments")
    config = load_config(args.config)
    vocabulary = build_canonical_vocabulary(args.dataset_root)
    index = build_dataset_index(vocabulary)
    summary = index.summary()
    if set(summary["roots"].values()) != {100_000}:
        raise ValueError(f"E01 expects 100,000 committed states per clip. Found {summary['roots']}")
    manifest = build_subset_manifest(index, vocabulary, config)
    selected = select_test_seeds(index)
    effective = apply_test_seeds(index, selected)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output / "training_vocabulary.json")
    index.save(output / "dataset_index.json")
    save_config(output / "config.yaml", config)
    _write_json(output / "subset_manifest.json", manifest)
    _write_json(output / SPLIT_FILE, {
        "version": 1, "source": prepared_data_identity(output), "world_seeds": selected,
    })
    print(json.dumps({
        "status": "PREPARED", "phase": PHASE, "output": str(output),
        "dataset": effective.summary(),
        "training_pool": training_pool_summary(effective, effective.indices(split="train")),
        "diagnostic_subsets": {name: len(rows) for name, rows in manifest["subsets"].items()},
    }, indent=2))


def rebalance(args):
    output = args.output_root.expanduser().resolve()
    (_, index, _), identity, source = load_prepared_e01(output)
    if source is None:
        selected = select_test_seeds(index)
        index = apply_test_seeds(index, selected)
        _write_json(output / SPLIT_FILE, {
            "version": 1, "source": identity, "world_seeds": selected,
        })
    print(json.dumps({
        "status": "REBALANCED", "split_file": str(output / SPLIT_FILE),
        "dataset": index.summary(),
    }, indent=2))


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if args.device:
            raise ValueError("With torchrun, use CUDA_VISIBLE_DEVICES and omit --device")
        from .distributed_training import cuda_training_group

        with cuda_training_group() as group:
            _run(args, group=group)
    else:
        _run(args)


def _run(args, *, group=None):
    primary = group is None or group.primary
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be a positive number of full training passes")
    output = args.output_root.expanduser().resolve()
    artifacts, identity, previous_prepared_data = load_prepared_e01(output)
    vocabulary, index, manifest = artifacts
    config = load_config(output / "config.yaml")
    train_indices = index.indices(split="train")
    diagnostics = _subset_indices(manifest, TRAIN_DIAGNOSTICS)
    validation = _subset_indices(manifest, VALIDATION)
    for split, selected in (("train", diagnostics), ("validation", validation)):
        if len(selected) != len(set(selected)) or not all(
                0 <= item < len(index.records) and index.records[item].split == split
                for item in selected):
            raise ValueError(f"Invalid prepared E01 {split} diagnostic subset")
    pool = manifest["training_pool"]
    phase_config = dict(config["training"], loss=config["loss"],
                        early_stopping=config["early_stopping"],
                        prepared_data=identity)
    if args.device:
        phase_config["device"] = args.device
    metadata = {"training_pool": pool, "train_diagnostic_states": len(diagnostics),
                "validation_states": len(validation),
                "split_counts": dict(Counter(row.split for row in index.records)),
                "prepared_data": identity,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if primary:
        workers = group.world_size if group is not None else 1
        print(f"E01-Masked uses {workers} training process(es), {pool['samples']:,} train states "
              f"and {len(validation):,} fixed validation states.", flush=True)
    result = _fit_prepared_masked(
        output, config, phase_config, vocabulary, index, train_indices,
        diagnostics, validation, resume="auto" if args.resume else None,
        phase=PHASE, metadata=metadata, lazy_masks=True, group=group, epochs=args.epochs,
        previous_prepared_data=previous_prepared_data, no_early_stop=args.no_early_stop)
    if primary:
        render_report(output, expected_phase=PHASE, artifacts=artifacts)
        print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="index both 100K clips and fix validation samples")
    setup.add_argument("--dataset-root", action="append", required=True)
    setup.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    migration = commands.add_parser(
        "rebalance", help="activate the 160,000 / 20,000 / 20,000 split using original test seeds")
    train = commands.add_parser("run", help="train E01-Masked using the full train split")
    train.add_argument("--device", help="single-process device, cuda:0 by default")
    train.add_argument("--resume", action="store_true", help="resume this E01 run's checkpoint_last.pt")
    train.add_argument("--epochs", type=int,
                       help="epochs to complete, counted from the last completed epoch on resume")
    train.add_argument("--no-early-stop", action="store_true",
                       help="disable early stopping for this invocation while keeping validation")
    report = commands.add_parser("report", help="render the current E01 report")
    for command in (setup, migration, train, report):
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args(argv)
    if args.command != "run" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Only E01 run uses torchrun.")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "rebalance":
        rebalance(args)
    elif args.command == "run":
        run(args)
    else:
        output = args.output_root.expanduser().resolve()
        artifacts, _, _ = load_prepared_e01(output)
        render_report(output, expected_phase=PHASE, artifacts=artifacts)
        print(output / "phase_masked/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
