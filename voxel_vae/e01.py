"""Prepare and manually run E01-Masked on both completed VAE clips."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import load_config, save_config
from .dataset import build_dataset_index
from .e00 import (ROOT, _balanced_subset, _fit_prepared_masked, _load_artifacts,
                  _record, _sha256, _subset_indices, _write_json, render_report)
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
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output / "training_vocabulary.json")
    index.save(output / "dataset_index.json")
    save_config(output / "config.yaml", config)
    _write_json(output / "subset_manifest.json", manifest)
    print(json.dumps({
        "status": "PREPARED", "phase": PHASE, "output": str(output),
        "dataset": summary, "training_pool": manifest["training_pool"],
        "diagnostic_subsets": {name: len(rows) for name, rows in manifest["subsets"].items()},
    }, indent=2))


def run(args):
    output = args.output_root.expanduser().resolve()
    vocabulary, index, manifest = _load_artifacts(output)
    if manifest.get("phase") != PHASE:
        raise ValueError("Run voxel_vae.e01 prepare for this E01 output root first")
    config = load_config(output / "config.yaml")
    train_indices = index.indices(split="train")
    diagnostics = _subset_indices(manifest, TRAIN_DIAGNOSTICS)
    validation = _subset_indices(manifest, VALIDATION)
    for split, selected in (("train", diagnostics), ("validation", validation)):
        if len(selected) != len(set(selected)) or not all(
                0 <= item < len(index.records) and index.records[item].split == split
                for item in selected):
            raise ValueError(f"Invalid prepared E01 {split} diagnostic subset")
    pool = training_pool_summary(index, train_indices)
    if pool != manifest["training_pool"]:
        raise ValueError("E01 training pool changed since prepare")
    phase_config = dict(config["training"], loss=config["loss"],
                        early_stopping=config["early_stopping"],
                        prepared_data=prepared_data_identity(output))
    if args.device:
        phase_config["device"] = args.device
    metadata = {"training_pool": pool, "train_diagnostic_states": len(diagnostics),
                "validation_states": len(validation)}
    print(f"E01-Masked has {pool['samples']:,} train states and {len(validation):,} "
          "fixed validation states.", flush=True)
    result = _fit_prepared_masked(
        output, config, phase_config, vocabulary, index, train_indices,
        diagnostics, validation, resume="auto" if args.resume else None,
        phase=PHASE, metadata=metadata, lazy_masks=True)
    render_report(output, expected_phase=PHASE)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="index both 100K clips and fix validation samples")
    setup.add_argument("--dataset-root", action="append", required=True)
    setup.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train = commands.add_parser("run", help="train E01-Masked using the full train split")
    train.add_argument("--device", help="default from the prepared config, cuda:0 by default")
    train.add_argument("--resume", action="store_true", help="resume this E01 run's checkpoint_last.pt")
    report = commands.add_parser("report", help="render the current E01 report")
    for command in (setup, train, report):
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        render_report(args.output_root, expected_phase=PHASE)
        print(args.output_root / "phase_masked/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
