"""Prepare and manually run the active E00-Masked VAE experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from .config import load_config, save_config
from .dataset import DatasetIndex, build_dataset_index
from .vocabulary import CanonicalVocabulary, build_canonical_vocabulary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "voxel_vae/configs/e00.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/voxel_vae/e00"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _record(index, sample_index):
    row = index.records[int(sample_index)]
    return {
        "sample_index": int(sample_index), "key": row.key,
        "dataset_root": index.roots[row.root_index], "shard": row.shard,
        "snapshot_index": row.snapshot_index, "category": row.category,
        "chain_id": row.chain_id, "chain_index": row.chain_index,
        "world_seed": row.world_seed, "split": row.split,
        "biome_group": row.biome_group, "stratum": row.stratum,
        "d4_subtype": row.d4_subtype,
    }


def _tiny_a(index):
    quotas = {"Static": 2, "D1": 2, "D2": 2, "D3": 1, "D4": 1}
    selected = []
    for category, count in quotas.items():
        options = index.indices(split="train", categories=[category])
        if len(options) < count:
            raise ValueError(f"Insufficient train {category} for E00-A")
        by_root = defaultdict(deque)
        for sample_index in options:
            by_root[index.records[sample_index].root_index].append(sample_index)
        while count:
            progressed = False
            for root_index in sorted(by_root):
                if by_root[root_index] and count:
                    selected.append(by_root[root_index].popleft())
                    count -= 1
                    progressed = True
            if not progressed:
                raise ValueError(f"Insufficient train {category} for E00-A")
    return selected


def _diverse_static(index, split, count=80):
    groups = defaultdict(deque)
    for sample_index in index.indices(split=split, categories=["Static"]):
        row = index.records[sample_index]
        groups[(row.biome_group or "", row.stratum or "", row.root_index)].append(sample_index)
    selected = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} diverse {split} Static samples, found {len(selected)}")
    return selected


def _complete_chains(index, split, category, count=4):
    groups = defaultdict(list)
    for sample_index in index.indices(split=split, categories=[category]):
        row = index.records[sample_index]
        if row.chain_id is not None:
            groups[(row.root_index, row.shard, row.chain_id)].append(sample_index)
    valid_by_root = defaultdict(deque)
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: index.records[item].chain_index)
        if len(members) == 5 and [index.records[item].chain_index
                                  for item in members] == list(range(5)):
            valid_by_root[key[0]].append(members)
    chosen = []
    while len(chosen) < count * 5:
        progressed = False
        for root_index in sorted(valid_by_root):
            if valid_by_root[root_index] and len(chosen) < count * 5:
                chosen.extend(valid_by_root[root_index].popleft())
                progressed = True
        if not progressed:
            break
    if len(chosen) == count * 5:
        return chosen
    raise ValueError(f"Need {count} complete {split} {category} chains")


def _balanced_subset(index, split):
    selected = _diverse_static(index, split, 80)
    for category in ("D1", "D2", "D3", "D4"):
        selected.extend(_complete_chains(index, split, category, 4))
    if len(selected) != 160:
        raise AssertionError("E00-B subset must contain 160 states")
    return selected


def build_subset_manifest(index, vocabulary):
    subsets = {
        "e00_a_train": _tiny_a(index),
        "e00_b_train": _balanced_subset(index, "train"),
        "e00_b_validation": _balanced_subset(index, "validation"),
    }
    return {
        "version": 1,
        "canonical_vocabulary_sha256": vocabulary.sha256,
        "selection": {
            "e00_a_train": "2 Static, 2 D1, 2 D2, 1 D3, 1 D4; deterministic index order",
            "e00_b_train": "80 diverse Static + four complete five-state clips per D1-D4",
            "e00_b_validation": "same fixed 160-state composition from validation split",
        },
        "subsets": {name: [_record(index, item) for item in items]
                    for name, items in subsets.items()},
    }


def _subset_indices(manifest, name):
    return [int(row["sample_index"]) for row in manifest["subsets"][name]]


def _load_artifacts(output):
    output = Path(output)
    vocabulary = CanonicalVocabulary.load(output / "training_vocabulary.json")
    index = DatasetIndex.load(output / "dataset_index.json")
    if index.vocabulary_hash != vocabulary.sha256:
        raise ValueError("Prepared index/vocabulary mismatch")
    manifest = json.loads((output / "subset_manifest.json").read_text())
    if manifest["canonical_vocabulary_sha256"] != vocabulary.sha256:
        raise ValueError("Subset/canonical vocabulary mismatch")
    return vocabulary, index, manifest


def _model(config, vocabulary):
    from .model import VoxelVAE

    model = config["model"]
    return VoxelVAE(
        vocabulary.K,
        embedding_dim=int(model["embedding_dim"]),
        latent_channels=int(model["latent_channels"]),
        base_channels=int(model["base_channels"]),
        hidden_channels=int(model["hidden_channels"]),
        logvar_min=float(model["logvar_clamp"][0]),
        logvar_max=float(model["logvar_clamp"][1]),
    )


def prepare(args):
    output = Path(args.output_root).expanduser().resolve()
    config = load_config(args.config)
    vocabulary = build_canonical_vocabulary(args.dataset_root)
    index = build_dataset_index(vocabulary)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output / "training_vocabulary.json")
    index.save(output / "dataset_index.json")
    subset = build_subset_manifest(index, vocabulary)
    _write_json(output / "subset_manifest.json", subset)
    resolved = dict(config)
    resolved["dataset_roots"] = list(index.roots)
    resolved["canonical_vocabulary_sha256"] = vocabulary.sha256
    save_config(output / "config.yaml", resolved)
    render_report(output)
    print(json.dumps({
        "status": "PREPARED", "output": str(output),
        "canonical_K": vocabulary.K, "vocabulary_sha256": vocabulary.sha256,
        "dataset": index.summary(),
        "subsets": {name: len(rows) for name, rows in subset["subsets"].items()},
    }, indent=2))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_masked(args):
    import torch

    from .change_masks import ChangeMaskCache
    from .train import fit_masked, set_seed

    output = Path(args.output_root).expanduser().resolve()
    required = ("subset_manifest.json", "config.yaml", "training_vocabulary.json",
                "dataset_index.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared Masked training artifacts are missing: " + ", ".join(missing))
    vocabulary, index, subset = _load_artifacts(output)
    prepared_config = load_config(output / "config.yaml")
    source_config = load_config(DEFAULT_CONFIG)
    if "loss" not in source_config:
        raise ValueError("Source e00.yaml lacks the E00-Masked loss block")
    train_indices = _subset_indices(subset, "e00_b_train")
    validation_indices = _subset_indices(subset, "e00_b_validation")
    phase_config = dict(prepared_config["training"], **prepared_config["e00_b"])
    phase_config["loss"] = dict(source_config["loss"])
    phase_config["early_stopping"] = dict(prepared_config.get(
        "early_stopping", source_config["early_stopping"]))
    if args.device:
        phase_config["device"] = args.device
    if abs(float(phase_config["beta"]) - float(phase_config["loss"]["beta_kl"])) > 1e-15:
        raise ValueError("Masked KL beta differs from the prepared E00 beta")

    phase_output = output / "phase_masked"
    resume = args.resume
    if resume == "auto":
        resume = phase_output / "checkpoint_last.pt"
    elif resume:
        resume = Path(resume).expanduser().resolve()
    if resume and not resume.is_file():
        raise FileNotFoundError(f"Masked resume checkpoint not found: {resume}")
    protected = ("metrics.jsonl", "final_metrics.json", "checkpoint_best.pt",
                 "checkpoint_last.pt", "recon_examples.npz")
    if not resume and any((phase_output / name).exists() for name in protected):
        raise FileExistsError(
            "phase_masked already contains run artifacts; use --resume or a clean output")

    set_seed(int(phase_config["seed"]))
    model = _model(prepared_config, vocabulary)
    initial_state_sha256 = _model_state_sha256(model)
    phase_output.mkdir(parents=True, exist_ok=True)
    save_config(phase_output / "config.yaml", phase_config)
    masks = ChangeMaskCache(
        index, vocabulary, list(dict.fromkeys(train_indices + validation_indices)))
    metadata = {
        "masked_directory": str(phase_output),
        "prepared_config_sha256": _sha256(output / "config.yaml"),
        "subset_manifest_sha256": _sha256(output / "subset_manifest.json"),
        "canonical_vocabulary_sha256": vocabulary.sha256,
        "masked_initial_state_sha256": initial_state_sha256,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_mask_statistics": masks.statistics(train_indices),
        "validation_mask_statistics": masks.statistics(validation_indices),
    }
    try:
        result = fit_masked(
            model, index, vocabulary, train_indices, train_indices,
            validation_indices, masks, phase_output, phase_config,
            resume=resume, run_metadata=metadata)
    except (Exception, KeyboardInterrupt) as exc:
        metadata_path = phase_output / "run_metadata.json"
        failed = (json.loads(metadata_path.read_text())
                  if metadata_path.is_file() else metadata)
        status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        failed.update(status=status, error=f"{type(exc).__name__}: {exc}")
        _write_json(metadata_path, failed)
        raise
    render_report(output)
    print(json.dumps(result, indent=2))


def render_report(output):
    output = Path(output)
    vocabulary, index, _ = _load_artifacts(output)
    masked_dir = output / "phase_masked"
    masked_path = masked_dir / "final_metrics.json"
    masked = json.loads(masked_path.read_text()) if masked_path.is_file() else None
    summary = index.summary()
    lines = [
        "# VoxWM E00-Masked Voxel VAE",
        "",
        "Masked is the only active VAE training path. Earlier Plain artifacts are retained "
        "as read-only history and are not an active comparison target.",
        "",
        "## Dataset",
        "",
        f"- Roots: {', '.join(index.roots)}",
        f"- Samples: {summary['samples']:,}; splits: `{summary['splits']}`",
        f"- Categories: `{summary['categories']}`",
        f"- Canonical vocabulary K: {vocabulary.K}; SHA256: `{vocabulary.sha256}`",
        f"- Per-root semantic remap: `{[not root.identity_mapping for root in vocabulary.roots]}`",
        "- Input: center `[16:32]^3` Combined Tokens only; no light/metadata/time/category input.",
        "",
        "## Model",
        "",
        "- Embedding: 32; encoder: 96/192/192; latent: `[16,4,4,4]`.",
        "- Decoder: 192/192/96/32; no skip, attention, Transformer or softmax.",
        "- Objective: 0.5 Static CE + 0.25 Dynamic-Unchanged CE + "
        "0.25 Dynamic-Changed CE + 1e-6 raw KL.",
        "",
        "## Current Masked run",
        "",
    ]
    if masked:
        lines += [
            f"- Steps: {masked['global_step']}; seconds: {masked['training_seconds']:.2f}",
            f"- Parameters: {masked['parameter_count']:,}",
            f"- Early stopped: `{masked.get('stopped_early', False)}`; "
            f"state: `{masked.get('early_stopping')}`",
            f"- Final train loss components: `{masked['final']['loss_components']}`",
            f"- Final validation loss components: "
            f"`{masked['validation']['loss_components']}`",
            f"- Final validation reconstruction: "
            f"`{masked['validation']['dynamic_reconstruction']}`",
            f"- Final validation static preservation: "
            f"`{masked['validation']['static_preservation']}`",
        ]
    else:
        lines.append("- No completed Masked run is present.")
    lines += [
        "", "## Training diagnostics", "",
        "- `loss_curves.svg` and `loss_curves.json` are refreshed during training.",
        "- Early stopping monitors validation total loss; the best checkpoint is retained separately.",
        "- The web monitor is read-only and does not start, stop or own the training process.",
        "",
    ]
    masked_dir.mkdir(parents=True, exist_ok=True)
    (masked_dir / "report.md").write_text("\n".join(lines))


def parser(argv=None):
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", action="append", required=True)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run_parser = commands.add_parser("run-masked")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--device", help="auto/cuda/cpu; default from prepared config")
    run_parser.add_argument("--resume", nargs="?", const="auto",
                            help="Resume a compatible checkpoint; omit path for checkpoint_last.pt")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return top.parse_args(argv)


def main(argv=None):
    args = parser(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-masked":
        run_masked(args)
    else:
        render_report(args.output_root)
        print(Path(args.output_root) / "phase_masked/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
