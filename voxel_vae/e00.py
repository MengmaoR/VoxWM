"""Prepare and manually run E00-A/B real-data VAE sanity overfits."""
from __future__ import annotations

import argparse
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


def run_phase(args, phase):
    from .train import fit, set_seed

    output = Path(args.output_root).expanduser().resolve()
    config = load_config(output / "config.yaml")
    vocabulary, index, subset = _load_artifacts(output)
    if phase == "a":
        phase_output = output / "phase_a"
        train_indices = _subset_indices(subset, "e00_a_train")
        validation_indices = None
        phase_config = dict(config["training"], **config["e00_a"])
        balanced = False
    else:
        phase_output = output
        phase_a_path = output / "phase_a/final_metrics.json"
        if not phase_a_path.is_file():
            raise RuntimeError("Run E00-A successfully before E00-B")
        phase_a = json.loads(phase_a_path.read_text())
        if (phase_a["final"]["exact_token_accuracy"] < 0.99 or
                (phase_a["final"]["non_air_exact_token_accuracy"] or 0) < 0.99):
            raise RuntimeError("E00-A did not reach 99% overall and non-Air accuracy; do not run E00-B")
        train_indices = _subset_indices(subset, "e00_b_train")
        validation_indices = _subset_indices(subset, "e00_b_validation")
        phase_config = dict(config["training"], **config["e00_b"])
        balanced = True
    if args.device:
        phase_config["device"] = args.device
    # Model parameters, including the token embedding, must be initialized
    # after the configured seed is applied. fit() resets the same seed again
    # for deterministic sampling/training after construction.
    set_seed(int(phase_config["seed"]))
    model = _model(config, vocabulary)
    resume = args.resume
    if resume == "auto":
        candidate = phase_output / "checkpoint_last.pt"
        resume = str(candidate) if candidate.is_file() else None
    result = fit(
        model, index, vocabulary, train_indices, train_indices, phase_output,
        phase_config, f"E00-{phase.upper()}", balanced=balanced, resume=resume,
        validation_indices=validation_indices)
    render_report(output)
    print(json.dumps(result, indent=2))


def render_report(output):
    output = Path(output)
    vocabulary, index, subset = _load_artifacts(output)
    config = load_config(output / "config.yaml")
    phase_a_path = output / "phase_a/final_metrics.json"
    phase_b_path = output / "final_metrics.json"
    phase_a = json.loads(phase_a_path.read_text()) if phase_a_path.is_file() else None
    phase_b = json.loads(phase_b_path.read_text()) if phase_b_path.is_file() else None
    a_pass = bool(phase_a and phase_a["final"]["exact_token_accuracy"] >= 0.99
                  and (phase_a["final"]["non_air_exact_token_accuracy"] or 0) >= 0.99)
    status = "GO" if a_pass and phase_b else "INCONCLUSIVE"
    summary = index.summary()
    try:
        model = _model(config, vocabulary)
        parameters = sum(parameter.numel() for parameter in model.parameters())
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        parameters = None
    lines = [
        f"# {status} — VoxWM E00 Voxel VAE",
        "",
        "E00 is a tiny real-data implementation/overfit sanity check, not a generalization result.",
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
        f"- Parameters: {'pending PyTorch environment' if parameters is None else f'{parameters:,}'}",
        "- Embedding: 32; encoder: 96/192/192; latent: `[16,4,4,4]`.",
        "- Decoder: 192/192/96/32; no skip, attention, Transformer or softmax.",
        "- Loss: equal-voxel CE + beta(1e-6) × raw KL.",
        "",
        "## E00-A",
        "",
    ]
    if phase_a:
        lines += [
            f"- Steps: {phase_a['global_step']}; seconds: {phase_a['training_seconds']:.2f}",
            f"- CE: {phase_a['initial']['cross_entropy']:.6f} → "
            f"{phase_a['final']['cross_entropy']:.6f}",
            f"- Exact accuracy: {phase_a['final']['exact_token_accuracy']:.6f}",
            f"- Non-Air exact accuracy: {phase_a['final']['non_air_exact_token_accuracy']}",
            f"- KL: {phase_a['final']['latent']['kl_raw']:.6f}",
            f"- Overfit criterion (overall/non-air ≥99%): {'PASS' if a_pass else 'FAIL'}",
        ]
    else:
        lines.append("- NOT RUN. Code and fixed 8-sample manifest are prepared.")
    lines += ["", "## E00-B", ""]
    if phase_b:
        lines += [
            f"- Steps: {phase_b['global_step']}; seconds: {phase_b['training_seconds']:.2f}",
            f"- Train-overfit CE: {phase_b['initial']['cross_entropy']:.6f} → "
            f"{phase_b['final']['cross_entropy']:.6f}",
            f"- Category reconstruction: `{phase_b['final']['categories']}`",
            f"- Dynamic diagnostics: `{phase_b['final']['dynamic']}`",
            f"- Dynamic diagnostics by category: "
            f"`{phase_b['final'].get('dynamic_by_category')}`",
            f"- Active latent channels: {phase_b['final']['latent']['active_latent_channels']}/16",
            f"- Embedding drift from step 0: `{phase_b.get('embedding_drift')}`",
            f"- Validation diagnostics: `{phase_b.get('validation')}`",
        ]
    else:
        lines.append("- NOT RUN. Fixed 160-train/160-validation manifests are prepared.")
    lines += [
        "", "## Issues", "",
        "- Verified in code/tests: canonical remap, committed-row index, center crop, balanced sampler, model/loss/checkpoint APIs.",
    ]
    if phase_a and phase_b:
        lines.append("- Real GPU E00-A and E00-B artifacts are present; E00 remains a sanity overfit, not a generalization result.")
    elif phase_a:
        lines.append("- Real GPU E00-A is present; E00-B numerical results remain unverified until its command completes.")
    else:
        lines.append("- E00-A/B numerical claims remain unverified until the corresponding real GPU commands complete.")
    lines += [
        "- E01 is blocked until E00-A overfits and E00-B/checkpoint-resume diagnostics complete.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def parser(argv=None):
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", action="append", required=True)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    for name in ("run-a", "run-b"):
        run_parser = commands.add_parser(name)
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
    elif args.command == "run-a":
        run_phase(args, "a")
    elif args.command == "run-b":
        run_phase(args, "b")
    else:
        render_report(args.output_root)
        print(Path(args.output_root) / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
