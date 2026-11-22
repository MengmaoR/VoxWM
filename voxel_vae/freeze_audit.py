"""Read-only E01-Masked freeze audit for the validation-selected 27k checkpoint."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .change_masks import _chain_key
from .checkpoint import git_commit, load_checkpoint
from .config import load_config
from .dataset import VALID_SPLITS, VoxelSnapshotDataset
from .e00 import _model, _sha256
from .e01 import DEFAULT_OUTPUT, PHASE, TRAIN_DIAGNOSTICS, VALIDATION, load_prepared_e01
from .e01_split import SPLIT_COUNTS, SPLIT_FILE
from .evaluate import write_json
from .freeze_d3 import CONFUSION_FIELDS, TRANSITION_FIELDS
from .freeze_reconstruction import ReconstructionAudit, posterior_mean, reconstruct
from .freeze_report import render_report
from .freeze_statistics import TrainLatents, add_token_counts, embedding_coverage
from .losses import DYNAMIC_CATEGORIES


STEP = 27_000
VOCABULARY_HASH = "25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a"
LATENT_SHAPE = [16, 4, 4, 4]
TEST_FILES = (
    "test_metrics.json", "d3_error_summary.json", "d3_combined_token_confusions.csv",
    "d3_node_confusions.csv", "d3_transition_analysis.csv", "d3_spatial_errors.json",
)
LATENT_FILES = ("latent_statistics.json", "latent_normalization.json")
EMBEDDING_FILES = ("token_frequency.csv", "embedding_audit.csv", "embedding_summary.json")


def ordered_groups(index, indices):
    """Order every complete chain together, preserving the original sample set."""
    static, chains = [], defaultdict(list)
    for i in indices:
        row = index.records[i]
        if row.category == "Static":
            if row.chain_id is not None or row.chain_index is not None:
                raise ValueError("Static sample has a chain position")
            static.append((i,))
        else:
            if row.category not in DYNAMIC_CATEGORIES or row.chain_id is None or row.chain_index is None:
                raise ValueError("Dynamic sample lacks a valid chain position")
            chains[_chain_key(row)].append(i)
    groups = static
    for key, members in chains.items():
        ordered = tuple(sorted(members, key=lambda i: index.records[i].chain_index))
        rows = [index.records[i] for i in ordered]
        if ([r.chain_index for r in rows] != list(range(5))
                or len({(r.world_seed, r.split, r.category) for r in rows}) != 1):
            raise ValueError(f"Expected one complete five-state chain for {key}")
        groups.append(ordered)
    return sorted(groups, key=min), Counter(index.records[members[0]].category for members in chains.values())


def audit_split(index, manifest, expected=SPLIT_COUNTS):
    selected = {name: index.indices(split=name) for name in VALID_SPLITS}
    counts = {name: len(indices) for name, indices in selected.items()}
    if counts != expected or sum(counts.values()) != len(index.records):
        raise ValueError(f"Expected split counts {expected}, found {counts}")
    seeds = {name: {index.records[i].world_seed for i in indices}
             for name, indices in selected.items()}
    intersections = {f"{a}_vs_{b}": len(seeds[a] & seeds[b])
                     for a, b in combinations(VALID_SPLITS, 2)}
    if any(intersections.values()):
        raise ValueError(f"World seed split leakage: {intersections}")
    test_groups = None
    chain_counts = {}
    for name, indices in selected.items():
        groups, chain_counts[name] = ordered_groups(index, indices)
        if name == "test":
            test_groups = groups
    diagnostic_counts = {}
    for name, split in ((TRAIN_DIAGNOSTICS, "train"), (VALIDATION, "validation")):
        rows = manifest["subsets"][name]
        ids = [int(row["sample_index"]) for row in rows]
        if len(set(ids)) != len(ids) or any(
                not 0 <= i < len(index.records) or index.records[i].split != split for i in ids):
            raise ValueError(f"Fixed {name} includes a row outside {split}")
        diagnostic_counts[name] = len(ids)
    audit = {
        "split_counts": counts, "seed_counts": {k: len(v) for k, v in seeds.items()},
        "seed_intersections": intersections, "complete_dynamic_chains": chain_counts,
        "diagnostic_subset_counts": diagnostic_counts, "test_selection": "index.indices(split='test')",
        "sample_source": "committed samples table; D4 anchors stored separately",
        "d4_anchor_samples": 0, "test_evaluation_writes_training_state": False,
    }
    return selected, test_groups, audit


def open_candidate(e01_root, checkpoint):
    (vocabulary, index, manifest), identity, _ = load_prepared_e01(e01_root)
    if SPLIT_FILE not in identity:
        raise ValueError("Freeze audit requires the prepared 160k/20k/20k split")
    if vocabulary.K != 1580 or vocabulary.sha256 != VOCABULARY_HASH:
        raise ValueError("Freeze audit vocabulary does not match the 27k candidate")
    config = load_config(e01_root / "config.yaml")
    model = _model(config, vocabulary)
    payload = load_checkpoint(
        checkpoint, model, None, vocabulary, index, restore_rng=False, expected_phase=PHASE,
        expected_loss=config["loss"], expected_config={"prepared_data": identity})
    if (checkpoint.name != "checkpoint_best.pt" or payload["global_step"] != STEP
            or payload["latent_shape"] != LATENT_SHAPE or model.embedding_dim != 32):
        raise ValueError("Freeze audit requires checkpoint_best.pt at step 27,000")
    if any(_sha256(Path(root) / "dataset_manifest.json") != digest
           for root, digest in zip(index.roots, index.manifest_hashes)):
        raise ValueError("Dataset manifest differs from the frozen checkpoint")
    provenance = {
        "phase": PHASE, "checkpoint": checkpoint.name, "checkpoint_step": STEP,
        "checkpoint_sha256": _sha256(checkpoint), "source_commit": payload["git_commit"],
        "validation_selection_metric": "validation_loss_total",
        "selected_validation_total_loss": payload["best_metric"],
        "architecture": payload["architecture"], "latent_shape": LATENT_SHAPE,
        "vocabulary_K": vocabulary.K, "vocabulary_sha256": vocabulary.sha256,
        "dataset_roots": [Path(root).name for root in index.roots],
        "dataset_manifest_hashes": list(index.manifest_hashes), "prepared_data": identity,
        "audit_source_commit": os.environ.get("VOXWM_AUDIT_COMMIT") or git_commit(),
        "precision": config["training"]["precision"],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    del payload
    model.eval().requires_grad_(False)
    return vocabulary, index, manifest, config, model, provenance


def _group_batches(groups, capacity):
    batch, size = [], 0
    for members in groups:
        if size and size + len(members) > capacity:
            yield batch
            batch, size = [], 0
        batch.append(members)
        size += len(members)
    if batch:
        yield batch


def _write_csv(path, fields, rows):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_streaming(index, vocabulary, groups, model, config, device, batch_size, workers, output):
    batches = list(_group_batches(groups, batch_size))
    sample_indices = [i for groups_in_batch in batches for members in groups_in_batch for i in members]
    batch_rows = []
    start = 0
    for members in batches:
        end = start + sum(map(len, members))
        batch_rows.append(range(start, end))
        start = end
    loader = DataLoader(VoxelSnapshotDataset(index, vocabulary, sample_indices), batch_sampler=batch_rows,
                        num_workers=workers, pin_memory=device.type == "cuda")
    accumulator = ReconstructionAudit(vocabulary, config["loss"])
    counts = np.zeros(vocabulary.K, dtype=np.int64)
    done = 0
    for members, batch in zip(batches, loader, strict=True):
        tokens = batch["tokens"]
        gt = tokens.numpy()
        add_token_counts(counts, gt)
        inference = [reconstruct(model, part, device, config["training"]["precision"])
                     for part in tokens.split(batch_size)]
        predictions = inference[0] if len(inference) == 1 else tuple(
            np.concatenate(block) for block in zip(*inference))
        rows = [[index.records[i] for i in group] for group in members]
        accumulator.observe(rows, gt, *predictions)
        done += len(tokens)
        if done % 4_096 < len(tokens) or done == len(sample_indices):
            print(f"test reconstruction {done:,}/{len(sample_indices):,}", flush=True)

    result = accumulator.result()
    d3 = accumulator.d3
    transitions = d3.transition_rows()
    d3_summary = d3.summary(transitions)
    d3_metrics = result["dynamic_by_category"]["D3"]
    if (d3_summary["total_changed_voxels"] != d3_metrics["changed_voxels"]
            or d3_summary["transition_voxels"] != d3_metrics["gt_changed_voxels"]):
        raise ValueError("D3 semantic and reconstruction denominators disagree")
    _write_csv(output / "d3_combined_token_confusions.csv", CONFUSION_FIELDS, d3.confusion_rows())
    _write_csv(output / "d3_node_confusions.csv", CONFUSION_FIELDS, d3.confusion_rows(node_only=True))
    _write_csv(output / "d3_transition_analysis.csv", TRANSITION_FIELDS, transitions)
    write_json(output / "d3_spatial_errors.json", d3.spatial_summary())
    write_json(output / "d3_error_summary.json", d3_summary)
    write_json(output / "test_metrics.json", {"checkpoint_step": STEP, "source_split": "test", **result})
    return counts


def _scan_split(index, vocabulary, indices, split, device, batch_size, workers, *, model=None, precision):
    loader = DataLoader(VoxelSnapshotDataset(index, vocabulary, indices), batch_size=batch_size,
                        num_workers=workers, pin_memory=model is not None and device.type == "cuda")
    counts = np.zeros(vocabulary.K, dtype=np.int64)
    latents = TrainLatents(LATENT_SHAPE[0]) if model is not None else None
    done = 0
    for batch in loader:
        tokens = batch["tokens"]
        add_token_counts(counts, tokens.numpy())
        if latents is not None:
            mu = posterior_mean(model, tokens, device, precision)
            latents.update(mu, [index.records[i].category for i in batch["sample_index"].tolist()])
        done += len(tokens)
        if done % 16_384 < len(tokens) or done == len(indices):
            target = "latent and tokens" if latents is not None else "tokens"
            print(f"{split} {target} {done:,}/{len(indices):,}", flush=True)
    if done != len(indices):
        raise ValueError(f"Incomplete {split} scan")
    return counts, latents


def _write_embedding(output, e01_root, config, vocabulary, model, counts):
    with np.load(e01_root / "phase_masked/embedding_initial.npz", allow_pickle=False) as saved:
        if (saved["vocabulary_sha256"].item() != vocabulary.sha256
                or not np.array_equal(saved["token_id"], np.arange(vocabulary.K))
                or saved["seed"].item() != int(config["training"]["seed"])):
            raise ValueError("Initial embedding does not match the frozen run")
        initial = saved["weight"]
    final = model.embedding.weight.detach().float().cpu().numpy()
    frequencies, drift, summary = embedding_coverage(vocabulary, counts, initial, final)
    _write_csv(output / "token_frequency.csv", list(frequencies[0]), frequencies)
    _write_csv(output / "embedding_audit.csv", list(drift[0]), drift)
    write_json(output / "embedding_summary.json", {"checkpoint_step": STEP, **summary})


def _complete(output, files):
    return all((output / name).is_file() for name in files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", choices=("all", "test", "d3", "embedding", "latent", "report"),
                        help="test and d3 share the same full held-out pass")
    parser.add_argument("--e01-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, help="defaults to phase_masked/checkpoint_best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers must be non-negative")
    root = args.e01_root.expanduser().resolve()
    output = root / "freeze_audit"
    if args.audit == "report":
        print(render_report(output), flush=True)
        return 0

    checkpoint = (args.checkpoint or root / "phase_masked/checkpoint_best.pt").expanduser().resolve()
    vocabulary, index, manifest, config, model, provenance = open_candidate(root, checkpoint)
    selected, groups, split = audit_split(index, manifest)
    existing = output / "checkpoint_provenance.json"
    if existing.is_file():
        recorded = json.loads(existing.read_text())
        if any(recorded[key] != provenance[key] for key in ("checkpoint_sha256", "prepared_data")):
            raise ValueError("Existing audit results belong to a different frozen checkpoint or split")
    else:
        write_json(existing, provenance)
    write_json(output / "split_audit.json", split)

    want_test = args.audit in ("all", "test", "d3") and not _complete(output, TEST_FILES)
    want_latent = args.audit in ("all", "latent") and not _complete(output, LATENT_FILES)
    want_embedding = args.audit in ("all", "embedding") and not _complete(output, EMBEDDING_FILES)
    device = torch.device(args.device)
    if want_test or want_latent:
        model.to(device)
    counts = {}
    if want_test:
        counts["test"] = evaluate_streaming(index, vocabulary, groups, model, config, device,
                                           args.batch_size, args.workers, output)
    if want_latent:
        counts["train"], latents = _scan_split(
            index, vocabulary, selected["train"], "train", device, args.batch_size, args.workers,
            model=model, precision=provenance["precision"])
        result = latents.summary()
        training = {"checkpoint_step": STEP, "source_split": "train", **result}
        normalization = {
            "checkpoint_step": STEP, "checkpoint": checkpoint.name, "source_split": "train",
            "samples": result["samples"], "latent_shape": LATENT_SHAPE,
            "representation": "posterior_mean_mu", "mean": result["mean"], "std": result["std"],
            "eps": 1e-6, "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        write_json(output / "latent_statistics.json", training)
        write_json(output / "latent_normalization.json", normalization)
    if want_embedding:
        for name, indices in selected.items():
            if name not in counts:
                counts[name], _ = _scan_split(index, vocabulary, indices, name, device,
                                              args.batch_size, args.workers, precision=provenance["precision"])
            if int(counts[name].sum()) != len(indices) * 16 ** 3:
                raise ValueError(f"Incorrect token count for {name}")
        _write_embedding(output, root, config, vocabulary, model, counts)
    if args.audit == "all":
        print(render_report(output), flush=True)
    else:
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
