"""Export a small fixed validation gallery and serve the read-only 3D viewer."""
from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

from .vocabulary import CanonicalVocabulary


ROOT = Path(__file__).resolve().parents[1]
VIEWER = Path(__file__).resolve().parent / "viewer"
VENDOR = ROOT / "docs/vendor"
DEFAULT_OUTPUT = ROOT / "outputs/voxel_vae/e00"
CATEGORY_ORDER = ("Static", "D1", "D2", "D3", "D4")


def select_validation_gallery(index, eligible, static_count=8, chains_per_dynamic=2):
    """Choose deterministic validation examples without ever touching test."""
    eligible = [int(item) for item in eligible]
    if any(index.records[item].split != "validation" for item in eligible):
        raise ValueError("Reconstruction gallery accepts validation samples only")
    static = [item for item in eligible if index.records[item].category == "Static"]
    if len(static) < static_count:
        raise ValueError(f"Need {static_count} validation Static samples")
    selected = static[:int(static_count)]
    for category in CATEGORY_ORDER[1:]:
        groups = defaultdict(list)
        for item in eligible:
            row = index.records[item]
            if row.category == category and row.chain_id is not None:
                groups[(row.root_index, row.shard, row.chain_id)].append(item)
        complete = []
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda item: index.records[item].chain_index)
            if len(members) == 5 and [index.records[item].chain_index
                                      for item in members] == list(range(5)):
                complete.append(members)
        if len(complete) < chains_per_dynamic:
            raise ValueError(f"Need {chains_per_dynamic} complete validation {category} chains")
        for members in complete[:int(chains_per_dynamic)]:
            selected.extend(members)
    return selected


def export_gallery(args):
    import torch

    from .checkpoint import load_checkpoint
    from .config import load_config
    from .e00 import _load_artifacts, _model, _subset_indices
    from .evaluate import evaluate_indices, write_json
    from .train import resolve_device, set_seed

    output_root = Path(args.output_root).expanduser().resolve()
    vocabulary, index, subsets = _load_artifacts(output_root)
    eligible = _subset_indices(subsets, "e00_b_validation")
    selected = select_validation_gallery(
        index, eligible, args.static_count, args.chains_per_dynamic)
    config = load_config(output_root / "config.yaml")
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device)
    model = _model(config, vocabulary).to(device)
    checkpoint = (Path(args.checkpoint).expanduser().resolve()
                  if args.checkpoint else output_root / "checkpoint_best.pt")
    load_checkpoint(checkpoint, model, None, vocabulary, index,
                    restore_rng=False, map_location=device)
    examples = (Path(args.examples).expanduser().resolve()
                if args.examples else output_root / "viewer_validation_examples.npz")
    precision = args.precision or config["training"].get("precision", "fp32")
    metrics = evaluate_indices(
        model, index, vocabulary, selected, device, precision=precision,
        batch_size=args.batch_size, save_examples=examples)
    metrics_path = examples.with_name(examples.stem + "_metrics.json")
    write_json(metrics_path, {
        "purpose": "read-only reconstruction viewer gallery",
        "checkpoint": str(checkpoint),
        "subset": "e00_b_validation",
        "selection": {
            "static": int(args.static_count),
            "complete_chains_per_dynamic_category": int(args.chains_per_dynamic),
            "samples": len(selected),
        },
        "metrics": metrics,
    })
    print(json.dumps({
        "status": "EXPORTED", "examples": str(examples),
        "metrics": str(metrics_path), "samples": len(selected),
        "device": str(device), "precision": precision,
    }, indent=2))
    if device.type == "cuda":
        torch.cuda.empty_cache()


def export_comparison(args):
    """Export identical validation samples with Plain and Masked predictions."""
    import torch

    from .change_masks import ChangeMaskCache
    from .checkpoint import load_checkpoint
    from .config import load_config
    from .e00 import _load_artifacts, _model, _subset_indices
    from .evaluate import evaluate_indices, write_json
    from .losses import validate_masked_loss_config
    from .train import resolve_device, set_seed

    output_root = Path(args.output_root).expanduser().resolve()
    vocabulary, index, subsets = _load_artifacts(output_root)
    eligible = _subset_indices(subsets, "e00_b_validation")
    selected = select_validation_gallery(
        index, eligible, args.static_count, args.chains_per_dynamic)
    config = load_config(output_root / "config.yaml")
    masked_config = load_config(output_root / "phase_masked/config.yaml")
    loss_config = validate_masked_loss_config(masked_config["loss"])
    masks = ChangeMaskCache(index, vocabulary, eligible)
    device = resolve_device(args.device)
    precision = args.precision or config["training"].get("precision", "fp32")

    def reconstruct(checkpoint, archive, expected_phase=None, expected_loss=None):
        set_seed(int(config["training"]["seed"]))
        model = _model(config, vocabulary).to(device)
        payload = load_checkpoint(
            checkpoint, model, None, vocabulary, index, restore_rng=False,
            map_location=device, expected_phase=expected_phase,
            expected_loss=expected_loss)
        metrics = evaluate_indices(
            model, index, vocabulary, selected, device, precision=precision,
            batch_size=args.batch_size, save_examples=archive,
            change_masks=masks, masked_loss_config=loss_config)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return payload, metrics

    destination = (Path(args.examples).expanduser().resolve() if args.examples
                   else output_root / "viewer_comparison_examples.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        plain_path = Path(folder) / "plain.npz"
        masked_path = Path(folder) / "masked.npz"
        plain_payload, plain_metrics = reconstruct(
            output_root / "checkpoint_best.pt", plain_path)
        masked_payload, masked_metrics = reconstruct(
            output_root / "phase_masked/checkpoint_best.pt", masked_path,
            "E00-Masked", loss_config)
        with np.load(plain_path, allow_pickle=False) as source:
            plain = {key: np.asarray(source[key]) for key in source.files}
        with np.load(masked_path, allow_pickle=False) as source:
            masked = {key: np.asarray(source[key]) for key in source.files}
        for key in plain.keys() - {"predicted_tokens"}:
            if key not in masked or not np.array_equal(plain[key], masked[key]):
                raise ValueError(f"Plain/Masked gallery mismatch: {key}")
        payload = {key: value for key, value in plain.items()
                   if key != "predicted_tokens"}
        payload.update(
            predicted_tokens=masked["predicted_tokens"],
            predicted_plain_tokens=plain["predicted_tokens"],
            predicted_masked_tokens=masked["predicted_tokens"],
        )
        np.savez_compressed(destination, **payload)
    metrics_path = destination.with_name(destination.stem + "_metrics.json")
    write_json(metrics_path, {
        "purpose": "paired Plain/Masked validation reconstruction viewer",
        "subset": "e00_b_validation",
        "samples": len(selected),
        "plain": {
            "checkpoint": str(output_root / "checkpoint_best.pt"),
            "global_step": int(plain_payload["global_step"]),
            "metrics": plain_metrics,
        },
        "masked": {
            "checkpoint": str(output_root / "phase_masked/checkpoint_best.pt"),
            "global_step": int(masked_payload["global_step"]),
            "metrics": masked_metrics,
        },
    })
    print(json.dumps({
        "status": "EXPORTED_COMPARISON", "examples": str(destination),
        "metrics": str(metrics_path), "samples": len(selected),
        "device": str(device), "precision": precision,
    }, indent=2))


def _text(array, index, default=""):
    return default if array is None else str(array[index])


class ReconstructionArchive:
    def __init__(self, examples, vocabulary):
        self.path = Path(examples).expanduser().resolve()
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: np.asarray(source[key]) for key in source.files}
        self.vocabulary = CanonicalVocabulary.load(vocabulary)
        required = {"gt_tokens", "predicted_tokens", "category", "chain_id",
                    "chain_index", "snapshot_id", "world_seed"}
        missing = required - self.arrays.keys()
        if missing:
            raise ValueError("Reconstruction archive is missing: " + ", ".join(sorted(missing)))
        self.gt = self.arrays["gt_tokens"]
        self.predictions = {}
        if "predicted_plain_tokens" in self.arrays:
            self.predictions["Plain"] = self.arrays["predicted_plain_tokens"]
        if "predicted_masked_tokens" in self.arrays:
            self.predictions["Masked"] = self.arrays["predicted_masked_tokens"]
        if not self.predictions:
            self.predictions["Reconstruction"] = self.arrays["predicted_tokens"]
        self.predicted = next(reversed(self.predictions.values()))
        if (self.gt.ndim != 4 or tuple(self.gt.shape[1:]) != (16, 16, 16)
                or any(self.gt.shape != value.shape for value in self.predictions.values())):
            raise ValueError("Expected matching reconstruction tensors [N,16,16,16]")
        if (self.gt.dtype.kind not in "iu"
                or any(value.dtype.kind not in "iu" for value in self.predictions.values())):
            raise ValueError("Reconstruction tokens must be integer IDs")
        maximum = max([self.gt.max(initial=0)]
                      + [value.max(initial=0) for value in self.predictions.values()])
        if int(maximum) >= self.vocabulary.K:
            raise ValueError("Reconstruction token is outside the canonical vocabulary")
        self.records = self._records()
        self.groups = self._groups()

    def _records(self):
        optional = {key: self.arrays.get(key) for key in (
            "shard", "dataset_root", "split", "biome_group", "stratum")}
        records = []
        for index in range(len(self.gt)):
            records.append({
                "index": index,
                "category": _text(self.arrays["category"], index),
                "chain_id": _text(self.arrays["chain_id"], index),
                "chain_index": int(self.arrays["chain_index"][index]),
                "snapshot_id": int(self.arrays["snapshot_id"][index]),
                "world_seed": int(self.arrays["world_seed"][index]),
                "shard": None if optional["shard"] is None else int(optional["shard"][index]),
                "dataset_root": _text(optional["dataset_root"], index),
                "split": _text(optional["split"], index, "unknown"),
                "biome_group": _text(optional["biome_group"], index),
                "stratum": _text(optional["stratum"], index),
            })
        return records

    def _groups(self):
        result = {category: [] for category in CATEGORY_ORDER}
        dynamic = defaultdict(list)
        static_number = 0
        for record in self.records:
            if record["category"] == "Static":
                static_number += 1
                detail = " / ".join(value for value in (
                    record["biome_group"], record["stratum"]) if value)
                result["Static"].append({
                    "label": f"Static {static_number:02d}" + (f" · {detail}" if detail else ""),
                    "indices": [record["index"]],
                })
            elif record["category"] in result:
                dynamic[(record["category"], record["chain_id"])].append(record)
        category_counts = defaultdict(int)
        for (category, chain_id), records in sorted(dynamic.items()):
            category_counts[category] += 1
            records.sort(key=lambda row: row["chain_index"])
            result[category].append({
                "label": f"{category} chain {category_counts[category]:02d}",
                "chain_id": chain_id,
                "indices": [row["index"] for row in records],
            })
        return {category: groups for category, groups in result.items() if groups}

    def manifest(self):
        return {
            "source": str(self.path),
            "samples": len(self.records),
            "shape": [16, 16, 16],
            "records": self.records,
            "groups": self.groups,
            "variants": list(self.predictions),
            "vocabulary": self.vocabulary.semantic_payload(),
            "air_ids": list(self.vocabulary.air_ids),
        }

    def sample(self, index):
        index = int(index)
        if not 0 <= index < len(self.records):
            raise IndexError("Sample index outside reconstruction archive")
        gt = self.gt[index].astype(np.uint16, copy=False)
        predictions = {name: value[index].astype(np.uint16, copy=False)
                       for name, value in self.predictions.items()}
        predicted = next(reversed(predictions.values()))
        correct = gt == predicted
        non_air = ~np.isin(gt, np.asarray(self.vocabulary.air_ids, dtype=np.uint16))
        return {
            "record": self.records[index],
            "gt": gt.reshape(-1).tolist(),
            "predicted": predicted.reshape(-1).tolist(),
            "predictions": {name: value.reshape(-1).tolist()
                            for name, value in predictions.items()},
            "changed_mask": (self.arrays["changed_mask"][index].astype(bool).reshape(-1).tolist()
                             if "changed_mask" in self.arrays else [False] * 4096),
            "metrics": {
                "exact_accuracy": float(correct.mean()),
                "non_air_accuracy": None if not non_air.any() else float(correct[non_air].mean()),
                "errors": int((~correct).sum()),
                "non_air_voxels": int(non_air.sum()),
            },
        }


def _send(handler, body, content_type, status=200):
    body = body if isinstance(body, bytes) else body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def serve(args):
    output_root = Path(args.output_root).expanduser().resolve()
    examples = Path(args.examples).expanduser().resolve() if args.examples else None
    if examples is None:
        candidates = (output_root / "viewer_comparison_examples.npz",
                      output_root / "viewer_validation_examples.npz",
                      output_root / "recon_examples.npz",
                      output_root / "phase_a/recon_examples.npz")
        examples = next((path for path in candidates if path.is_file()), None)
    if examples is None or not examples.is_file():
        raise FileNotFoundError("No reconstruction archive found; run export first")
    archive = ReconstructionArchive(examples, output_root / "training_vocabulary.json")
    manifest_bytes = json.dumps(archive.manifest(), ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            try:
                if request.path == "/api/manifest":
                    return _send(self, manifest_bytes, "application/json; charset=utf-8")
                if request.path == "/api/sample":
                    raw = parse_qs(request.query).get("index", [None])[0]
                    if raw is None:
                        raise ValueError("Missing sample index")
                    body = json.dumps(archive.sample(int(raw)), ensure_ascii=False)
                    return _send(self, body, "application/json; charset=utf-8")
                if request.path.startswith("/vendor/"):
                    root, relative = VENDOR, request.path[len("/vendor/"):]
                else:
                    root = VIEWER
                    relative = request.path.lstrip("/") or "index.html"
                path = (root / unquote(relative)).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    return _send(self, "Not found\n", "text/plain", 404)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                return _send(self, path.read_bytes(), content_type)
            except (ValueError, IndexError) as exc:
                return _send(self, json.dumps({"error": str(exc)}),
                             "application/json; charset=utf-8", 400)

        def log_message(self, format_, *values):
            if args.verbose:
                super().log_message(format_, *values)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"VAE reconstruction viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Archive: {examples}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parser(argv=None):
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="reconstruct a small fixed validation gallery")
    export.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--examples", type=Path)
    export.add_argument("--static-count", type=int, default=8)
    export.add_argument("--chains-per-dynamic", type=int, default=2)
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    export.add_argument("--batch-size", type=int, default=2)
    comparison = commands.add_parser(
        "export-comparison", help="export paired Plain/Masked validation reconstructions")
    comparison.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    comparison.add_argument("--examples", type=Path)
    comparison.add_argument("--static-count", type=int, default=8)
    comparison.add_argument("--chains-per-dynamic", type=int, default=2)
    comparison.add_argument("--device", default="cuda:0")
    comparison.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    comparison.add_argument("--batch-size", type=int, default=2)
    viewer = commands.add_parser("serve", help="serve an existing reconstruction archive")
    viewer.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    viewer.add_argument("--examples", type=Path)
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8770)
    viewer.add_argument("--verbose", action="store_true")
    return top.parse_args(argv)


def main(argv=None):
    args = parser(argv)
    if args.command == "export":
        export_gallery(args)
    elif args.command == "export-comparison":
        export_comparison(args)
    else:
        serve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
