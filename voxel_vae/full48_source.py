"""Read category-repacked VAE archives without extracting or changing their data."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "voxel_vae/reports"
CANONICAL_SHA256 = "25af338d8200eaceedf62bd33dc76d769dcec27f526e6df34280c43c9b3f979a"
SPLITS = ("train", "validation", "test")
SPLIT_COUNTS = dict(train=160_000, validation=20_000, test=20_000)
HEADER_NAMES = {"vocabulary.json", "dataset_manifest.json", "split_160k.json"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_canonical(raw, *, expected_k=1580, expected_sha256=CANONICAL_SHA256):
    """Check exact Node Name/raw-param2 semantics; never normalize param2."""
    tokens = raw.get("tokens")
    _require(isinstance(tokens, list) and raw.get("K") == len(tokens) == expected_k,
             "Canonical vocabulary K differs from the frozen 1580-token vocabulary")
    pairs = []
    for token_id, row in enumerate(tokens):
        _require(type(row.get("id")) is int and row["id"] == token_id,
                 f"Non-contiguous canonical ID at {token_id}")
        name, param2 = row.get("node_name"), row.get("param2")
        _require(isinstance(name, str) and bool(name), f"Invalid Node Name at token {token_id}")
        _require(type(param2) is int and 0 <= param2 <= 255,
                 f"Invalid raw uint8 param2 at token {token_id}")
        pairs.append((name, param2))
    _require(len(set(pairs)) == len(pairs) and pairs == sorted(pairs),
             "Canonical tokens must be a unique sorted semantic list")
    payload = [{"id": i, "node_name": name, "param2": param2}
               for i, (name, param2) in enumerate(pairs)]
    semantic = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(semantic).hexdigest()
    _require(digest == expected_sha256 and raw.get("sha256") == digest,
             "Canonical vocabulary semantic SHA256 mismatch")
    return dict(K=len(pairs), pairs=tuple(pairs), sha256=digest)


def load_reference(training_vocab_path=None, baseline_csv=None, split_path=None):
    """Load published references only; do not recreate the missing original index."""
    training_vocab_path = Path(training_vocab_path or
                              REPORTS / "e01_masked_20261122/training_vocabulary.json")
    baseline_csv = Path(baseline_csv or
                        REPORTS / "e01_freeze_audit_20261122/token_frequency.csv")
    split_path = Path(split_path or REPORTS / "e01_masked_20261122/split_160k.json")
    vocab_bytes, split_bytes = training_vocab_path.read_bytes(), split_path.read_bytes()
    raw = json.loads(vocab_bytes)
    reference = validate_canonical(raw)
    mapping_checks = []
    for root in raw.get("roots", []):
        mapping = root.get("local_to_canonical")
        count = root.get("local_K")
        _require(type(count) is int and isinstance(mapping, list) and len(mapping) == count,
                 "Original training root mapping length mismatch")
        _require(all(type(i) is int and 0 <= i < reference["K"] for i in mapping),
                 "Original training root mapping has an out-of-range ID")
        _require(len(set(mapping)) == len(mapping), "Original training root mapping is not injective")
        identity = mapping == list(range(len(mapping)))
        _require(root.get("identity_mapping") == identity,
                 "Original training root identity_mapping flag is inconsistent")
        mapping_checks.append(dict(
            root=root["root"], local_K=count, mapping_count=len(mapping),
            ids_in_range=True, injective=True, identity_mapping=identity,
            recorded_local_vocabulary_sha256=root.get("local_sha256"),
            recorded_dataset_manifest_sha256=root.get("manifest_sha256"),
            actual_original_local_vocabulary_verified=False,
            semantic_check_boundary="original local vocabulary files not supplied; structural LUT checks only"))
    _require(len(mapping_checks) == 2, "Expected two original training roots")
    baseline = {split: np.zeros(reference["K"], dtype=np.int64) for split in SPLITS}
    seen = set()
    with baseline_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            token_id = int(row["token_id"])
            _require(0 <= token_id < reference["K"] and token_id not in seen,
                     "Duplicate or invalid canonical ID in center frequency reference")
            seen.add(token_id)
            _require((row["node_name"], int(row["param2"])) == reference["pairs"][token_id],
                     f"Center frequency reference token semantics mismatch: {token_id}")
            for split in SPLITS:
                count = int(row[f"{split}_count"])
                _require(count >= 0, "Negative center token frequency")
                baseline[split][token_id] = count
    _require(len(seen) == reference["K"], "Center frequency reference omits tokens")
    for split in SPLITS:
        _require(int(baseline[split].sum()) == SPLIT_COUNTS[split] * 16**3,
                 f"Center reference count denominator mismatch for {split}")
    unseen = baseline["train"] == 0
    heldout_only = unseen & ((baseline["validation"] + baseline["test"]) > 0)
    never = unseen & ~heldout_only
    _require((int((~unseen).sum()), int(unseen.sum()), int(heldout_only.sum()), int(never.sum()))
             == (1306, 274, 27, 247), "Center reference does not reproduce the U0 baseline")
    split = json.loads(split_bytes)
    _require(split.get("version") == 1 and set(split.get("world_seeds", {})) == {"train", "validation"},
             "Invalid existing split_160k plan")
    moves = {}
    for target, seeds in split["world_seeds"].items():
        for seed in seeds:
            _require(type(seed) is int and seed not in moves, "Invalid/repeated split world seed")
            moves[seed] = target
    validation = dict(
        canonical_K=reference["K"], canonical_semantic_sha256=reference["sha256"],
        training_vocabulary_file_sha256=hashlib.sha256(vocab_bytes).hexdigest(),
        token_frequency_file_sha256=hashlib.sha256(baseline_csv.read_bytes()).hexdigest(),
        split_file_sha256=hashlib.sha256(split_bytes).hexdigest(),
        original_dataset_index_expected_sha256=split.get("source", {}).get("dataset_index.json"),
        center_train_unique=1306, U0=274, heldout_only=27, never_seen_in_center_splits=247,
        original_training_roots_mapping_checks=mapping_checks)
    return dict(**reference, baseline=baseline, unseen=unseen, split_moves=moves,
                split_plan=split, split_bytes=split_bytes, validation=validation)


def validate_source_header(header, reference):
    """Validate canonical repack identity and its unchanged saved split patch."""
    docs = header["json"]
    _require(HEADER_NAMES <= docs.keys(), "Archive must place vocabulary, split and manifest before shard data")
    source = validate_canonical(docs["vocabulary.json"], expected_k=reference["K"],
                                expected_sha256=reference["sha256"])
    _require(source["pairs"] == reference["pairs"], "Repacked canonical token order/semantics differs")
    _require(docs["split_160k.json"] == reference["split_plan"]
             and header["raw"]["split_160k.json"] == reference["split_bytes"],
             "Archive split_160k differs from the existing published split patch")
    manifest = docs["dataset_manifest.json"]
    repack = manifest.get("repack", {})
    _require(manifest.get("region_size") == 48 and manifest.get("chunk_size") == 16,
             "Repack region/chunk dimensions differ from 48/16")
    _require(repack.get("canonical_vocabulary_K") == reference["K"]
             and repack.get("canonical_vocabulary_sha256") == reference["sha256"],
             "Repack manifest canonical identity mismatch")
    _require(repack.get("capture_split_preserved") is True,
             "Repack must preserve capture splits for applying the existing E01 seed patch")
    category = repack.get("category")
    _require(category in ("Static", "Dynamic"), "Expected a Static or Dynamic category repack")
    lut = np.arange(reference["K"], dtype=np.uint16)
    validation = dict(
        archive=header["archive"], root_name=header["root_name"], category=category,
        canonical_K=source["K"], canonical_semantic_sha256=source["sha256"],
        source_vocabulary_file_sha256=hashlib.sha256(header["raw"]["vocabulary.json"]).hexdigest(),
        repack_local_to_canonical="identity; every exact semantic pair and raw param2 checked",
        ids_in_range=True, injective=True, param2_preserved=True,
        split_patch_bytes_match_published=True,
        original_training_roots_mapping_checks=reference["validation"]["original_training_roots_mapping_checks"],
        original_dataset_index=header["index_search"])
    return dict(lut=lut, validation=validation, category=category, manifest=manifest)


@dataclass(frozen=True)
class ArchiveShard:
    header: dict
    root_name: str
    shard_name: str
    files: dict[str, bytes]


def _selected_shard_file(relative):
    if relative in ("journal.sqlite", "READY", "manifest.json"):
        return True
    for prefix in ("snapshots.zarr/tokens/", "anchors.zarr/tokens/"):
        if relative.startswith(prefix):
            leaf = relative[len(prefix):]
            return bool(leaf) and (not leaf.startswith(".") or leaf in (".zarray", ".zattrs"))
    return False


def iter_archive_shards(path, *, select_file=_selected_shard_file):
    """Yield one shard's compressed Zarr chunks and SQLite image in bounded memory.

    Only regular selected files are cached. Directory headers do not start a
    shard: these archives list all 200 sibling directories before their contents.
    Revisited shard contents are rejected, not silently combined or omitted.
    The shared mutable header records any dataset_index.json found anywhere.
    """
    path = Path(path).expanduser().resolve()
    header = dict(archive=str(path), archive_bytes=path.stat().st_size, root_name=None,
                  json={}, raw={}, shard_names=[], shards_yielded=0,
                  index_search=dict(status="not supplied", locations=[],
                                    original_index_identity_verified=False))
    process = subprocess.Popen(["zstd", "-dc", "--", str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    current, files, finished = None, {}, set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                # Python 3.11 otherwise retains every TarInfo even in stream mode.
                archive.members.clear()
                parts = PurePosixPath(member.name).parts
                _require(parts and not member.name.startswith("/") and ".." not in parts,
                         f"Invalid archive member path: {member.name}")
                root = parts[0]
                if header["root_name"] is None:
                    header["root_name"] = root
                _require(root == header["root_name"], "Archive contains multiple dataset roots")
                relative = "/".join(parts[1:])
                if parts[-1] == "dataset_index.json" and member.isfile():
                    header["index_search"]["status"] = "found; immutable original index identity not verified"
                    header["index_search"]["locations"].append(dict(path=member.name, bytes=member.size))
                if member.isfile() and relative in HEADER_NAMES:
                    _require(relative not in header["raw"], f"Duplicate root header: {relative}")
                    raw = archive.extractfile(member).read()
                    header["raw"][relative], header["json"][relative] = raw, json.loads(raw)
                    continue
                if len(parts) < 4 or parts[1] != "shards":
                    continue
                shard, filename = parts[2], "/".join(parts[3:])
                if shard not in header["shard_names"]:
                    header["shard_names"].append(shard)
                if member.isdir():
                    continue
                if not select_file(filename):
                    continue
                _require(member.isfile(), f"Selected archive member is not a regular file: {member.name}")
                _require(HEADER_NAMES <= header["json"].keys(), "Required root metadata occurs after shard data")
                if shard != current:
                    _require(shard not in finished,
                             f"Interleaved archive revisits {shard}; single-shard streaming cannot complete it")
                    if current is not None:
                        finished.add(current)
                        header["shards_yielded"] += 1
                        yield ArchiveShard(header, root, current, files)
                    current, files = shard, {}
                _require(filename not in files, f"Duplicate selected shard file: {member.name}")
                files[filename] = archive.extractfile(member).read()
        # Drain tar padding so zstd verifies the complete compressed frame/checksum.
        for _ in iter(lambda: process.stdout.read(1024 * 1024), b""):
            pass
        error = process.stderr.read().decode(errors="replace")
        _require(process.wait() == 0, f"zstd could not completely read {path.name}: {error[-2000:]}")
        _require(current is not None, f"No readable shard data in {path}")
        header["shards_yielded"] += 1
        yield ArchiveShard(header, header["root_name"], current, files)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stderr.close()
