"""Replace legacy Static rows with height-adaptive Static in a complete 100K dataset.

The default invocation is read-only.  ``--apply`` builds a fully separate
staging dataset, validates every shard, then installs it at the dynamic
source's original path while retaining the original directory as a backup.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc

from vae_p0.codec import Vocabulary
from .centers import height_statistics
from .protocol import (BIOMES, CATEGORIES, CONFIG, STRATA, VERSION,
                       STATIC_HEIGHT_POLICY_VERSION, d4_targets)
from .storage import atomic_json
from .validate import check_shard


ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_COUNTS = {key: value for key, value in CATEGORIES.items() if key != "Static"}


def parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dynamic-root", type=Path,
                   default=ROOT / "outputs/vae_100k_fixed24",
                   help="Dataset that supplies all D1-D4 rows and is replaced after validation")
    p.add_argument("--static-root", type=Path,
                   default=ROOT / "outputs/vae_100k_heightadaptive",
                   help="Dataset that supplies all height-adaptive Static rows")
    p.add_argument("--staging-root", type=Path,
                   help="Separate build directory; defaults beside dynamic-root")
    p.add_argument("--backup-root", type=Path,
                   help="Retained original dynamic dataset; defaults beside dynamic-root")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Zarr snapshots copied per bounded-memory batch")
    p.add_argument("--apply", action="store_true",
                   help="Build, validate, and replace dynamic-root with rollback protection")
    p.add_argument("--resume-staging", action="store_true",
                   help="Reuse validated READY shards in a matching interrupted staging build")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 64:
        p.error("--batch-size must be in 1..64")
    if args.resume_staging and not args.apply:
        p.error("--resume-staging requires --apply")
    return args


def _resolved(path):
    return Path(path).expanduser().resolve()


def _shard_path(root, shard):
    final = root / "shards" / f"shard_{shard:04d}"
    partial = final.parent / ("." + final.name + ".partial")
    if final.is_dir():
        return final
    if partial.is_dir():
        return partial
    raise ValueError(f"Missing shard {shard:04d} in {root}")


def _read_rows(path, where="1"):
    db = sqlite3.connect(f"file:{(path / 'journal.sqlite').resolve()}?mode=ro", uri=True)
    try:
        return [(int(index), category, source_key, json.loads(metadata))
                for index, category, source_key, metadata in db.execute(
                    f"SELECT idx,category,source_key,metadata FROM samples WHERE {where} ORDER BY idx")]
    finally:
        db.close()


def _read_records(path, table, where="1"):
    db = sqlite3.connect(f"file:{(path / 'journal.sqlite').resolve()}?mode=ro", uri=True)
    try:
        column = "data"
        return [json.loads(row[0]) for row in db.execute(
            f"SELECT {column} FROM {table} WHERE {where} ORDER BY id")]
    finally:
        db.close()


def merge_vocabularies(dynamic, static):
    """Keep every dynamic ID stable and append only missing Static pairs."""
    merged = Vocabulary()
    for pair in dynamic.entries:
        merged.add(*pair)
    if merged.entries != dynamic.entries:
        raise ValueError("Dynamic vocabulary contains duplicate/noncanonical entries")
    mapping = np.asarray([merged.add(*pair) for pair in static.entries], dtype=np.uint16)
    return merged, mapping


def reindex_metadata(dynamic_rows, static_rows, merged_vocabulary_size):
    rows = []
    dynamic_index = {}
    for source_index, category, source_key, original in dynamic_rows:
        new_index = len(rows)
        dynamic_index[source_index] = new_index
        metadata = dict(original)
        metadata["snapshot_id"] = new_index
        metadata["merge_source"] = {
            "dataset_role": "dynamic",
            "snapshot_id": source_index,
        }
        rows.append((new_index, category, source_key, metadata))
    for source_index, category, source_key, original in static_rows:
        new_index = len(rows)
        metadata = dict(original)
        metadata["snapshot_id"] = new_index
        metadata["merge_source"] = {
            "dataset_role": "height_adaptive_static",
            "snapshot_id": source_index,
            "vocabulary_size_at_commit": original.get("vocabulary_size_at_commit"),
        }
        metadata["vocabulary_size_at_commit"] = merged_vocabulary_size
        rows.append((new_index, category, source_key, metadata))
    return rows, dynamic_index


def reindex_chains(chains, dynamic_index):
    result = []
    for original in chains:
        record = dict(original)
        old = list(record["snapshot_indices"])
        try:
            record["snapshot_indices"] = [dynamic_index[index] for index in old]
        except KeyError as exc:
            raise ValueError(f"Chain {record.get('id')} references a discarded row {exc.args[0]}") from exc
        result.append(record)
    return result


def _manifest(root):
    path = root / "dataset_manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing dataset manifest: {path}")
    value = json.loads(path.read_text())
    if value.get("version") != VERSION or value.get("source_mode") != "native":
        raise ValueError(f"Wrong dataset version/source mode: {path}")
    return value


def _array_shapes(path):
    snapshots = zarr.open_group(str(path / "snapshots.zarr"), mode="r")
    anchors = zarr.open_group(str(path / "anchors.zarr"), mode="r")
    for name, dtype in (("tokens", np.uint16), ("light", np.uint8)):
        if snapshots[name].dtype != dtype or snapshots[name].shape[1:] != (48, 48, 48):
            raise ValueError(f"Bad snapshot {name} array in {path}")
        if anchors[name].dtype != dtype or anchors[name].shape[1:] != (48, 48, 48):
            raise ValueError(f"Bad anchor {name} array in {path}")
    return snapshots, anchors


def _preflight_shard(dynamic_root, static_root, shard):
    dynamic_path = _shard_path(dynamic_root, shard)
    static_path = _shard_path(static_root, shard)
    dynamic_all = _read_rows(dynamic_path)
    dynamic_rows = [row for row in dynamic_all if row[1] != "Static"]
    static_all = _read_rows(static_path)
    static_rows = [row for row in static_all if row[1] == "Static"]
    dynamic_counts = Counter(row[1] for row in dynamic_rows)
    static_counts = Counter(row[1] for row in static_all)
    if dynamic_counts != Counter(DYNAMIC_COUNTS):
        raise ValueError(f"Shard {shard:04d} dynamic quota mismatch: {dict(dynamic_counts)}")
    if static_counts != Counter({"Static": CATEGORIES["Static"]}):
        raise ValueError(f"Shard {shard:04d} Static source is not exactly 500 Static rows: {dict(static_counts)}")

    chains = _read_records(dynamic_path, "chains")
    if Counter(record["category"] for record in chains) != Counter({key: 25 for key in DYNAMIC_COUNTS}):
        raise ValueError(f"Shard {shard:04d} dynamic chain quota mismatch")
    if _read_records(static_path, "chains"):
        raise ValueError(f"Shard {shard:04d} Static source unexpectedly contains chains")
    subtypes = Counter(record.get("subtype") for record in chains if record["category"] == "D4")
    if subtypes != Counter(d4_targets(shard)):
        raise ValueError(f"Shard {shard:04d} D4 subtype quota mismatch: {dict(subtypes)}")

    static_quota = Counter((row[3].get("biome_group"), row[3].get("stratum"))
                           for row in static_rows)
    expected = Counter({(biome, stratum): quota for biome in BIOMES
                        for stratum, quota in STRATA.items()})
    if static_quota != expected:
        raise ValueError(f"Shard {shard:04d} height-adaptive Static quota mismatch")

    dynamic_arrays, dynamic_anchors = _array_shapes(dynamic_path)
    static_arrays, static_anchors = _array_shapes(static_path)
    if len(dynamic_arrays["tokens"]) != len(dynamic_all) or len(dynamic_anchors["tokens"]) != 25:
        raise ValueError(f"Shard {shard:04d} dynamic SQLite/Zarr length mismatch")
    if len(static_arrays["tokens"]) != 500 or len(static_anchors["tokens"]) != 0:
        raise ValueError(f"Shard {shard:04d} Static SQLite/Zarr length mismatch")

    source_keys = [row[2] for row in dynamic_rows + static_rows]
    frame_ids = [row[3].get("physical_frame_id") for row in dynamic_rows + static_rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError(f"Shard {shard:04d} has a cross-source source_key collision")
    if None in frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"Shard {shard:04d} has a cross-source physical_frame_id collision")
    seeds = Counter(row[3]["world_seed"] for row in dynamic_rows + static_rows)
    if max(seeds.values()) > CONFIG["max_samples_per_seed_per_shard"]:
        raise ValueError(f"Shard {shard:04d} exceeds the combined per-seed cap")
    return {
        "discarded_legacy_static": len(dynamic_all) - len(dynamic_rows),
        "dynamic": len(dynamic_rows),
        "static": len(static_rows),
        "chains": len(chains),
        "max_samples_per_seed": max(seeds.values()),
    }


def inspect(dynamic_root, static_root):
    dynamic_manifest = _manifest(dynamic_root)
    static_manifest = _manifest(static_root)
    if static_manifest.get("static_height_policy_version") != STATIC_HEIGHT_POLICY_VERSION:
        raise ValueError("Static source does not use the current height-adaptive policy")
    dynamic_vocabulary = Vocabulary(dynamic_root / "vocabulary.json")
    static_vocabulary = Vocabulary(static_root / "vocabulary.json")
    merged_vocabulary, mapping = merge_vocabularies(dynamic_vocabulary, static_vocabulary)
    shards = [_preflight_shard(dynamic_root, static_root, shard) for shard in range(100)]
    registry_path = static_root / "static_positions.sqlite"
    if not registry_path.is_file():
        raise ValueError("Static source is missing its global position registry")
    registry = sqlite3.connect(f"file:{registry_path.resolve()}?mode=ro", uri=True)
    try:
        registry_count = registry.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    finally:
        registry.close()
    if registry_count != 50000:
        raise ValueError(f"Static source registry contains {registry_count}, expected 50000")
    return {
        "dynamic_manifest": dynamic_manifest,
        "static_manifest": static_manifest,
        "dynamic_vocabulary": dynamic_vocabulary,
        "static_vocabulary": static_vocabulary,
        "merged_vocabulary": merged_vocabulary,
        "static_token_mapping": mapping,
        "static_registry_count": registry_count,
        "shards": shards,
    }


def _lock_paths(root):
    paths = [root.parent / ("." + root.name + ".collection_supervisor.lock"),
             root / ".dataset.lock", root / ".vocabulary.lock"]
    paths.extend(root / ".locks" / f"shard_{shard:04d}.lock" for shard in range(100))
    return paths


@contextmanager
def source_locks(roots, create_missing=False):
    streams = []
    try:
        for path in sorted({path for root in roots for path in _lock_paths(root)}):
            if not path.exists() and not create_missing:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(path, "a+" if create_missing else "r")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                raise RuntimeError(f"Collection is active or a lock is still held: {path}")
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()


def _copy_selected(source, destination, indices, start, batch_size, mapping=None):
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        try:
            data = source.oindex[np.asarray(selected), :, :, :]
        except (AttributeError, IndexError, TypeError):
            data = np.stack([source[index] for index in selected])
        if mapping is not None:
            data = mapping[data]
        destination[start + offset:start + offset + len(selected)] = data


def _create_arrays(path, count, anchors):
    compressor = Blosc(cname="zstd", clevel=3)
    snapshots = zarr.open_group(str(path / "snapshots.zarr"), mode="w")
    anchor_group = zarr.open_group(str(path / "anchors.zarr"), mode="w")
    for name, dtype in (("tokens", "<u2"), ("light", "u1")):
        snapshots.create_dataset(name, shape=(count, 48, 48, 48),
                                 chunks=(1, 48, 48, 48), dtype=dtype,
                                 compressor=compressor)
        anchor_group.create_dataset(name, shape=(anchors, 48, 48, 48),
                                    chunks=(1, 48, 48, 48), dtype=dtype,
                                    compressor=compressor)
    return snapshots, anchor_group


def _attempts(path, category_test):
    records = _read_records(path, "attempts")
    invalid = [row for row in records if not isinstance(row.get("category"), str)]
    if invalid:
        raise ValueError(f"Attempt without category in {path}")
    return [row for row in records if category_test(row["category"])]


def _write_database(path, rows, chains, attempts):
    db = sqlite3.connect(path / "journal.sqlite")
    try:
        db.executescript("""CREATE TABLE samples(idx INTEGER PRIMARY KEY,category TEXT,source_key TEXT UNIQUE,metadata TEXT);
            CREATE TABLE chains(id TEXT PRIMARY KEY,data TEXT,anchor_idx INTEGER);
            CREATE TABLE attempts(id INTEGER PRIMARY KEY,data TEXT);""")
        db.executemany("INSERT INTO samples VALUES(?,?,?,?)",
                       [(index, category, key, json.dumps(metadata))
                        for index, category, key, metadata in rows])
        db.executemany("INSERT INTO chains VALUES(?,?,?)",
                       [(record["id"], json.dumps(record), record.get("anchor_idx"))
                        for record in chains])
        db.executemany("INSERT INTO attempts(data) VALUES(?)",
                       [(json.dumps(record),) for record in attempts])
        db.commit()
    finally:
        db.close()


def _write_derived(path, shard, rows, chains, attempts):
    parquet_rows = [dict(snapshot_id=metadata["snapshot_id"], category=metadata["category"],
                         world_seed=metadata["world_seed"], split=metadata["split"],
                         biome_group=metadata.get("biome_group"), stratum=metadata.get("stratum"),
                         chain_id=metadata["chain_id"], metadata_json=json.dumps(metadata))
                    for _, _, _, metadata in rows]
    pq.write_table(pa.Table.from_pylist(parquet_rows), path / "metadata.parquet", compression="zstd")
    counts = Counter(category for _, category, _, _ in rows)
    subtype_counts = Counter(record.get("subtype") for record in chains
                             if record["category"] == "D4")
    missing = {
        "categories": {key: CATEGORIES[key] - counts[key] for key in CATEGORIES},
        "static": [],
        "d4": {key: value - subtype_counts[key] for key, value in d4_targets(shard).items()},
    }
    atomic_json(path / "statistics.json", {
        "height_sampling": height_statistics(attempts),
        "counts": dict(counts),
        "samples": len(rows),
        "missing": missing,
        "attempts": len(attempts),
    })
    atomic_json(path / "manifest.json", {
        "version": VERSION,
        "shard": shard,
        "source_mode": "native",
        "samples": len(rows),
        "config": CONFIG,
        "center_slice": [[16, 32]] * 3,
        "region_shape": [48] * 3,
        "d4_chain_targets": d4_targets(shard),
        "merge": {
            "dynamic": "vae_100k_fixed24",
            "static": "vae_100k_heightadaptive",
        },
    })


def _safe_generated_remove(path, staging_root):
    path = path.resolve()
    if (not path.is_relative_to(staging_root.resolve()) or
            not path.name.startswith((".shard_", "shard_"))):
        raise RuntimeError(f"Refusing to remove non-staging path: {path}")
    if path.exists():
        shutil.rmtree(path)


def build_shard(dynamic_root, static_root, staging_root, shard, merged_size,
                static_mapping, batch_size):
    dynamic_path = _shard_path(dynamic_root, shard)
    static_path = _shard_path(static_root, shard)
    output = staging_root / "shards" / f".shard_{shard:04d}.partial"
    output.mkdir(parents=True, exist_ok=False)

    dynamic_rows = _read_rows(dynamic_path, "category!='Static'")
    static_rows = _read_rows(static_path, "category='Static'")
    rows, dynamic_index = reindex_metadata(dynamic_rows, static_rows, merged_size)
    chains = reindex_chains(_read_records(dynamic_path, "chains"), dynamic_index)
    attempts = (_attempts(dynamic_path, lambda category: category != "Static") +
                _attempts(static_path, lambda category: category == "Static"))

    dynamic_arrays, dynamic_anchors = _array_shapes(dynamic_path)
    static_arrays, _ = _array_shapes(static_path)
    snapshots, anchors = _create_arrays(output, len(rows), len(dynamic_anchors["tokens"]))
    dynamic_indices = [row[0] for row in dynamic_rows]
    static_indices = [row[0] for row in static_rows]
    for name in ("tokens", "light"):
        _copy_selected(dynamic_arrays[name], snapshots[name], dynamic_indices, 0, batch_size)
        _copy_selected(static_arrays[name], snapshots[name], static_indices, len(dynamic_indices),
                       batch_size, static_mapping if name == "tokens" else None)
        anchors[name][:] = dynamic_anchors[name][:]

    _write_database(output, rows, chains, attempts)
    _write_derived(output, shard, rows, chains, attempts)
    result = check_shard(staging_root, output, require_complete=True, require_native=True)
    if not result["valid"]:
        raise ValueError(f"Merged shard {shard:04d} failed validation: {result['errors']}")
    final = output.parent / f"shard_{shard:04d}"
    os.rename(output, final)
    atomic_json(final / "READY", {"version": VERSION, "samples": 1000,
                                  "source_mode": "native", "merged": True})
    return result


def _prepare_staging(staging_root, plan, merged_vocabulary, resume):
    record_path = staging_root / "merge_record.json"
    if staging_root.exists():
        if not resume:
            raise FileExistsError(f"Staging exists; inspect it and use --resume-staging: {staging_root}")
        record = json.loads(record_path.read_text())
        if record.get("plan") != plan or record.get("status") not in ("BUILDING", "VALIDATED"):
            raise ValueError("Existing staging merge_record does not match this merge")
        if Vocabulary(staging_root / "vocabulary.json").entries != merged_vocabulary.entries:
            raise ValueError("Existing staging vocabulary differs from the planned merge")
        return
    staging_root.mkdir(parents=True)
    (staging_root / "shards").mkdir()
    (staging_root / ".locks").mkdir()
    (staging_root / ".dataset.lock").touch()
    (staging_root / ".vocabulary.lock").touch()
    for shard in range(100):
        (staging_root / ".locks" / f"shard_{shard:04d}.lock").touch()
    manifest = dict(CONFIG, source_mode="native",
                    vocabulary="append-only global Node Name/raw param2")
    atomic_json(staging_root / "dataset_manifest.json", manifest)
    merged_vocabulary.save(staging_root / "vocabulary.json")
    atomic_json(record_path, {
        "version": 1,
        "status": "BUILDING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
    })


def _build_static_registry(staging_root):
    path = staging_root / "static_positions.sqlite"
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE positions(position_key TEXT PRIMARY KEY,shard INTEGER NOT NULL,snapshot INTEGER NOT NULL)")
        for shard in range(100):
            shard_path = staging_root / "shards" / f"shard_{shard:04d}"
            source = sqlite3.connect(f"file:{(shard_path / 'journal.sqlite').resolve()}?mode=ro", uri=True)
            try:
                rows = source.execute("SELECT idx,metadata FROM samples WHERE category='Static'").fetchall()
            finally:
                source.close()
            db.executemany("INSERT INTO positions VALUES(?,?,?)",
                           [(json.loads(metadata)["static_position_key"], shard, index)
                            for index, metadata in rows])
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    finally:
        db.close()
    if count != 50000:
        raise ValueError(f"Static registry contains {count}, expected 50000")


def _swap(dynamic_root, staging_root, backup_root):
    if backup_root.exists():
        raise FileExistsError(f"Backup path already exists: {backup_root}")
    os.rename(dynamic_root, backup_root)
    try:
        os.rename(staging_root, dynamic_root)
    except BaseException:
        os.rename(backup_root, dynamic_root)
        raise


def _plan(dynamic_root, static_root, staging_root, backup_root, inspected):
    shards = inspected["shards"]
    return {
        "dynamic_root": str(dynamic_root),
        "static_root": str(static_root),
        "staging_root": str(staging_root),
        "backup_root": str(backup_root),
        "discarded_legacy_static": sum(row["discarded_legacy_static"] for row in shards),
        "kept_dynamic": sum(row["dynamic"] for row in shards),
        "imported_height_adaptive_static": sum(row["static"] for row in shards),
        "merged_samples": sum(row["dynamic"] + row["static"] for row in shards),
        "dynamic_vocabulary": len(inspected["dynamic_vocabulary"].entries),
        "static_vocabulary": len(inspected["static_vocabulary"].entries),
        "appended_vocabulary_entries": (len(inspected["merged_vocabulary"].entries) -
                                         len(inspected["dynamic_vocabulary"].entries)),
        "merged_vocabulary": len(inspected["merged_vocabulary"].entries),
        "static_registry_count": inspected["static_registry_count"],
        "max_samples_per_seed_per_shard": max(row["max_samples_per_seed"] for row in shards),
        "operations": [
            "keep D1-D4 snapshots/chains/anchors from dynamic-root",
            "discard every legacy Static snapshot and Static attempt from dynamic-root",
            "append missing Static vocabulary pairs and remap every imported Static token",
            "import all Static snapshots and Static attempts from static-root",
            "rebuild journal.sqlite, metadata.parquet, statistics.json, manifest.json and READY per shard",
            "rebuild root dataset_manifest.json, vocabulary.json and static_positions.sqlite",
            "omit obsolete source runtime logs, cursors and failure markers from the active merged root",
            "move original dynamic-root to backup-root; keep static-root unchanged",
        ],
    }


def run(args):
    dynamic_root = _resolved(args.dynamic_root)
    static_root = _resolved(args.static_root)
    if dynamic_root == static_root:
        raise ValueError("Dynamic and Static roots must differ")
    staging_root = _resolved(args.staging_root or
                             dynamic_root.parent / ("." + dynamic_root.name + ".merge_staging"))
    backup_root = _resolved(args.backup_root or
                            dynamic_root.parent / (dynamic_root.name + "_pre_heightadaptive"))
    if staging_root in (dynamic_root, static_root) or backup_root in (dynamic_root, static_root):
        raise ValueError("Staging/backup roots must be separate from both source roots")
    if not dynamic_root.is_dir() or not static_root.is_dir():
        raise FileNotFoundError("Both source dataset roots must exist")

    with source_locks((dynamic_root, static_root), create_missing=args.apply):
        inspected = inspect(dynamic_root, static_root)
        plan = _plan(dynamic_root, static_root, staging_root, backup_root, inspected)
        if not args.apply:
            return {"status": "DRY_RUN_OK", "plan": plan}
        if backup_root.exists():
            raise FileExistsError(f"Backup path already exists: {backup_root}")
        _prepare_staging(staging_root, plan, inspected["merged_vocabulary"], args.resume_staging)
        results = []
        for shard in range(100):
            final = staging_root / "shards" / f"shard_{shard:04d}"
            partial = final.parent / ("." + final.name + ".partial")
            if final.is_dir():
                result = check_shard(staging_root, final, require_complete=True, require_native=True)
                if result["valid"] and not (final / "READY").is_file():
                    if not args.resume_staging:
                        raise FileExistsError(f"Validated staging shard lacks READY: {final}")
                    atomic_json(final / "READY", {"version": VERSION, "samples": 1000,
                                                  "source_mode": "native", "merged": True})
                elif not result["valid"]:
                    if args.resume_staging and not (final / "READY").is_file():
                        _safe_generated_remove(final, staging_root)
                        result = build_shard(dynamic_root, static_root, staging_root, shard,
                            len(inspected["merged_vocabulary"].entries),
                            inspected["static_token_mapping"], args.batch_size)
                    else:
                        raise ValueError(
                            f"Existing staged shard {shard:04d} is invalid: {result['errors']}")
            else:
                if partial.exists():
                    if not args.resume_staging:
                        raise FileExistsError(f"Incomplete staging shard exists: {partial}")
                    _safe_generated_remove(partial, staging_root)
                result = build_shard(dynamic_root, static_root, staging_root, shard,
                    len(inspected["merged_vocabulary"].entries),
                    inspected["static_token_mapping"], args.batch_size)
            results.append(result)
            print(json.dumps({"shard": shard, "status": "VALID", "samples": result["samples"]}),
                  flush=True)
        _build_static_registry(staging_root)
        record_path = staging_root / "merge_record.json"
        record = json.loads(record_path.read_text())
        record.update(status="VALIDATED", validated_utc=datetime.now(timezone.utc).isoformat(),
                      valid_shards=sum(result["valid"] for result in results),
                      samples=sum(result["samples"] for result in results))
        atomic_json(record_path, record)
        _swap(dynamic_root, staging_root, backup_root)
        active_record = dynamic_root / "merge_record.json"
        record = json.loads(active_record.read_text())
        record.update(status="COMPLETE", completed_utc=datetime.now(timezone.utc).isoformat(),
                      active_root=str(dynamic_root), retained_backup=str(backup_root),
                      retained_static_source=str(static_root))
        atomic_json(active_record, record)
        return {"status": "COMPLETE", "active_root": str(dynamic_root),
                "backup_root": str(backup_root), "static_root": str(static_root),
                "samples": 100000, "valid_shards": 100,
                "merged_vocabulary": len(inspected["merged_vocabulary"].entries)}


def main(argv=None):
    args = parser(argv)
    try:
        result = run(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
