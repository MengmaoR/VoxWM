"""Read one committed shard from archive bytes without extracting or changing it."""
from collections import Counter
from contextlib import closing
from itertools import product
import json
import sqlite3

import numpy as np
import zarr


SPLITS = ("train", "validation", "test")
DYNAMIC = ("D1", "D2", "D3", "D4")
GROUPS = ("Static", "Dynamic", *DYNAMIC)
VOLUME = 48 ** 3
CENTER = (slice(16, 32),) * 3


def _tokens_array(files, pool, expected, label):
    prefix = pool + ".zarr/tokens/"
    store = {key[len(prefix):]: value for key, value in files.items() if key.startswith(prefix)}
    if ".zarray" not in store:
        raise ValueError(f"{label}: missing {pool} token array metadata")
    array = zarr.open_array(store=store, mode="r")
    if array.shape != (expected, 48, 48, 48) or array.dtype != np.dtype("uint16"):
        raise ValueError(f"{label}: invalid {pool} shape/dtype {array.shape}/{array.dtype}")
    separator = json.loads(store[".zarray"]).get("dimension_separator", ".")
    chunk_ranges = [range((size + chunk - 1) // chunk)
                    for size, chunk in zip(array.shape, array.chunks)]
    for coordinates in product(*chunk_ranges):
        key = separator.join(map(str, coordinates))
        if key not in store:
            raise ValueError(f"{label}: missing committed {pool} token chunk {key}")
    return array


def analyze_shard(files, root_name, shard_name, lut, K, split_moves, unseen, canonical_params):
    """Return full48/center counters and per-anchor comparisons for this shard.

    ``files`` contains relative-path byte strings (journal, READY and token Zarr
    members). ``split_moves`` maps whole original-test world seeds to their E01
    destination. ``lut`` maps this root's IDs to canonical IDs. No embeddings,
    RGB, light, native simulation, original files or training state are changed.
    """
    label = f"{root_name}/{shard_name}"

    def require(condition, message):
        if not condition:
            raise ValueError(f"{label}: {message}")

    lut = np.asarray(lut)
    require(lut.ndim == 1 and lut.size > 0 and lut.dtype.kind in "iu", "invalid local-to-canonical table")
    require(np.all(lut < K) and np.all(lut >= 0), "canonical ID outside vocabulary")
    require(len(np.unique(lut)) == len(lut), "local-to-canonical mapping is not one-to-one")
    lut = lut.astype(np.int64, copy=False)
    params = np.asarray(canonical_params)
    require(params.shape == (K,) and params.dtype.kind in "iu"
            and np.all((params >= 0) & (params <= 255)), "canonical param2 is not raw uint8-valued data")
    if isinstance(unseen, (set, frozenset)):
        mask = np.zeros(K, dtype=bool)
        require(all(isinstance(i, (int, np.integer)) and 0 <= i < K for i in unseen), "invalid U0 IDs")
        mask[list(unseen)] = True
    else:
        mask = np.asarray(unseen)
        require(mask.shape == (K,) and mask.dtype == np.bool_, "U0 must be a K-length boolean mask or ID set")
    require("journal.sqlite" in files and "READY" in files, "missing committed journal or READY")
    ready = json.loads(files["READY"])
    with closing(sqlite3.connect(":memory:")) as db:
        db.deserialize(files["journal.sqlite"])
        db.execute("PRAGMA query_only=ON")
        sample_rows = db.execute("SELECT idx,category,source_key,metadata FROM samples ORDER BY idx").fetchall()
        chain_rows = db.execute("SELECT id,data,anchor_idx FROM chains ORDER BY id").fetchall()
    require(ready.get("samples") == len(sample_rows), "READY sample count differs from committed journal")
    records, sample_info = [], []
    for expected, (index, category, source_key, raw) in enumerate(sample_rows):
        meta = json.loads(raw)
        require(index == expected and meta.get("snapshot_id") == index, f"snapshot index mismatch at {index}")
        require(category in ("Static", *DYNAMIC) and meta.get("category") == category,
                f"snapshot {index} category mismatch")
        seed = meta.get("world_seed")
        require(isinstance(seed, int) and not isinstance(seed, bool), f"snapshot {index} has invalid seed")
        original = meta.get("split")
        require(original in SPLITS, f"snapshot {index} has invalid split")
        split = split_moves.get(seed, original)
        require(split in SPLITS, f"snapshot {index} has invalid E01 split")
        if seed in split_moves:
            require(original == "test" and split in ("train", "validation"),
                    f"moved seed {seed} is not from original test")
        records.append(meta)
        sample_info.append(dict(key=f"{root_name}:{shard_name}:{index}", root_name=root_name,
                                shard_name=shard_name, snapshot_index=index, world_seed=seed,
                                original_split=original, split=split, category=category,
                                chain_id=meta.get("chain_id"), chain_index=meta.get("chain_index"),
                                source_key=source_key, repack_source=meta.get("repack_source")))

    chains, membership, anchor_indices = {}, {}, []
    for sql_id, raw, sql_anchor in chain_rows:
        chain = json.loads(raw)
        require(chain.get("id") == sql_id and chain.get("anchor_idx") == sql_anchor,
                f"chain SQL/JSON identity mismatch: {sql_id}")
        category = chain.get("category")
        require(category in DYNAMIC, f"invalid dynamic chain category: {sql_id}")
        ids = chain.get("snapshot_indices")
        require(isinstance(ids, list) and len(ids) == 5 and len(set(ids)) == 5,
                f"chain does not contain five distinct indices: {sql_id}")
        require(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(records) for i in ids),
                f"chain index outside committed snapshots: {sql_id}")
        first = records[ids[0]]
        for position, index in enumerate(ids):
            row = records[index]
            require(index not in membership, f"snapshot {index} belongs to multiple chains")
            require(row.get("category") == category and row.get("chain_id") == sql_id
                    and row.get("chain_index") == position, f"broken chain order/identity: {sql_id}/{index}")
            for field in ("world_seed", "center_origin", "region_origin", "rollout_id"):
                require(field in first and field in row and row[field] == first[field],
                        f"chain mixes {field}: {sql_id}")
            require(sample_info[index]["split"] == sample_info[ids[0]]["split"],
                    f"chain crosses effective splits: {sql_id}")
            membership[index] = sql_id
        if category == "D4":
            require(isinstance(sql_anchor, int) and not isinstance(sql_anchor, bool) and sql_anchor >= 0,
                    f"D4 anchor index missing/invalid: {sql_id}")
            require(chain.get("subtype") in ("A", "B", "C"), f"invalid D4 subtype: {sql_id}")
            anchor = chain.get("anchor_metadata")
            require(isinstance(anchor, dict), f"D4 anchor metadata missing: {sql_id}")
            for field in ("world_seed", "center_origin", "region_origin", "rollout_id"):
                require(field in anchor and anchor[field] == first[field], f"D4 anchor mixes {field}: {sql_id}")
            anchor_indices.append(sql_anchor)
        else:
            require(sql_anchor is None, f"non-D4 chain references anchor: {sql_id}")
        chains[sql_id] = chain
    require(set(membership) == {i for i, row in enumerate(records) if row["category"] != "Static"},
            "incomplete or orphaned dynamic chain membership")
    require(sorted(anchor_indices) == list(range(len(anchor_indices))), "duplicate, missing or orphaned anchor indices")
    for index, row in enumerate(records):
        if row["category"] == "Static":
            require(row.get("chain_id") is None and row.get("chain_index") is None,
                    f"Static snapshot {index} unexpectedly belongs to a chain")

    snapshots = _tokens_array(files, "snapshots", len(records), label)
    anchors = (_tokens_array(files, "anchors", len(anchor_indices), label)
               if anchor_indices or "anchors.zarr/tokens/.zarray" in files else None)
    zero = lambda: np.zeros(K, dtype=np.int64)
    frequency = {split: {group: dict(voxel_count=zero(), sample_count=zero(), snapshot_count=0)
                         for group in GROUPS} for split in SPLITS}
    center_counts = {split: zero() for split in SPLITS}
    d4_counts = {key: zero() for key, chain in chains.items() if chain["category"] == "D4"}

    def read_counts(array, index):
        local = np.asarray(array[index])
        require(local.shape == (48, 48, 48) and local.dtype == np.uint16,
                f"invalid decoded full48 state at {index}")
        maximum = int(local.max())
        require(maximum < len(lut), f"unmapped local token {maximum} at state {index}")
        canonical = lut[local]
        count = np.bincount(canonical.ravel(), minlength=K).astype(np.int64, copy=False)
        require(int(count.sum()) == VOLUME, f"full48 occurrence total mismatch at state {index}")
        return canonical, count

    for index, row in enumerate(records):
        canonical, count = read_counts(snapshots, index)
        split, category = sample_info[index]["split"], row["category"]
        for group in (("Static",) if category == "Static" else ("Dynamic", category)):
            stats = frequency[split][group]
            stats["voxel_count"] += count
            stats["sample_count"] += count > 0
            stats["snapshot_count"] += 1
        center_counts[split] += np.bincount(canonical[CENTER].ravel(), minlength=K)
        if category == "D4":
            d4_counts[row["chain_id"]] += count

    anchor_voxels, anchor_samples, paired_voxels = zero(), zero(), zero()
    exclusive_voxels, exclusive_anchors = zero(), zero()
    links = {}
    anchors_with_unseen = 0
    subtype_counts = Counter()
    for chain_id, chain in chains.items():
        if chain["category"] != "D4":
            continue
        index = chain["anchor_idx"]
        _, count = read_counts(anchors, index)
        paired = d4_counts[chain_id]
        present = count > 0
        anchor_voxels += count
        anchor_samples += present
        paired_voxels += paired * present
        unseen_here = present & mask
        exclusive = unseen_here & (paired == 0)
        exclusive_voxels += count * exclusive
        exclusive_anchors += exclusive
        anchors_with_unseen += int(unseen_here.any())
        subtype_counts[chain["subtype"]] += 1
        for token in np.flatnonzero(unseen_here):
            links.setdefault(int(token), []).append(dict(
                anchor_id=f"{root_name}:{shard_name}:{index}", anchor_idx=index,
                root_name=root_name, shard_name=shard_name, chain_id=chain_id,
                subtype=chain["subtype"], anchor_occurrences=int(count[token]),
                chain_occurrences=int(paired[token]), exclusive=bool(exclusive[token])))

    for split in SPLITS:
        for group, stats in frequency[split].items():
            require(int(stats["voxel_count"].sum()) == stats["snapshot_count"] * VOLUME,
                    f"occurrence denominator mismatch for {split}/{group}")
            require(np.all(stats["sample_count"] <= stats["snapshot_count"]),
                    f"sample count exceeds denominator for {split}/{group}")
    return dict(frequency=frequency, center_counts=center_counts, samples=sample_info,
                anchor_voxel_count=anchor_voxels, anchor_sample_count=anchor_samples,
                paired_chain_voxel_count=paired_voxels, exclusive_voxel_count=exclusive_voxels,
                exclusive_anchor_count=exclusive_anchors, anchor_links=links,
                anchor_total=len(anchor_indices), anchors_with_unseen=anchors_with_unseen,
                anchor_subtypes=dict(subtype_counts),
                integrity=dict(committed_snapshots=len(records), dynamic_chains=len(chains),
                               linked_d4_anchors=len(anchor_indices), full48_shapes=True,
                               all_required_chunks_present=True, legal_mapped_ids=True,
                               chain_links_complete=True, canonical_param2_raw_uint8=True))
