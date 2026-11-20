"""CPU-only changed-voxel masks for complete five-state dynamic chains."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .dataset import VoxelSnapshotDataset


DYNAMIC_CATEGORIES = ("D1", "D2", "D3", "D4")


def compute_chain_masks(states, chain_indices=range(5)):
    """Return endpoint/neighbor-union masks for canonical center-token states."""
    states = np.asarray(states)
    indices = list(map(int, chain_indices))
    if states.shape != (5, 16, 16, 16):
        raise ValueError(f"Expected five canonical [16,16,16] states, got {states.shape}")
    if indices != list(range(5)):
        raise ValueError(f"Dynamic chain_index must be exactly 0..4, got {indices}")
    adjacent = [states[index] != states[index + 1] for index in range(4)]
    if any(not difference.any() for difference in adjacent):
        raise ValueError("Every adjacent dynamic state pair must differ")
    masks = np.empty(states.shape, dtype=bool)
    masks[0] = adjacent[0]
    masks[4] = adjacent[3]
    for index in range(1, 4):
        masks[index] = adjacent[index - 1] | adjacent[index]
    if any(not mask.any() for mask in masks):
        raise ValueError("Every dynamic state must have a non-empty changed mask")
    return masks


def _chain_key(row):
    return row.root_index, row.shard, row.chain_id


class ChangeMaskCache:
    """Cache masks only for selected dynamic chains; Static has no mask."""

    def __init__(self, index, vocabulary, selected_indices):
        self.index = index
        self.vocabulary = vocabulary
        selected_indices = tuple(map(int, selected_indices))
        requested = set()
        for sample_index in selected_indices:
            row = index.records[sample_index]
            if row.category in DYNAMIC_CATEGORIES:
                if row.chain_id is None:
                    raise ValueError(f"Dynamic sample lacks chain_id: {row.key}")
                requested.add(_chain_key(row))
        groups = defaultdict(list)
        for sample_index, row in enumerate(index.records):
            if row.chain_id is not None and _chain_key(row) in requested:
                groups[_chain_key(row)].append(sample_index)
        if set(groups) != requested:
            raise ValueError("Unable to resolve every requested dynamic chain")

        dataset = VoxelSnapshotDataset(index, vocabulary)
        self._masks = {}
        self._chains = {}
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda item: index.records[item].chain_index)
            rows = [index.records[item] for item in members]
            if len(rows) != 5 or [row.chain_index for row in rows] != list(range(5)):
                raise ValueError(f"Dynamic chain must contain ordered indices 0..4: {key}")
            categories = {row.category for row in rows}
            seeds = {row.world_seed for row in rows}
            splits = {row.split for row in rows}
            if len(categories) != 1 or next(iter(categories)) not in DYNAMIC_CATEGORIES:
                raise ValueError(f"Dynamic chain category mismatch: {key}")
            if len(seeds) != 1 or len(splits) != 1:
                raise ValueError(f"Dynamic chain crosses seed or split: {key}")
            states = np.stack([dataset.read_numpy(item)[0] for item in members])
            masks = compute_chain_masks(states, [row.chain_index for row in rows])
            self._chains[key] = tuple(members)
            for sample_index, mask in zip(members, masks):
                self._masks[sample_index] = np.ascontiguousarray(mask)

        missing = [item for item in selected_indices
                   if index.records[item].category in DYNAMIC_CATEGORIES
                   and item not in self._masks]
        if missing:
            raise ValueError(f"Missing changed masks for selected samples: {missing[:5]}")

    def __contains__(self, sample_index):
        return int(sample_index) in self._masks

    def __getitem__(self, sample_index):
        return self.changed(sample_index)

    def changed(self, sample_index):
        sample_index = int(sample_index)
        if sample_index not in self._masks:
            raise KeyError(f"No dynamic mask for sample {sample_index}")
        return self._masks[sample_index]

    def unchanged(self, sample_index):
        return ~self.changed(sample_index)

    def statistics(self, selected_indices):
        selected = set(map(int, selected_indices))
        category_counts = defaultdict(list)
        chain_rows = []
        empty_unchanged = 0
        for key, members in sorted(self._chains.items()):
            active = [item for item in members if item in selected]
            if not active:
                continue
            row = self.index.records[members[0]]
            counts = [int(self._masks[item].sum()) for item in members]
            category_counts[row.category].extend(
                int(self._masks[item].sum()) for item in active)
            empty_unchanged += sum(bool(self._masks[item].all()) for item in active)
            chain_rows.append({
                "category": row.category,
                "chain_id": row.chain_id,
                "root_index": row.root_index,
                "shard": row.shard,
                "world_seed": row.world_seed,
                "changed_voxels_per_state": counts,
                "selected_states": len(active),
                "d4_subtype": row.d4_subtype,
            })

        def distribution(values):
            array = np.asarray(values, dtype=np.float64)
            return {
                "states": int(array.size),
                "changed_voxels": int(array.sum()),
                "mean_changed_voxels": float(array.mean()),
                "min_changed_voxels": int(array.min()),
                "max_changed_voxels": int(array.max()),
                "mean_changed_ratio": float((array / 4096.0).mean()),
                "min_changed_ratio": float(array.min() / 4096.0),
                "max_changed_ratio": float(array.max() / 4096.0),
            }

        by_category = {category: distribution(category_counts[category])
                       for category in DYNAMIC_CATEGORIES if category_counts[category]}
        all_counts = [value for values in category_counts.values() for value in values]
        d4_subtypes = Counter(row["d4_subtype"] or "unspecified" for row in chain_rows
                              if row["category"] == "D4")
        return {
            "chains": len(chain_rows),
            "dynamic_states": len(all_counts),
            "overall": None if not all_counts else distribution(all_counts),
            "by_category": by_category,
            "chain_changed_mask_counts": chain_rows,
            "empty_unchanged_states": int(empty_unchanged),
            "d4_chain_subtypes": dict(sorted(d4_subtypes.items())),
        }
