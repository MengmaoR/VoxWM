"""Deterministic category-balanced effective batches."""
from __future__ import annotations

from collections import Counter

import numpy as np


BALANCE = {"Static": 32, "D1": 8, "D2": 8, "D3": 8, "D4": 8}


class BalancedBatchSampler:
    """Draw exactly 32/8/8/8/8 train samples for each global optimizer step."""

    def __init__(self, index, eligible_indices=None, seed=42, balance=None):
        self.index = index
        self.seed = int(seed)
        self.balance = dict(BALANCE if balance is None else balance)
        if sum(self.balance.values()) != 64:
            raise ValueError("Balanced effective batch must contain exactly 64 samples")
        eligible = index.indices(split="train") if eligible_indices is None else map(int, eligible_indices)
        self.pools = {category: [] for category in self.balance}
        for sample_index in eligible:
            row = index.records[sample_index]
            if row.split != "train":
                continue
            if row.category in self.pools:
                self.pools[row.category].append(sample_index)
        for category, quota in self.balance.items():
            if len(self.pools[category]) < quota:
                raise ValueError(
                    f"Need {quota} train {category} samples, found {len(self.pools[category])}")
        self.pools = {category: np.asarray(pool, dtype=np.int64)
                      for category, pool in self.pools.items()}

    def batch(self, global_step):
        rng = np.random.default_rng(self.seed + 1_000_003 * int(global_step))
        result = []
        for category, quota in self.balance.items():
            selected = rng.choice(self.pools[category], size=quota, replace=False)
            result.extend(map(int, selected))
        rng.shuffle(result)
        return result

    def counts(self, batch):
        return Counter(self.index.records[index].category for index in batch)


class BalancedEpochSampler(BalancedBatchSampler):
    """Shuffle each category pool and use each state once per epoch."""

    def __init__(self, index, eligible_indices=None, seed=42):
        super().__init__(index, eligible_indices, seed)
        self.samples = sum(map(len, self.pools.values()))
        if np.unique(np.concatenate(list(self.pools.values()))).size != self.samples:
            raise ValueError("Epoch sampler requires unique training indices")
        self.steps_per_epoch = (self.samples + 63) // 64
        if any(len(pool) < self.steps_per_epoch for pool in self.pools.values()):
            raise ValueError("Epoch sampler needs at least one state per category in each batch")
        self._epoch = None
        self._layout = {}

    def _prepare_epoch(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        slots = rng.permutation(self.steps_per_epoch)
        cursor = 0
        for category, pool in self.pools.items():
            base, remainder = divmod(len(pool), self.steps_per_epoch)
            counts = np.full(self.steps_per_epoch, base, dtype=np.int64)
            # Spread extras across batches, keeping their total sizes within one state.
            counts[slots[np.arange(cursor, cursor + remainder) % self.steps_per_epoch]] += 1
            cursor += remainder
            self._layout[category] = (rng.permutation(pool), np.r_[0, counts.cumsum()])
        self._epoch = epoch

    def batch(self, step):
        """Return a batch by its zero-based step relative to the epoch schedule start."""
        epoch, position = divmod(int(step), self.steps_per_epoch)
        if epoch != self._epoch:
            self._prepare_epoch(epoch)
        batch = []
        for pool, offsets in self._layout.values():
            batch.extend(pool[offsets[position]:offsets[position + 1]].tolist())
        rng = np.random.default_rng([self.seed, epoch, position, 1])
        rng.shuffle(batch)
        return batch
