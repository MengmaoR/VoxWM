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
