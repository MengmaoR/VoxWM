"""Rebalance the prepared E01 split using whole seeds from its old test set."""
from collections import Counter
from dataclasses import replace
from random import Random

from .dataset import DatasetIndex


SPLIT_FILE = "split_160k.json"
SPLIT_COUNTS = {"train": 160_000, "validation": 20_000, "test": 20_000}


def _select_exact_seed_count(groups, target):
    # Keep one predecessor per reachable count. Only extend counts that were
    # reachable before this seed so it cannot be used twice.
    reachable = {0: None}
    for seed, size in groups.items():
        if target in reachable:
            break
        for count in list(reachable):
            if count + size <= target:
                reachable.setdefault(count + size, (count, seed))
    if target not in reachable:
        raise ValueError(f"Cannot transfer exactly {target:,} test states as whole world seeds")
    selected = []
    while target:
        target, seed = reachable[target]
        selected.append(seed)
    return sorted(selected)


def select_test_seeds(index):
    counts = Counter(row.split for row in index.records)
    if (sum(counts.values()) != sum(SPLIT_COUNTS.values())
            or any(counts[split] > SPLIT_COUNTS[split] for split in ("train", "validation"))):
        raise ValueError(f"Cannot reach {SPLIT_COUNTS} by moving only original test states: {dict(counts)}")
    sizes = Counter(row.world_seed for row in index.records if row.split == "test")
    seeds = sorted(sizes)
    Random(42).shuffle(seeds)
    groups = {seed: sizes[seed] for seed in seeds}
    selected = {}
    for split in ("validation", "train"):
        selected[split] = _select_exact_seed_count(groups, SPLIT_COUNTS[split] - counts[split])
        for seed in selected[split]:
            del groups[seed]
    return selected


def apply_test_seeds(index, selected):
    if set(selected) != {"train", "validation"}:
        raise ValueError("E01 split must specify train and validation seed lists")
    moves = {seed: split for split, seeds in selected.items() for seed in seeds}
    if len(moves) != sum(map(len, selected.values())):
        raise ValueError("Repeated world seed in E01 split")
    seen = set()
    records = []
    for row in index.records:
        target = moves.get(row.world_seed)
        if target is not None:
            if row.split != "test":
                raise ValueError(f"World seed {row.world_seed} was outside the original test split")
            seen.add(row.world_seed)
            row = replace(row, split=target)
        records.append(row)
    if seen != moves.keys():
        raise ValueError("E01 split references a world seed absent from the prepared index")
    counts = Counter(row.split for row in records)
    if counts != SPLIT_COUNTS:
        raise ValueError(f"Expected exact E01 split {SPLIT_COUNTS}, found {dict(counts)}")
    return DatasetIndex(index.roots, records, index.manifest_hashes, index.vocabulary_hash)
