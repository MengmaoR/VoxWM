"""Streaming moments and input-token diagnostics for the frozen VAE."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .losses import DYNAMIC_CATEGORIES


class Moments:
    """Merge batches by their counts, means, and centered sums of squares."""

    def __init__(self, channels):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)
        self.minimum = np.full_like(self.mean, np.inf)
        self.maximum = np.full_like(self.mean, -np.inf)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        count = len(values)
        if not count:
            return
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite audit observation")
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += ((values - mean) ** 2).sum(axis=0) + delta ** 2 * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def summary(self):
        if not self.count:
            raise ValueError("No observations for statistics")
        variance = self.m2 / self.count
        global_mean = self.mean.mean()
        global_variance = (variance + (self.mean - global_mean) ** 2).mean()
        return {
            "observations_per_channel": self.count,
            "mean": self.mean.tolist(), "std": np.sqrt(variance).tolist(),
            "variance": variance.tolist(),
            "min": self.minimum.tolist(), "max": self.maximum.tolist(),
            "global_mean": float(global_mean), "global_std": float(np.sqrt(global_variance)),
            "global_variance": float(global_variance),
            "global_min": float(self.minimum.min()), "global_max": float(self.maximum.max()),
        }


def channel_observations(mu):
    return np.moveaxis(mu, 1, -1).reshape(-1, mu.shape[1])


class TrainLatents:
    def __init__(self, channels):
        self.global_moments = Moments(channels)
        self.by_category = {name: Moments(channels) for name in ("Static", *DYNAMIC_CATEGORIES)}
        self.samples = Counter()

    def update(self, mu, categories):
        categories = np.asarray(categories)
        self.global_moments.update(channel_observations(mu))
        for name, moments in self.by_category.items():
            selected = categories == name
            self.samples[name] += int(selected.sum())
            moments.update(channel_observations(mu[selected]))

    def summary(self):
        result = self.global_moments.summary()
        active = int(np.count_nonzero(np.asarray(result["std"]) >= 1e-3))
        return {"samples": sum(self.samples.values()), **result,
                "active_latent_channels": active, "near_collapse_std_threshold": 1e-3,
                "by_category": {name: {"samples": self.samples[name], **moments.summary()}
                                for name, moments in self.by_category.items() if moments.count}}


def add_token_counts(counts, tokens):
    counts += np.bincount(np.asarray(tokens).reshape(-1), minlength=len(counts))


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    breaks = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(order)]
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.repeat((breaks[:-1] + breaks[1:] + 1) / 2, np.diff(breaks))
    return ranks


def embedding_coverage(vocabulary, counts, initial, final):
    initial, final = np.asarray(initial, dtype=np.float64), np.asarray(final, dtype=np.float64)
    if initial.shape != final.shape or initial.shape[0] != vocabulary.K:
        raise ValueError("Embedding matrices do not match the canonical vocabulary")
    if not (np.isfinite(initial).all() and np.isfinite(final).all()):
        raise FloatingPointError("Non-finite embedding rows")
    l2 = np.linalg.norm(final - initial, axis=1)
    initial_norm = np.linalg.norm(initial, axis=1)
    final_norm = np.linalg.norm(final, axis=1)
    product = initial_norm * final_norm
    cosine = np.divide(np.sum(initial * final, axis=1), product,
                       out=np.zeros_like(product), where=product > 0)
    train = counts["train"]
    held_out = (counts["validation"] + counts["test"]) > 0
    base_rows, drift_rows = [], []
    for token, (name, param2) in enumerate(vocabulary.pairs):
        identity = {"token_id": token, "node_name": name, "param2": param2}
        base_rows.append({**identity, **{f"{split}_count": int(values[token])
                                         for split, values in counts.items()}})
        drift_rows.append({**identity, **{f"{split}_frequency": int(values[token])
                                          for split, values in counts.items()},
                           "embedding_l2_drift": float(l2[token]),
                           "embedding_cosine": float(cosine[token]),
                           "initial_norm": float(initial_norm[token]),
                           "final_norm": float(final_norm[token])})
    low_drift = l2 <= 1e-6
    anomalies = [drift_rows[i] for i in np.flatnonzero((train > 0) & low_drift)]
    bins = {}
    for label, lower, upper in (
        ("0", 0, 0), ("1–10", 1, 10), ("11–100", 11, 100), ("101–1k", 101, 1_000),
        ("1k–10k", 1_001, 10_000), (">10k", 10_001, np.inf),
    ):
        selected = l2[(train >= lower) & (train <= upper)]
        bins[label] = {"token_count": int(selected.size),
                       "median_drift": float(np.median(selected)) if selected.size else None,
                       "mean_drift": float(selected.mean()) if selected.size else None}
    x, y = _average_ranks(np.log1p(train)), _average_ranks(l2)
    spearman = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    summary = {
        "vocabulary_K": vocabulary.K, "low_drift_threshold": 1e-6,
        "seen_in_train": int((train > 0).sum()),
        "unseen_in_train_but_seen_in_held_out": int(((train == 0) & held_out).sum()),
        "never_in_any_center_split": int(((train == 0) & ~held_out).sum()),
        "rows_l2_above_1e-6": int((~low_drift).sum()),
        "non_drifting_rows": int(low_drift.sum()),
        "non_drifting_unseen_train_rows": int(((train == 0) & low_drift).sum()),
        "seen_but_no_drift": anomalies,
        "spearman_log1p_train_frequency_vs_drift": spearman, "frequency_bins": bins,
    }
    return base_rows, drift_rows, summary
