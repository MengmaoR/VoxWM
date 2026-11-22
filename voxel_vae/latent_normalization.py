"""Use one frozen train-set normalization for the posterior-mean voxel latent."""
from __future__ import annotations

import numpy as np


def _parameters(value, stats):
    shape = tuple(stats["latent_shape"])
    if tuple(value.shape[-len(shape):]) != shape:
        raise ValueError(f"Expected trailing latent shape {shape}")
    broadcast = (shape[0],) + (1,) * (len(shape) - 1)
    mean = np.asarray(stats["mean"]).reshape(broadcast)
    scale = np.asarray(stats["std"]).reshape(broadcast) + float(stats["eps"])
    if hasattr(value, "new_tensor"):
        return value.new_tensor(mean), value.new_tensor(scale)
    return mean.astype(value.dtype, copy=False), scale.astype(value.dtype, copy=False)


def normalize_latent(mu, stats):
    mean, scale = _parameters(mu, stats)
    return (mu - mean) / scale


def denormalize_latent(z_norm, stats):
    mean, scale = _parameters(z_norm, stats)
    return z_norm * scale + mean
