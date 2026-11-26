"""Rank the 26 peripheral 16³ blocks of a canonical 48³ snapshot."""
from itertools import groupby

import numpy as np


CENTER_INDEX = 13


def blocks(frame):
    """Return blocks in C-order of the 3×3×3 block coordinates, then local XYZ."""
    if frame.shape != (48, 48, 48):
        raise ValueError("Expected a single 48³ voxel state")
    return frame.reshape(3, 16, 3, 16, 3, 16).transpose(0, 2, 4, 1, 3, 5).reshape(27, 4096)


def motion_scores(frames):
    """Count adjacent saved-state transitions with any token change per block."""
    frames = np.asarray(frames)
    if frames.shape != (5, 27, 4096):
        raise ValueError("Expected five ordered states with 27 blocks of 4096 voxels")
    return np.any(frames[1:] != frames[:-1], axis=2).sum(axis=0)


def _rank_entry(block, global_counts, motion, index):
    counts = np.bincount(block)
    ids = np.flatnonzero(counts)
    frequencies = global_counts[ids]
    negative_counts = -counts[ids]
    negative_ids = -ids
    # This also assigns a deterministic correspondence to equal-frequency entries.
    order = np.lexsort((negative_ids, negative_counts, frequencies))
    return (-motion, tuple(frequencies[order].tolist()), tuple(negative_counts[order].tolist()),
            tuple(negative_ids[order].tolist()), index)


def select_frame(frame_blocks, global_counts, motion=None):
    """Return the top-eight peripheral indices and per-frame tie counts.

    Whole frequency vectors compare first, with Python's shorter-prefix-first
    lexicographic order. Local counts and canonical IDs each break ties only
    after the entire preceding vector is equal. The final key is the spatial index.
    """
    ranked = sorted(_rank_entry(block, global_counts, int(motion[i]) if motion is not None else 0, i)
                    for i, block in enumerate(frame_blocks) if i != CENTER_INDEX)
    stats = dict(tied_pairs=0, tied_groups=0, tied_chunks=0, frames_with_tie=0,
                 pairs_tied_after_id=0)
    for k in (4, 6, 8):
        stats[f"cutoff_{k}_tie"] = int(ranked[k - 1][:3] == ranked[k][:3])
    for _, group in groupby(ranked, key=lambda entry: entry[:3]):
        members = list(group)
        n = len(members)
        if n > 1:
            stats["frames_with_tie"] = 1
            stats["tied_groups"] += 1
            stats["tied_chunks"] += n
            stats["tied_pairs"] += n * (n - 1) // 2
            for _, identical in groupby(members, key=lambda entry: entry[3]):
                m = sum(1 for _ in identical)
                stats["pairs_tied_after_id"] += m * (m - 1) // 2
    return [entry[-1] for entry in ranked[:8]], stats
