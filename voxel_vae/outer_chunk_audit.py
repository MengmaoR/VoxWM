"""Coverage of center plus rarity/motion-selected outer 16³ chunks in the VAE packs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from .full48_scan import _tokens_array
from .full48_source import REPORTS, iter_archive_shards, load_reference, validate_source_header
from .outer_chunk_rank import blocks, motion_scores, select_frame

LIMITS = (0, 4, 6, 8)
FREQUENCY = REPORTS / 'full48_embedding_audit_20261126/static_dynamic_embedding_frequency.csv'


def pooled_counts(path, ref):
    counts = np.zeros(ref['K'], dtype=np.int64)
    keys = set()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            i = int(row['canonical_token_id'])
            key = (row['split'], row['category'], i)
            if key in keys or not 0 <= i < len(counts) or ref['pairs'][i] != (row['node_name'], int(row['raw_param2'])):
                raise ValueError(f'Invalid token frequency row {key}')
            keys.add(key)
            counts[i] += int(row['voxel_count'])
    if len(keys) != 6 * ref['K'] or counts.sum() != 200_000 * 48**3:
        raise ValueError('Expected all six Static/Dynamic split frequency tables')
    return counts


def record_frame(frame, motion, frequency, increments, ties):
    ranked, frame_ties = select_frame(frame, frequency, motion)
    ties.update(frame_ties)
    for target, coords in zip(increments, ([13], ranked[:4], ranked[4:6], ranked[6:8])):
        target += np.bincount(frame[coords].ravel(), minlength=len(frequency))


def chain_indices(files, n):
    with closing(sqlite3.connect(':memory:')) as db:
        db.deserialize(files['journal.sqlite'])
        chains = [json.loads(row[0])['snapshot_indices']
                  for row in db.execute('SELECT data FROM chains ORDER BY id')]
    if any(len(c) != 5 for c in chains) or sorted(i for c in chains for i in c) != list(range(n)):
        raise ValueError('Expected disjoint, complete five-frame chains')
    return chains


def scan(archive, frequency):
    ref = load_reference()
    increments = np.zeros((len(LIMITS), len(frequency)), dtype=np.int64)
    ties = Counter()
    frames = n_chains = n_shards = 0
    category = None
    for item in iter_archive_shards(archive):
        if category is None:
            category = validate_source_header(item.header, ref)['category']
        n = json.loads(item.files['READY'])['samples']
        snapshots = _tokens_array(item.files, 'snapshots', n, item.shard_name)
        if category == 'Static':
            for i in range(n):
                record_frame(blocks(snapshots[i]), None, frequency, increments, ties)
        else:
            for indices in chain_indices(item.files, n):
                chain = np.stack([blocks(snapshots[i]) for i in indices])
                motion = motion_scores(chain)
                for frame in chain:
                    record_frame(frame, motion, frequency, increments, ties)
                n_chains += 1
        frames += n
        n_shards += 1
        if n_shards == 1 or n_shards % 10 == 0:
            print(f'{category}: {n_shards}/200 shards, {frames:,} frames', flush=True)
    if frames != 100_000 or (category == 'Dynamic' and n_chains != 20_000):
        raise ValueError(f'Incomplete {category} archive')
    counts = increments.cumsum(axis=0)
    for limit, row in zip(LIMITS, counts):
        if row.sum() != frames * (limit + 1) * 16**3:
            raise ValueError(f'Unexpected selected voxel total for {category} K={limit}')
    return category, counts, dict(ties), frames, n_chains


def report(results, frequency, ref, seconds):
    by_category = {category: counts for category, counts, *_ in results}
    if set(by_category) != {'Static', 'Dynamic'}:
        raise ValueError('Supply one Static pack and one Dynamic pack')
    counts = by_category['Static'] + by_category['Dynamic']
    if not np.array_equal(counts[0], sum(ref['baseline'].values())) or np.any(counts[-1] > frequency):
        raise ValueError('Selected counts disagree with the pooled reference')
    output = dict(
        seconds=round(seconds, 2), vocabulary_size=len(frequency),
        ordinary_snapshots={name: frames for name, _, _, frames, _ in results},
        five_frame_chains=sum(r[-1] for r in results),
        globally_absent_ids=np.flatnonzero(frequency == 0).tolist(),
        ties={category: ties for category, _, ties, *_ in results}, selection={},
    )
    for index, limit in enumerate(LIMITS):
        row = counts[index]
        ids = np.flatnonzero((row == 0) & (frequency > 0))
        output['selection'][limit] = dict(
            selected_chunks=int(row.sum() // 16**3),
            unique=int(np.count_nonzero(row)), missing=int(np.count_nonzero(row == 0)),
            recovered_outer_only=int(np.count_nonzero((row > 0) & (counts[0] == 0))),
            static_unique=int(np.count_nonzero(by_category['Static'][index])),
            dynamic_unique=int(np.count_nonzero(by_category['Dynamic'][index])),
            present_but_missed=[dict(id=int(i), node=ref['pairs'][i][0], raw_param2=ref['pairs'][i][1],
                                     full48_voxels=int(frequency[i])) for i in ids],
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, nargs=2, help='Static and Dynamic canonical .tar.zst packs')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    args = parser.parse_args()
    start = time.monotonic()
    ref = load_reference()
    frequency = pooled_counts(FREQUENCY, ref)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(scan, archive, frequency) for archive in args.archive]
        results = [f.result() for f in futures]
    summary = report(results, frequency, ref, time.monotonic() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for limit, row in summary['selection'].items():
        print(f"center + {limit}: {row['unique']} observed / {row['missing']} missing")
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
