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
from .outer_chunk_rank import blocks, mixed_outer_indices, motion_scores, select_frame

LIMITS = (0, 4, 6, 8)
MIXED_LIMITS = (0, 2, 3, 4)
FREQUENCY = REPORTS / 'full48_embedding_audit_20261126/static_dynamic_embedding_frequency.csv'


def full_counts_by_category(path, ref):
    counts = {category: np.zeros(ref['K'], dtype=np.int64) for category in ('Static', 'Dynamic')}
    keys = set()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            i = int(row['canonical_token_id'])
            key = (row['split'], row['category'], i)
            if key in keys or not 0 <= i < ref['K'] or ref['pairs'][i] != (row['node_name'], int(row['raw_param2'])):
                raise ValueError(f'Invalid token frequency row {key}')
            keys.add(key)
            counts[row['category']][i] += int(row['voxel_count'])
    if len(keys) != 6 * ref['K'] or any(row.sum() != 100_000 * 48**3 for row in counts.values()):
        raise ValueError('Expected all six Static/Dynamic split frequency tables')
    return counts


def pooled_counts(path, ref):
    return sum(full_counts_by_category(path, ref).values())


def record_mixed_frame(frame, motion, frequency, increments, _ties):
    selected = mixed_outer_indices(frame, frequency, motion, limits=MIXED_LIMITS[1:])
    previous = set()
    increments[0] += np.bincount(frame[13], minlength=len(frequency))
    for target, limit in zip(increments[1:], MIXED_LIMITS[1:]):
        current = set(selected[limit])
        if len(current) != limit or 13 in current or not previous <= current:
            raise ValueError('Expected nested, distinct outer selections')
        added = list(current - previous)
        target += np.bincount(frame[added].ravel(), minlength=len(frequency))
        previous = current


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


def scan(archive, frequency, mixed=False):
    record = record_mixed_frame if mixed else record_frame
    limits = MIXED_LIMITS if mixed else LIMITS
    ref = load_reference()
    increments = np.zeros((len(limits), len(frequency)), dtype=np.int64)
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
                record(blocks(snapshots[i]), None, frequency, increments, ties)
        else:
            for indices in chain_indices(item.files, n):
                chain = np.stack([blocks(snapshots[i]) for i in indices])
                motion = motion_scores(chain)
                for frame in chain:
                    record(frame, motion, frequency, increments, ties)
                n_chains += 1
        frames += n
        n_shards += 1
        if n_shards == 1 or n_shards % 10 == 0:
            print(f'{category}: {n_shards}/200 shards, {frames:,} frames', flush=True)
    if frames != 100_000 or (category == 'Dynamic' and n_chains != 20_000):
        raise ValueError(f'Incomplete {category} archive')
    counts = increments.cumsum(axis=0)
    for limit, row in zip(limits, counts):
        if row.sum() != frames * (limit + 1) * 16**3:
            raise ValueError(f'Unexpected selected voxel total for {category} K={limit}')
    return category, counts, dict(ties), frames, n_chains


def report(results, frequency, ref, seconds, limits=LIMITS):
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
    for index, limit in enumerate(limits):
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


def add_mixed_details(summary, full, results):
    groups = {'static_only': (full['Static'] > 0) & (full['Dynamic'] == 0),
              'dynamic_only': (full['Dynamic'] > 0) & (full['Static'] == 0),
              'shared': (full['Dynamic'] > 0) & (full['Static'] > 0)}
    del summary['ties']  # Tie statistics belong to the original audit.
    summary['policy'] = 'Static rarity top K; Dynamic motion top K-1 plus rarest remaining; always include center'
    for index, limit in enumerate(MIXED_LIMITS):
        selection = summary['selection'][limit]
        misses = selection['present_but_missed']
        totals = dict.fromkeys(groups, 0)
        for token in misses:
            i = token['id']
            group, = (name for name, mask in groups.items() if mask[i])
            token['full48_category'] = group
            token['static_voxels'] = int(full['Static'][i])
            token['dynamic_voxels'] = int(full['Dynamic'][i])
            totals[group] += 1
        selection['missed_by_category'] = totals
        selection['selected_id6_voxels'] = sum(int(counts[index, 6]) for _, counts, *_ in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, nargs=2, help='Static and Dynamic canonical .tar.zst packs')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--mixed', action='store_true', help='Reserve one Dynamic outer slot for rarity (K=2,3,4)')
    args = parser.parse_args()
    start = time.monotonic()
    ref = load_reference()
    full = full_counts_by_category(FREQUENCY, ref)
    frequency = sum(full.values())
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(scan, archive, frequency, args.mixed) for archive in args.archive]
        results = [f.result() for f in futures]
    limits = MIXED_LIMITS if args.mixed else LIMITS
    summary = report(results, frequency, ref, time.monotonic() - start, limits)
    if args.mixed:
        add_mixed_details(summary, full, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for limit, row in summary['selection'].items():
        print(f"center + {limit}: {row['unique']} observed / {row['missing']} missing")
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
