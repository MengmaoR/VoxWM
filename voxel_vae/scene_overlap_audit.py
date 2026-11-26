"""Count geometrically overlapping 48³ regions in the canonical VAE category packs."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .full48_source import iter_archive_shards
from .scene_overlap_geometry import summarize

REGION_FILES = frozenset({'READY', 'journal.sqlite'})


def read_scenes(path):
    scenes = []
    frames = 0
    category = None
    for ns, shard in enumerate(iter_archive_shards(path, select_file=REGION_FILES.__contains__), 1):
        category = shard.header['json']['dataset_manifest.json']['repack']['category']
        if category not in ('Static', 'Dynamic'):
            raise ValueError(f'Unexpected pack {category}')
        with closing(sqlite3.connect(':memory:')) as db:
            db.deserialize(shard.files['journal.sqlite'])
            rows = db.execute("""
                SELECT idx, category, json_extract(metadata, '$.world_seed',
                    '$.region_origin', '$.center_origin', '$.chain_id', '$.chain_index',
                    '$.repack_source.clip') FROM samples ORDER BY idx
            """).fetchall()
            chain_lengths = dict(db.execute("SELECT id, json_array_length(data, '$.snapshot_indices') FROM chains"))
        if len(rows) != json.loads(shard.files['READY'])['samples']:
            raise ValueError(f'{shard.shard_name}: incomplete journal')
        chains = defaultdict(list)
        for idx, event, fields in rows:
            seed, origin, inner, chain, step, clip = json.loads(fields)
            if (len(origin) != 3 or any(type(v) is not int for v in origin) or
                    [v + 16 for v in origin] != inner or type(seed) is not int or not clip):
                raise ValueError(f'{shard.shard_name}:{idx}: invalid voxel coordinates or source')
            center = tuple(v + 24 for v in origin)
            if category == 'Static':
                if event != 'Static' or chain is not None:
                    raise ValueError(f'{shard.shard_name}:{idx}: expected Static')
                scenes.append((seed, center, event, clip, f'{shard.shard_name}:{idx}'))
            else:
                if event not in ('D1','D2','D3','D4') or not chain:
                    raise ValueError(f'{shard.shard_name}:{idx}: expected a dynamic chain')
                scene = (seed, center, event, clip, f'{shard.shard_name}:{chain}')
                chains[chain].append((step, scene))
        if set(chains) != set(chain_lengths):
            raise ValueError(f'{shard.shard_name}: recorded chains disagree with sample rows')
        for chain, group in chains.items():
            positions = sorted(step for step, _ in group)
            scene = group[0][1]
            if (chain_lengths[chain] != 5 or positions != list(range(5)) or
                    any(other != scene for _, other in group)):
                raise ValueError(f'{chain}: inconsistent five-frame footprint')
            scenes.append(scene)
        frames += len(rows)
        if ns == 1 or ns % 25 == 0:
            print(f'{category}: {ns} shards, {frames:,} frames', flush=True)
    if category is None or frames != 100_000 or len(scenes) != (100_000 if category == 'Static' else 20_000):
        raise ValueError(f'{path}: incomplete scene count')
    return category, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs=2, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=2) as pool:
        packs = dict(pool.map(read_scenes, args.archives))
    static, dynamic = packs['Static'], packs['Dynamic']
    result = dict(center='region_origin + (24,24,24); integer axes east/north/up',
                  region_edge_voxels=48, dynamic_frames_per_scene=5,
                  static=summarize(static), dynamic=summarize(dynamic),
                  static_dynamic=summarize(static + dynamic, cross=True))
    d = result['dynamic']
    within = len(dynamic) * 10
    between = d['overlap_pairs'] * 25
    result['dynamic_snapshot_pairs'] = dict(
        within_chain=within, between_chains=between, total_overlapping=within+between)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
