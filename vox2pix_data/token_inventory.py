"""Read-only counts of observed (node ID, raw param2) in a Vox2Pix production root."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import zarr
from numcodecs import blosc


def count_batch(job):
    directories, layer, names = job
    blosc.set_nthreads(1)
    # Reserve one extra bucket for the white-model sky / grid-exit sentinel.
    background = len(names) * 256
    total = np.zeros(background + 1, dtype=np.int64)
    lookup = np.full(65536, -1, dtype=np.int64)
    lookup[:len(names)] = np.arange(len(names), dtype=np.int64) * 256
    if layer == 'surface':
        lookup[65535] = background
    views = 0
    kinds = Counter()
    pixels = 0
    for scene in directories:
        meta = json.loads((scene / 'scene.json').read_text())
        nviews = len(meta['views'])
        kinds[meta['kind']] += 1
        views += nviews
        group = zarr.open_group(str(scene / 'voxels.zarr'), mode='r')
        nodes = group[layer + '_node_id'][:]
        param2 = group[layer + '_param2'][:]
        shape = (48, 48, 48) if layer == 'scene' else (nviews, 360, 640)
        if nodes.shape != shape or nodes.dtype != np.uint16 or param2.shape != shape or param2.dtype != np.uint8:
            raise ValueError(f'Unexpected arrays in {scene}')
        codes = lookup[nodes]
        if codes.min() < 0:
            raise ValueError(f'Unregistered node ID in {scene}')
        codes += param2
        hist = np.bincount(codes.ravel())
        if len(hist) > len(total):
            raise ValueError(f'Invalid BACKGROUND param2 in {scene}')
        total[:len(hist)] += hist
        pixels += nodes.size
    if total.sum() != pixels:
        raise ValueError('Occurrence denominator mismatch')
    present = np.flatnonzero(total)
    return (len(directories), views, kinds, pixels, {int(i): int(total[i]) for i in present})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--layer', choices=('scene', 'surface'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    names = json.loads((args.root / 'node_vocabulary.json').read_text())
    manifest = json.loads((args.root / 'manifest.json').read_text())
    directories = sorted((args.root / 'scenes').glob('scene_*'))
    if len(directories) != manifest['config']['scenes'] or len(set(names)) != len(names):
        raise ValueError('Incomplete dataset or duplicate node names')
    jobs = ((directories[i:i+25], args.layer, names) for i in range(0, len(directories), 25))
    counts = Counter()
    kinds = Counter()
    scenes = views = positions = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, v, k, size, observed in pool.map(count_batch, jobs):
            counts.update(observed)
            kinds.update(k)
            scenes += n
            views += v
            positions += size
            if scenes % 500 == 0 or scenes == len(directories):
                print(f'{args.layer}: {scenes}/{len(directories)} scenes', file=sys.stderr, flush=True)
    background = counts.pop(len(names)*256, 0)
    print(json.dumps(dict(layer=args.layer, scenes=scenes, views=views, scene_kinds=kinds,
                          positions=positions, background=background, names=names, counts=counts)))


if __name__ == '__main__':
    main()
