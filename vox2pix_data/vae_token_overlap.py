"""Compare a Vox2Pix scene/surface inventory with frozen VAE semantic IDs and input weights."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

VAE_REPORTS = Path(__file__).resolve().parents[1] / 'voxel_vae/reports'
LAYERS = ('scene', 'surface')


def read_inventory(path, layer):
    data = json.loads(path.read_text())
    names = data['names']
    counts = {}
    for encoded, n in data['counts'].items():
        node, param2 = divmod(int(encoded), 256)
        if not (0 <= node < len(names) and n > 0):
            raise ValueError(f'Invalid {layer} token {encoded}')
        counts[names[node], param2] = n
    expected = data['scenes'] * 48**3 if layer == 'scene' else data['views'] * 360 * 640
    if sum(counts.values()) + data['background'] != expected or data['positions'] != expected:
        raise ValueError(f'Occurrence denominator mismatch for {layer}')
    return data, counts


def vae_reference():
    vocab, trained, full = {}, set(), Counter()
    with (VAE_REPORTS / 'e01_freeze_audit_20261122/embedding_audit.csv').open() as f:
        for row in csv.DictReader(f):
            key = (row['node_name'], int(row['param2']))
            vocab[key] = int(row['token_id'])
            if int(row['train_frequency']) > 0 and float(row['embedding_l2_drift']) > 1e-6:
                trained.add(key)
    with (VAE_REPORTS / 'full48_embedding_audit_20261126/static_dynamic_embedding_frequency.csv').open() as f:
        for row in csv.DictReader(f):
            key = (row['node_name'], int(row['raw_param2']))
            if vocab[key] != int(row['canonical_token_id']):
                raise ValueError('VAE semantic ID mismatch')
            full[key] += int(row['voxel_count'])
    return vocab, trained, full


def summarize(layer, counts, vocab, trained, full):
    observed = set(counts)
    full_set = {key for key in full if full[key]}
    vae_names = {key[0] for key in full_set}
    masks = dict(all=observed, in_frozen_vocab=observed & vocab.keys(),
                 present_in_vae=observed & full_set, trained_embedding=observed & trained,
                 in_vae_without_trained_embedding=(observed & full_set) - trained,
                 frozen_vocab_but_absent_from_vae=(observed & vocab.keys()) - full_set,
                 outside_vae=observed - full_set,
                 outside_vae_vocab=observed - vocab.keys(),
                 new_param2_on_vae_node={t for t in observed-full_set if t[0] in vae_names},
                 node_absent_from_vae={t for t in observed if t[0] not in vae_names})
    return dict(layer=layer, observed_node_names=len({t[0] for t in observed}),
                vae_only=len(full_set-observed), union=len(full_set|observed),
                groups={label: dict(types=len(keys), occurrences=sum(counts[t] for t in keys))
                        for label,keys in masks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene, raw = read_inventory(args.scene, 'scene')
    surface, white = read_inventory(args.surface, 'surface')
    if (scene['names'] != surface['names'] or scene['scenes'] != surface['scenes']
            or scene['views'] != surface['views'] or scene['scene_kinds'] != surface['scene_kinds']):
        raise ValueError('Layer inventories refer to different datasets')
    if not white.keys() <= raw.keys():
        raise ValueError('White surface contains a state absent from the raw 48³ inventory')
    vocab, trained, full = vae_reference()
    summary = dict(scenes=scene['scenes'], views=surface['views'], kinds=scene['scene_kinds'],
                   registered_node_names=len(scene['names']), surface_background=surface['background'],
                   actual_vae_states=sum(n>0 for n in full.values()), vae_vocab_K=len(vocab),
                   vae_embedding_rows_trained=len(trained), layers={})
    for layer, counts in zip(LAYERS, (raw, white)):
        summary['layers'][layer] = summarize(layer, counts, vocab, trained, full)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    (out/'coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (out/'token_counts.csv').open('w', newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['node_name','raw_param2','vox2pix_48_voxels','vox2pix_surface_pixels',
                         'vae_48_voxels','vae_canonical_id','vae_trained_input_embedding'])
        for key in sorted(raw.keys() | vocab.keys()):
            writer.writerow([*key,raw.get(key,0),white.get(key,0), full[key],vocab.get(key,''),key in trained])


if __name__ == '__main__':
    main()
