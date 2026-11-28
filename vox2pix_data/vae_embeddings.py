"""Use the trained VAE's shared 32D embedding with raw Vox2Pix node states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


BACKGROUND = 65535


def build_lookup(names, canonical, train_ids):
    if len(names) != len(set(names)) or len(names) > BACKGROUND:
        raise ValueError("Invalid Vox2Pix node vocabulary")
    name_ids = {name: i for i, name in enumerate(names)}
    lookup = np.full((len(names), 256), -1, dtype=np.int32)
    for token in train_ids:
        name, param2 = canonical[token]
        if name in name_ids:
            lookup[name_ids[name], param2] = token
    return lookup


def canonical_ids(nodes, param2, lookup):
    """Keep BACKGROUND separate; raise on any unknown real node state."""
    if nodes.shape != param2.shape or nodes.dtype != np.uint16 or param2.dtype != np.uint8:
        raise ValueError("Expected equal-shaped uint16 node IDs and uint8 raw param2")
    real = nodes != BACKGROUND
    if np.any(nodes[real] >= len(lookup)) or np.any(param2[~real] != 0):
        raise ValueError("Invalid Vox2Pix node ID or BACKGROUND state")
    resolved = lookup[nodes[real], param2[real]]
    if np.any(resolved < 0):
        missing_index = np.flatnonzero(real)[np.flatnonzero(resolved < 0)[0]]
        coordinate = np.unravel_index(missing_index, nodes.shape)
        raise ValueError(f"VAE training lacks node ID {nodes[coordinate]} "
                         f"and param2 {param2[coordinate]} at {coordinate}")
    ids = np.full(nodes.shape, -1, dtype=np.int32)
    ids[real] = resolved
    return ids


def embed(nodes, param2, lookup, weight):
    """Return channels-last embeddings and the real-node mask for one tensor."""
    ids = canonical_ids(nodes, param2, lookup)
    valid = ids >= 0
    result = np.zeros((*ids.shape, weight.shape[1]), dtype=weight.dtype)
    result[valid] = weight[ids[valid]]
    return result, valid


def _write_archive(output, **values):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".vox2pix_embeddings_", dir=output.parent) as temporary:
        stage = Path(temporary) / "values.npz"
        np.savez_compressed(stage, **values)
        os.link(stage, output)


def export(dataset, prepared, checkpoint, output):
    import torch

    from voxel_vae.curated_data import read_manifest
    from voxel_vae.curated_vae import PHASE
    from voxel_vae.vocabulary import sha256_file

    manifest, vocabulary = read_manifest(prepared)
    # Read checkpoints created by this project's training command.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    step = saved.get("global_step", 0)
    if (saved.get("phase") != PHASE or step <= 0
            or saved.get("early_stopping", {}).get("best_step") != step
            or saved.get("vocabulary_K") != vocabulary.K
            or saved.get("canonical_vocabulary") != vocabulary.semantic_payload()
            or saved.get("vocabulary_sha256") != vocabulary.sha256
            or saved.get("config", {}).get("prepared_data", {}).get("manifest_sha256")
            != sha256_file(prepared / "manifest.json")):
        raise ValueError("Expected the best checkpoint trained on the supplied prepared chunks")
    weight = saved["model"]["embedding.weight"].detach().float().numpy()
    if weight.shape != (vocabulary.K, 32) or not np.isfinite(weight).all():
        raise ValueError("Expected finite VAE embedding rows with width 32")
    names = json.loads((dataset / "node_vocabulary.json").read_text(encoding="utf-8"))
    lookup = build_lookup(names, vocabulary.pairs, manifest["train_token_ids"])
    _write_archive(output, node_names=np.asarray(names), node_param2_to_canonical=lookup,
                   weight=weight, checkpoint_step=np.asarray(step),
                   checkpoint_sha256=np.asarray(sha256_file(checkpoint)),
                   vocabulary_sha256=np.asarray(vocabulary.sha256))
    print(f"{output}: {weight.shape[0]} shared rows, {np.count_nonzero(lookup >= 0)} mapped node states")


def read_scene(dataset, scene, table, view):
    """Read the shared scene map and one white-model view using a saved lookup."""
    import zarr

    with np.load(table, allow_pickle=False) as saved:
        names = json.loads((dataset / "node_vocabulary.json").read_text(encoding="utf-8"))
        if saved["node_names"].tolist() != names:
            raise ValueError("Vox2Pix node ID order differs from the embedding lookup")
        lookup, weight = saved["node_param2_to_canonical"], saved["weight"]
    group = zarr.open_group(str(dataset / "scenes" / scene / "voxels.zarr"), mode="r")
    scene_tokens, scene_valid = embed(group["scene_node_id"][:], group["scene_param2"][:],
                                     lookup, weight)
    if not scene_valid.all():
        raise ValueError("BACKGROUND is only valid in the white model")
    if not 0 <= view < group["surface_node_id"].shape[0]:
        raise ValueError("View is out of range")
    surface, valid = embed(group["surface_node_id"][view], group["surface_param2"][view],
                           lookup, weight)
    return {"scene_embedding": scene_tokens, "surface_embedding": surface, "surface_valid": valid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("export", help="Export an embedding lookup after curated VAE training")
    build.add_argument("--prepared", required=True, type=Path)
    build.add_argument("--checkpoint", required=True, type=Path)
    sample = commands.add_parser("sample", help="Convert one saved Vox2Pix scene and white-model view")
    sample.add_argument("--scene", required=True)
    sample.add_argument("--view", type=int, default=0)
    sample.add_argument("--table", required=True, type=Path)
    for command in (build, sample):
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.dataset, args.prepared, args.checkpoint, args.output)
    else:
        _write_archive(args.output, **read_scene(args.dataset, args.scene, args.table, args.view))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
