"""Write losslessly compressed white-model embeddings, one view at a time."""

import argparse
import fcntl
from pathlib import Path
import tempfile
import time

from numcodecs import Blosc
import numpy as np
import zarr

from .storage import committed_paths, read_json, read_vocabulary, write_json
from .vae_embeddings import embed


def write_scene(scene, destination, shape, lookup, weight):
    source = zarr.open_group(str(scene / "voxels.zarr"), mode="r")
    nodes, param2 = source["surface_node_id"], source["surface_param2"]
    if nodes.shape != shape or param2.shape != shape:
        raise ValueError(f"{scene.name}: expected white-model arrays with shape {shape}")

    view_chunk = (1, *shape[1:])
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    with tempfile.TemporaryDirectory(prefix=f".{scene.name}.partial-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        output = zarr.open_group(str(stage), mode="w")
        vectors = output.create_dataset(
            "embedding", shape=(*shape, weight.shape[1]), chunks=(*view_chunk, weight.shape[1]),
            dtype=np.float32, compressor=compressor, write_empty_chunks=True)
        mask = output.create_dataset(
            "valid", shape=shape, chunks=view_chunk, dtype=np.bool_,
            compressor=compressor, write_empty_chunks=True)
        for view in range(shape[0]):
            try:
                embedding, valid = embed(nodes[view], param2[view], lookup, weight)
            except ValueError as exc:
                raise ValueError(f"{scene.name}, view {view}: {exc}") from exc
            vectors[view] = embedding
            mask[view] = valid
            del embedding, valid
        stage.rename(destination)


def convert(dataset, table, output, max_scenes=None):
    dataset, output = dataset.resolve(), output.resolve()
    if output.is_relative_to(dataset):
        raise ValueError("Use an output directory outside the input dataset")
    config = read_json(dataset / "manifest.json")["config"]
    shape = tuple(config[key] for key in ("views_per_scene", "height", "width"))
    scenes = list(committed_paths(dataset).values())
    if not scenes or (max_scenes is None and len(scenes) != config["scenes"]):
        raise ValueError(f"Found {len(scenes)} of {config['scenes']} expected committed scenes")
    selected = scenes[:max_scenes]
    names = read_vocabulary(dataset)
    with np.load(table, allow_pickle=False) as saved:
        if saved["node_names"].tolist() != names:
            raise ValueError("Vox2Pix node ID order differs from the embedding lookup")
        lookup = saved["node_param2_to_canonical"]
        weight = saved["weight"]
        if (weight.ndim != 2 or weight.shape[1] != 32 or weight.dtype != np.float32
                or not np.isfinite(weight).all()):
            raise ValueError("Expected finite float32 VAE embedding rows with width 32")
        manifest = {
            "format": "vox2pix-white-embeddings-v1",
            "dataset": str(dataset),
            "expected_scenes": config["scenes"],
            "white_model_shape": list(shape),
            "embedding_dim": 32,
            "dtype": "float32",
            "checkpoint_step": int(saved["checkpoint_step"]),
            "checkpoint_sha256": str(saved["checkpoint_sha256"]),
            "vocabulary_sha256": str(saved["vocabulary_sha256"]),
        }
    if (lookup.shape != (len(names), 256) or lookup.dtype != np.int32
            or lookup.min() < -1 or lookup.max() >= len(weight)):
        raise ValueError("Invalid node/param2 to canonical ID lookup")

    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lock_path = output / ".writer.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another writer is using {output}") from exc
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError(f"Output already belongs to a different dataset or checkpoint: {output}")
        else:
            if any(path != lock_path for path in output.iterdir()):
                raise ValueError(f"Use a new output directory: {output}")
            write_json(manifest_path, manifest)

        scenes_dir = output / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        written = 0
        for i, scene in enumerate(selected, start=1):
            destination = scenes_dir / f"{scene.name}.zarr"
            if not destination.is_dir():
                write_scene(scene, destination, shape, lookup, weight)
                written += 1
            if i == 1 or i % 100 == 0 or i == len(selected):
                print(f"{i:,}/{len(selected):,} scenes, {written:,} written, "
                      f"{i - written:,} already complete, {time.monotonic() - started:.1f}s",
                      flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int, help="Limit conversion to the first N scenes")
    args = parser.parse_args(argv)
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be positive")
    convert(args.dataset, args.table, args.output, args.max_scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
