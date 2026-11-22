"""Scene-level lossless storage. Only renamed scene directories are committed."""

import fcntl
import json
from pathlib import Path
import re
import shutil
import tempfile

from numcodecs import Blosc
import numpy as np
from PIL import Image
import zarr

from .protocol import VERSION, plan

RAW_FIELDS = {"scene_node_id": np.dtype("<u2"), "scene_param2": np.dtype("u1")}
WHITE_FIELDS = {
    "white_node_id": np.dtype("<u2"),
    "white_param2": np.dtype("u1"),
    "white_face": np.dtype("u1"),
}
FIELDS = RAW_FIELDS | WHITE_FIELDS
EMPTY_NODE = 65535
EMPTY_FACE = 255
SCENE_PATTERN = re.compile(r"scene_([0-9]{4})")


def write_json(path, payload):
    """Publish complete JSON without exposing partial content to readers."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_vocabulary(root):
    return validate_vocabulary(read_json(Path(root) / "node_vocabulary.json"))


def validate_vocabulary(names):
    if (not isinstance(names, list) or not 0 < len(names) <= EMPTY_NODE
            or any(not isinstance(name, str) or not name for name in names)
            or names != sorted(set(names)) or "air" not in names):
        raise ValueError("Node vocabulary must be a sorted, unique list of names including air")
    return names


def committed_paths(root):
    """Return scene ID to committed scene path. Dot-prefixed staging is invisible."""
    scenes = Path(root) / "scenes"
    if not scenes.is_dir():
        return {}
    result = {}
    for path in sorted(scenes.iterdir()):
        match = SCENE_PATTERN.fullmatch(path.name)
        if match is None and path.name.startswith(".scene_"):
            continue
        if match is None or not path.is_dir():
            raise ValueError(f"Unexpected scene directory entry: {path}")
        result[int(match[1])] = path
    return result


def scene_shapes(config):
    white = (config["views_per_scene"], 3, config["height"], config["width"])
    return {name: (48, 48, 48) for name in RAW_FIELDS} | dict.fromkeys(WHITE_FIELDS, white)


def array_schema(arrays, config):
    if set(arrays) != set(FIELDS):
        raise ValueError(f"Expected exactly these array fields: {', '.join(FIELDS)}")
    for name, shape in scene_shapes(config).items():
        if arrays[name].shape != shape or arrays[name].dtype != FIELDS[name]:
            raise ValueError(f"{name} must have shape {shape} and dtype {FIELDS[name]}")


def read_arrays(scene, config):
    group = zarr.open_group(str(Path(scene) / "voxels.zarr"), mode="r")
    if set(group.group_keys()):
        raise ValueError("Unexpected nested Zarr group")
    arrays = {name: group[name] for name in group.array_keys()}
    array_schema(arrays, config)
    for name, array in arrays.items():
        if array.nchunks_initialized != array.nchunks:
            raise ValueError(f"Missing stored chunks for {name}")
        compressor = array.compressor
        if not isinstance(compressor, Blosc) or compressor.cname != "zstd":
            raise ValueError(f"{name} must use lossless Blosc/Zstd")
    return {name: array[:] for name, array in arrays.items()}


def build_provenance(meta):
    return tuple(meta.get(key) for key in ("engine_build", "game_build", "water_nodes", "glass_nodes"))


class DatasetWriter:
    """A single writer publishes independent scenes after complete local encoding."""

    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        lock_path = self.root / ".writer.lock"
        pending_manifest = self.root / ".manifest.json.tmp"
        if not manifest.exists() and any(path not in (lock_path, pending_manifest) for path in self.root.iterdir()):
            raise ValueError(f"Refusing to initialize nonempty directory without Vox2Pix manifest: {self.root}")
        self.lock = lock_path.open("a+b")
        try:
            try:
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"Vox2Pix collection is already writing {self.root}") from exc
            expected = {"version": VERSION, "config": self.config, "axes": "ENU",
                        "node_vocabulary": "node_vocabulary.json", "scene_storage": "scene-directories"}
            if manifest.exists():
                if read_json(manifest) != expected:
                    raise ValueError(f"Incompatible existing dataset configuration at {self.root}")
            else:
                if any(path not in (lock_path, pending_manifest) for path in self.root.iterdir()):
                    raise ValueError(f"Refusing to initialize nonempty directory without Vox2Pix manifest: {self.root}")
                write_json(manifest, expected)
            self.scenes_dir = self.root / "scenes"
            self.scenes_dir.mkdir(exist_ok=True)
            self.slots = {slot["scene_id"]: slot for slot in plan(self.config["seed"])}
            self.paths = committed_paths(self.root)
            if not set(self.paths) <= self.slots.keys():
                raise ValueError("Existing scene IDs do not belong to the pilot plan")
            self.vocabulary = (read_vocabulary(self.root)
                               if (self.root / "node_vocabulary.json").exists() else None)
            if self.paths and self.vocabulary is None:
                raise ValueError("Committed scenes are missing their node vocabulary")
            self.seeds = set()
            self.provenance = None
            for scene_id, path in self.paths.items():
                meta = read_json(path / "scene.json")
                self.verify_slot(meta, scene_id)
                provenance = build_provenance(meta)
                if self.provenance is None:
                    self.provenance = provenance
                if provenance != self.provenance:
                    raise ValueError("Existing scenes have inconsistent engine/game build or node classification")
                self.seeds.add(meta["world_seed"])
            for stage in self.scenes_dir.glob(".scene_*.partial-*"):
                shutil.rmtree(stage)
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *exc):
        self.lock.close()

    @property
    def completed(self):
        return set(self.paths)

    def verify_slot(self, meta, scene_id):
        slot = self.slots.get(scene_id)
        if (not isinstance(meta, dict) or slot is None
                or any(meta.get(key) != value for key, value in slot.items())):
            raise ValueError(f"Scene {scene_id} does not match its scheduled slot")
        seed = meta.get("world_seed")
        if type(seed) is not int or seed in self.seeds:
            raise ValueError(f"Invalid or duplicate world seed for scene {scene_id}")

    def commit(self, meta, vocabulary, arrays, rgbs):
        scene_id = meta.get("scene_id")
        self.verify_slot(meta, scene_id)
        if scene_id in self.paths:
            raise ValueError(f"Scene {scene_id} is already committed")
        validate_vocabulary(vocabulary)
        if self.vocabulary is not None and self.vocabulary != vocabulary:
            raise ValueError("Native node vocabulary changed between worlds")
        provenance = build_provenance(meta)
        if self.provenance is not None and self.provenance != provenance:
            raise ValueError("Engine/game build or registered node classification changed")
        array_schema(arrays, self.config)
        if len(rgbs) != self.config["views_per_scene"]:
            raise ValueError("Incorrect number of RGB views")
        from .qa import validate_payload

        validate_payload(meta, self.config, vocabulary, self.slots[scene_id], arrays)
        name = f"scene_{scene_id:04d}"
        final = self.scenes_dir / name
        with tempfile.TemporaryDirectory(prefix=f".{name}.partial-", dir=self.scenes_dir) as temporary:
            stage = Path(temporary)
            group = zarr.open_group(str(stage / "voxels.zarr"), mode="w")
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
            for key, array in arrays.items():
                chunks = array.shape if key in RAW_FIELDS else (1, 1, min(128, array.shape[-2]),
                                                                  min(128, array.shape[-1]))
                group.create_dataset(key, data=array, chunks=chunks, compressor=compressor,
                                     write_empty_chunks=True)
            rgb_path = stage / "rgb"
            rgb_path.mkdir()
            for index, rgb in enumerate(rgbs):
                if (rgb.shape != (self.config["height"], self.config["width"], 3)
                        or rgb.dtype != np.uint8):
                    raise ValueError(f"RGB view {index} must be H×W×3 uint8")
                Image.fromarray(rgb).save(rgb_path / f"{index:03d}.png", format="PNG")
            write_json(stage / "scene.json", meta)
            if self.vocabulary is None:
                write_json(self.root / "node_vocabulary.json", vocabulary)
                self.vocabulary = vocabulary
            stage.rename(final)
        self.paths[scene_id] = final
        self.seeds.add(meta["world_seed"])
        self.provenance = provenance
        return final
