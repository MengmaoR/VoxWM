"""Read-only pilot validation and measured compression/coverage reporting."""

from collections import Counter
from functools import lru_cache
from itertools import product
import math
from pathlib import Path

import numpy as np
from PIL import Image
from zarr.errors import GroupNotFoundError

from vae_data.sharded.protocol import biome_group

from .protocol import (
    BIOMES, HEIGHTS, NOMINAL_CAMERA_HEIGHT, NOMINAL_CAMERA_XYZ, PLAYER_AIR_OFFSETS,
    SETTINGS, VERSION, VISIBILITY_LOOKUP_VERSION, VISIBILITY_RADIUS, WORLD_BOTTOM,
    WORLD_TOP, RejectedScene, camera_angles, camera_local_xyz, direction_for_angles,
    plan, region_id, require_air_camera, scene_visibility, visibility_lookup,
)
from .storage import (
    EMPTY_FACE, EMPTY_NODE, FIELDS, RAW_FIELDS, WHITE_FIELDS,
    array_schema, build_provenance, committed_paths, read_arrays, read_json, read_vocabulary, write_json,
)

PILOT = {"scenes": 200, "views_per_scene": 5, "width": 640, "height": 360}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def vector(meta, key, integer=False):
    value = np.asarray(meta.get(key))
    require(value.shape == (3,) and value.dtype.kind in ("iu" if integer else "iuf")
            and np.isfinite(value).all(), f"Invalid {key}")
    return value.astype(np.int64 if integer else np.float64)


def number(meta, key):
    value = meta.get(key)
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {key}")
    return value


def same_angle(left, right):
    return abs((left - right + 180) % 360 - 180) <= 0.05


def named_ids(meta, key, vocabulary):
    names = meta.get(key)
    require(isinstance(names, list) and all(isinstance(name, str) for name in names)
            and len(set(names)) == len(names) and set(names) <= set(vocabulary),
            f"Invalid native {key}")
    present = set(names)
    return [ident for ident, name in enumerate(vocabulary) if name in present]


@lru_cache(maxsize=1)
def validate_visibility_lookup():
    bounds = (range(math.ceil(coordinate - VISIBILITY_RADIUS),
                    math.floor(coordinate + VISIBILITY_RADIUS) + 1)
              for coordinate in NOMINAL_CAMERA_XYZ)
    expected = {(x, z, y) for x, y, z in product(*bounds)
                if math.dist((x, y, z), NOMINAL_CAMERA_XYZ) <= VISIBILITY_RADIUS + 1e-12}
    lookup = visibility_lookup()
    require(set(lookup) == expected, "Visibility lookup does not match the nominal radius and voxel centers")
    require(all(y >= 0 for _, _, y in lookup), "Visibility lookup includes supporting floor voxels")


def validate_camera_protocol(meta):
    require(number(meta, "nominal_camera_height") == NOMINAL_CAMERA_HEIGHT, "Incorrect nominal camera height")
    require(number(meta, "visibility_radius") == VISIBILITY_RADIUS, "Incorrect visibility radius")
    require(meta.get("visibility_lookup_version") == VISIBILITY_LOOKUP_VERSION, "Incorrect visibility lookup version")


def validate_view(view, config, foot, active, meta, nodes, definitions):
    ident = view.get("region_id")
    require(type(ident) is int and 0 <= ident < 40 and active[ident], "View region is not active")
    requested_yaw = number(view, "requested_yaw_deg")
    requested_pitch = number(view, "requested_pitch_deg")
    require(region_id(requested_yaw, requested_pitch) == ident, "Requested angles fall outside view region")
    validate_camera_protocol(view)
    require(np.array_equal(vector(view, "foot_anchor_world_xyz", integer=True), foot),
            "View foot anchor differs from the scene anchor")
    camera = vector(view, "actual_camera_world_xyz")
    try:
        local_xyz = camera_local_xyz(camera, foot)
        require_air_camera(nodes, definitions, local_xyz)
    except RejectedScene as exc:
        raise ValueError(str(exc)) from exc
    require(math.isclose(number(view, "actual_camera_height_above_foot"), local_xyz[1], rel_tol=0, abs_tol=1e-10),
            "Recorded camera height differs from the captured position")
    require(np.array_equal(vector(view, "projection_origin_local_xyz"), local_xyz),
            "White Model origin differs from the captured RGB camera position")
    direction = vector(view, "cam_dir_xyz")
    norm = np.linalg.norm(direction)
    require(math.isfinite(norm) and norm > 0, "Invalid camera direction")
    direction /= norm
    actual_yaw, actual_pitch = camera_angles(direction)
    yaw, pitch = number(view, "yaw_deg"), number(view, "pitch_deg")
    require(same_angle(yaw, actual_yaw) and abs(pitch - actual_pitch) <= .05
            and direction @ direction_for_angles(requested_yaw, requested_pitch) >= math.cos(math.radians(.04))
            and region_id(yaw, pitch) == ident, "Inconsistent actual camera angles")
    fov_x, fov_y = number(view, "fov_x_rad"), number(view, "fov_y_rad")
    width, height = config["width"], config["height"]
    require(0 < fov_x < math.pi and 0 < fov_y < math.pi
            and math.isclose(math.tan(fov_x / 2) / math.tan(fov_y / 2), width / height, rel_tol=1e-4),
            "FOV and native viewport aspect do not agree")
    require(view.get("resolution") == [width, height], "Incorrect native RGB resolution")
    for key in ("world_time", "weather", "gamma"):
        require(view.get(key) == meta[key], f"Changed {key} during camera capture")
    tick = view.get("simulation_tick")
    require(type(tick) is int and tick >= 0, "Invalid native simulation tick")
    if meta.get("data_source") == "native":
        require(type(view.get("render_simulation_tick")) is int and view["render_simulation_tick"] == tick - 1,
                "Native RGB render tick was not bracketed by consecutive voxel snapshots")
    return ident, tick


def validate_payload(meta, config, vocabulary, slot, arrays):
    """Validate decoded arrays and metadata used to write or inspect one scene."""
    require(isinstance(meta, dict), "Malformed scene metadata")
    array_schema(arrays, config)
    require(all(meta.get(key) == value for key, value in slot.items()), "Scene does not match its scheduled slot")
    require(type(meta.get("world_seed")) is int, "Invalid world seed")
    biome = meta.get("native_biome")
    require(isinstance(biome, str) and biome_group(biome, BIOMES) == slot["biome"],
            "Native surface biome differs from the planned biome")
    foot = vector(meta, "foot_anchor_xyz", integer=True)
    validate_camera_protocol(meta)
    validate_visibility_lookup()
    origin = vector(meta, "region_origin_xyz", integer=True)
    origin_enu = vector(meta, "region_origin_enu", integer=True)
    require(np.array_equal(origin, foot - 24) and np.array_equal(origin_enu, origin[[0, 2, 1]])
            and WORLD_BOTTOM <= origin[1] <= WORLD_TOP - 47, "Invalid foot-centered coordinates")
    surface = meta.get("surface_y")
    require(type(surface) is int and WORLD_BOTTOM <= surface <= WORLD_TOP, "Invalid native surface reference")
    foot_y = int(foot[1])
    anchor_kind = meta.get("anchor_kind")
    height = slot["height"]
    if height == "surface":
        require(anchor_kind in ("surface_floor", "surface_water"), "Invalid surface anchor kind")
        if anchor_kind == "surface_floor":
            require(foot_y >= surface + 1, "Surface floor is below the native surface")
    else:
        lo, hi = ((surface - 24, surface - 8) if height == "shallow" else (WORLD_BOTTOM + 24, surface - 32))
        require(anchor_kind == "cave_floor" and lo <= foot_y <= hi, "Foot anchor is outside its cave height band")

    for key in ("engine_build", "game_build"):
        require(isinstance(meta.get(key), str) and meta[key].strip(), f"Missing {key}")
    require(math.isclose(number(meta, "world_time"), .5, abs_tol=1e-5) and meta.get("weather") == "none"
            and number(meta, "gamma") == SETTINGS["display_gamma"], "Invalid fixed renderer state")
    settings = meta.get("render_settings")
    require(isinstance(settings, dict) and all(settings.get(key) == value for key, value in SETTINGS.items()),
            "Incomplete or incompatible renderer settings")
    geometry = meta.get("geometry_counts")
    require(isinstance(geometry, dict) and all(isinstance(key, str) and type(value) is int and value >= 0
                                               for key, value in geometry.items()),
            "Invalid geometry counts")

    nodes, param2 = arrays["scene_node_id"], arrays["scene_param2"]
    require(nodes.max() < len(vocabulary), "Scene node ID is outside the vocabulary")
    water = named_ids(meta, "water_nodes", vocabulary)
    named_ids(meta, "glass_nodes", vocabulary)
    air_id = vocabulary.index("air")
    require(air_id not in water and (vocabulary.index("ignore") not in water if "ignore" in vocabulary else True),
            "Native water classification includes Air/Ignore")
    definitions = {ident: {"name": name, "water": False} for ident, name in enumerate(vocabulary)}
    for ident in water:
        definitions[ident]["water"] = True
    try:
        center, full, active = scene_visibility(nodes, definitions)
    except RejectedScene as exc:
        raise ValueError(str(exc)) from exc
    require(math.isclose(number(meta, "center_free_ratio"), center, abs_tol=1e-10)
            and math.isclose(number(meta, "full_free_ratio"), full, abs_tol=1e-10), "Inconsistent free-space ratios")
    recorded_active = meta.get("active_view_regions")
    require(isinstance(recorded_active, list) and all(type(value) is bool for value in recorded_active)
            and recorded_active == active, "Invalid or inconsistent visibility mask")

    center_index = nodes.shape[0] // 2
    support = nodes[center_index, center_index, center_index - 1]
    body = nodes[center_index, center_index, center_index + np.asarray(PLAYER_AIR_OFFSETS)]
    require((body == air_id).all(), "Foot anchor lacks player body clearance")
    if anchor_kind == "surface_water":
        require(support in water, "Virtual water anchor lacks water immediately below the foot")
    else:
        require(support not in [air_id, *water], "Floor anchor lacks support")
    views = meta.get("views")
    require(isinstance(views, list) and len(views) == config["views_per_scene"], "Missing scene views")
    require(all(isinstance(view, dict) for view in views), "Invalid view metadata")
    ids, ticks = zip(*(validate_view(view, config, foot, active, meta, nodes, definitions) for view in views))
    require(len(set(ids)) == len(ids), "Repeated view region")
    require(all(a < b for a, b in zip(ticks, ticks[1:])), "Native view ticks must increase")

    white, state, faces = arrays["white_node_id"], arrays["white_param2"], arrays["white_face"]
    valid = white != EMPTY_NODE
    require(not (valid[:, 1:] & ~valid[:, :-1]).any(), "Nonempty White Model layer follows an empty layer")
    require((state[~valid] == 0).all() and (faces[~valid] == EMPTY_FACE).all(), "Incorrect empty layer sentinel")
    hits = white[valid]
    require((faces[valid] < 6).all() and (hits < len(vocabulary)).all() and (hits != air_id).all(),
            "Invalid face, node ID, or Air in White Model hits")
    present = np.zeros((len(vocabulary), 256), dtype=bool)
    present[nodes, param2] = True
    require(present[hits, state[valid]].all(), "White Model node/state is absent from its raw voxel map")


def _validate_scene(path, config, vocabulary, slot):
    path = Path(path)
    meta = read_json(path / "scene.json")
    arrays = read_arrays(path, config)
    validate_payload(meta, config, vocabulary, slot, arrays)
    expected = [path / "rgb" / f"{index:03d}.png" for index in range(config["views_per_scene"])]
    require(sorted((path / "rgb").iterdir()) == expected, "Missing or extra scene RGB views")
    for rgb in expected:
        with Image.open(rgb) as png:
            require(png.format == "PNG" and png.mode == "RGB" and png.size == (config["width"], config["height"]),
                    f"Invalid RGB view {rgb.name}")
            png.load()
    return meta, arrays, expected


def validate_scene(path, config, vocabulary, slot):
    """Read and validate one committed scene; return its scene metadata."""
    return _validate_scene(path, config, vocabulary, slot)[0]


def _add_count(counter, values):
    unique, counts = np.unique(values, return_counts=True)
    counter.update({int(value): int(count) for value, count in zip(unique, counts, strict=True)})


class Coverage:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.biomes, self.heights, self.biome_heights = Counter(), Counter(), Counter()
        self.active, self.selected, self.closed = Counter(), Counter(), Counter()
        self.pairs, self.faces = Counter(), Counter()
        self.layer_valid, self.layer_water, self.layer_glass = Counter(), Counter(), Counter()
        self.geometry = Counter()
        self.camera_heights = []
        self.scenes = self.frames = self.pixels = 0

    def add(self, meta, arrays):
        white, state, faces = (arrays[key] for key in WHITE_FIELDS)
        self.scenes += 1
        self.frames += white.shape[0]
        self.pixels += white.shape[0] * white.shape[-2] * white.shape[-1]
        biome, height = meta["biome"], meta["height"]
        self.biomes[biome] += 1
        self.heights[height] += 1
        self.biome_heights[biome, height] += 1
        active = sum(meta["active_view_regions"])
        self.active[active] += 1
        self.closed[40 - active] += 1
        self.selected.update(view["region_id"] for view in meta["views"])
        self.camera_heights.extend(view["actual_camera_height_above_foot"] for view in meta["views"])
        self.geometry.update(meta["geometry_counts"])
        water = named_ids(meta, "water_nodes", self.vocabulary)
        glass = named_ids(meta, "glass_nodes", self.vocabulary)
        for layer in range(3):
            ids = white[:, layer]
            valid = ids != EMPTY_NODE
            hits = ids[valid]
            self.layer_valid[layer] += hits.size
            self.layer_water[layer] += int(np.isin(hits, water).sum())
            self.layer_glass[layer] += int(np.isin(hits, glass).sum())
            _add_count(self.pairs, hits.astype(np.uint32) * 256 + state[:, layer][valid])
            _add_count(self.faces, faces[:, layer][valid])

    def report(self):
        nodes = Counter()
        for pair, count in self.pairs.items():
            nodes[pair // 256] += count
        return {
            "scene_count": self.scenes,
            "frame_count": self.frames,
            "actual_camera_height": {
                "min": float(np.min(self.camera_heights)), "mean": float(np.mean(self.camera_heights)),
                "max": float(np.max(self.camera_heights)), "std": float(np.std(self.camera_heights)),
            },
            "biome_scene_counts": {biome: self.biomes[biome] for biome in sorted(BIOMES)},
            "height_scene_counts": {height: self.heights[height] for height in HEIGHTS},
            "biome_height_scene_counts": {
                biome: {height: self.biome_heights[biome, height] for height in HEIGHTS}
                for biome in sorted(BIOMES)
            },
            "active_view_region_histogram": dict(sorted(self.active.items())),
            "closed_view_region_histogram": dict(sorted(self.closed.items())),
            "selected_view_region_histogram": {region: self.selected[region] for region in range(40)},
            "node_visibility_frequency": {self.vocabulary[ident]: count for ident, count in nodes.most_common()},
            "node_param2_visibility_frequency": [
                {"node_id": pair // 256, "node_name": self.vocabulary[pair // 256],
                 "raw_param2": pair % 256, "hits": count} for pair, count in self.pairs.most_common()
            ],
            "face_dir_frequency": {direction: self.faces[direction] for direction in range(6)},
            "geometry_counts": dict(self.geometry),
            "layers": [
                {"layer": layer + 1, "pixels": self.pixels,
                 "valid_hits": self.layer_valid[layer], "valid_ratio": self.layer_valid[layer] / self.pixels,
                 "water_hits": self.layer_water[layer], "water_ratio": self.layer_water[layer] / self.pixels,
                 "glass_hits": self.layer_glass[layer], "glass_ratio": self.layer_glass[layer] / self.pixels}
                for layer in range(3)
            ],
        }


def _scan(root, visit=None):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    require(isinstance(manifest, dict), "Malformed Vox2Pix dataset manifest")
    config = manifest.get("config", {})
    require(isinstance(config, dict), "Malformed Vox2Pix dataset configuration")
    require(manifest.get("version") == VERSION and manifest.get("axes") == "ENU"
            and manifest.get("node_vocabulary") == "node_vocabulary.json"
            and manifest.get("scene_storage") == "scene-directories",
            "Unsupported Vox2Pix dataset manifest")
    require(all(type(config.get(key)) is int and config[key] == value for key, value in PILOT.items())
            and type(config.get("seed")) is int and config["seed"] >= 0 and config.get("renderer") == "software",
            "Expected the 200-scene, 1,000-frame software-rendered 640×360 pilot")
    slots = {slot["scene_id"]: slot for slot in plan(config["seed"])}
    paths = committed_paths(root)
    require(paths.keys() == slots.keys(), f"Incomplete pilot: {len(paths)}/{len(slots)} committed scenes")
    vocabulary = read_vocabulary(root)
    coverage = Coverage(vocabulary)
    seeds = set()
    builds = None
    for scene_id, path in paths.items():
        try:
            meta, arrays, rgbs = _validate_scene(path, config, vocabulary, slots[scene_id])
            require(meta.get("data_source") == "native", "Expected native provenance for the complete pilot")
            require(meta["world_seed"] not in seeds, "Repeated world seed")
            seeds.add(meta["world_seed"])
            provenance = build_provenance(meta)
            if builds is None:
                builds = provenance
            require(provenance == builds, "Engine/game build or registered node classification changed")
            coverage.add(meta, arrays)
            if visit is not None:
                visit(path, arrays, rgbs)
        except (ValueError, OSError, KeyError, TypeError, GroupNotFoundError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    return coverage.report()


def check(root):
    """Validate complete Pilot and return coverage without writing files."""
    return _scan(root)


def _sizes(raw, actual):
    return {"raw_bytes": raw, "actual_bytes": actual,
            "compression_ratio": raw / actual if actual else None}


def compression_report(root):
    root = Path(root)
    png_sizes = []
    byte_counts = Counter()
    raw_counts = Counter()
    metadata_bytes = sum((root / key).stat().st_size for key in ("manifest.json", "node_vocabulary.json"))

    def measure(path, arrays, rgbs):
        nonlocal metadata_bytes
        png_sizes.extend(rgb.stat().st_size for rgb in rgbs)
        metadata_bytes += (path / "scene.json").stat().st_size
        zarr_dir = path / "voxels.zarr"
        for file in zarr_dir.rglob("*"):
            if not file.is_file():
                continue
            name = file.parent.name
            if name in FIELDS and not file.name.startswith("."):
                byte_counts[name] += file.stat().st_size
            else:
                metadata_bytes += file.stat().st_size
        raw_counts.update({name: array.nbytes for name, array in arrays.items()})

    coverage = _scan(root, measure)
    rgb = _sizes(PILOT["width"] * PILOT["height"] * 3 * len(png_sizes), sum(png_sizes))
    rgb.update(mean_bytes_per_frame=float(np.mean(png_sizes)), median_bytes_per_frame=float(np.median(png_sizes)),
               p95_bytes_per_frame=float(np.percentile(png_sizes, 95)))

    def fields(names):
        rows = {name: _sizes(raw_counts[name], byte_counts[name]) for name in names}
        rows["total"] = _sizes(sum(raw_counts[name] for name in names), sum(byte_counts[name] for name in names))
        return rows

    total = sum(png_sizes) + sum(byte_counts.values()) + metadata_bytes
    projected = total * 50
    result = dict(
        rgb=rgb,
        white_model=fields(WHITE_FIELDS),
        raw_voxel_maps=fields(RAW_FIELDS),
        metadata={"actual_bytes": metadata_bytes},
        pilot_total_bytes=total, pilot_total_GB=total / 1e9, pilot_total_GiB=total / 2**30,
        projected_50k_total_bytes=projected, projected_50k_GB=projected / 1e9, projected_50k_GiB=projected / 2**30,
        gate_decimal_GB=100, recommended_resolution="640x360" if projected <= 100_000_000_000 else "320x180",
        actual_camera_height=coverage.pop("actual_camera_height"),
        coverage=coverage,
    )
    write_json(root / "compression_report.json", result)
    markdown = root / ".compression_report.md.tmp"
    markdown.write_text(_markdown(result))
    markdown.replace(root / "compression_report.md")
    return result


def _markdown(report):
    coverage = report["coverage"]
    lines = [
        "# Vox2Pix native pilot compression", "",
        f"Measured **{coverage['scene_count']} scenes and {coverage['frame_count']} RGB frames**.", "",
        "| Payload | Raw bytes | Stored bytes | Raw / stored |",
        "| --- | ---: | ---: | ---: |",
    ]
    fields = {"RGB PNG": report["rgb"]}
    for group in ("white_model", "raw_voxel_maps"):
        for name, values in report[group].items():
            fields[f"{group} total" if name == "total" else name] = values
    for name, values in fields.items():
        ratio = values["compression_ratio"]
        lines.append(f"| {name} | {values['raw_bytes']:,} | {values['actual_bytes']:,} | {ratio:.2f}× |")
    rgb = report["rgb"]
    eye = report["actual_camera_height"]
    lines.extend([
        f"| Metadata | — | {report['metadata']['actual_bytes']:,} | — |", "",
        f"PNG bytes per frame averaged {rgb['mean_bytes_per_frame']:,.0f}, with a median of "
        f"{rgb['median_bytes_per_frame']:,.0f} and p95 of {rgb['p95_bytes_per_frame']:,.0f}.", "",
        f"Pilot size is **{report['pilot_total_GB']:.3f} GB / {report['pilot_total_GiB']:.3f} GiB**.",
        f"The 50× projection is **{report['projected_50k_GB']:.3f} GB / "
        f"{report['projected_50k_GiB']:.3f} GiB**.",
        f"The decimal 100 GB gate recommends **{report['recommended_resolution']}**.", "",
        f"Measured camera height above the foot across all views is {eye['mean']:.6f} voxels on average, "
        f"with min {eye['min']:.6f}, max {eye['max']:.6f}, and standard deviation {eye['std']:.6f}.", "",
        "## Coverage", "",
        "| Layer | Valid pixels | Valid ratio | Water hits | Glass hits |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for layer in coverage["layers"]:
        lines.append(f"| {layer['layer']} | {layer['valid_hits']:,} | {layer['valid_ratio']:.2%} | "
                     f"{layer['water_hits']:,} | {layer['glass_hits']:,} |")
    lines.extend(["", "| Biome | Surface | Shallow | Deep |", "| --- | ---: | ---: | ---: |"])
    for biome, counts in coverage["biome_height_scene_counts"].items():
        lines.append(f"| {biome} | {counts['surface']} | {counts['shallow']} | {counts['deep']} |")
    lines.extend(["", "Full view-region histograms and node, raw param2, and face frequencies are in "
                      "`compression_report.json`.",
                  "Validation covers stored inputs and camera metadata. Native RGB-to-White-Model alignment "
                  "still requires visual verification.", ""])
    return "\n".join(lines)
