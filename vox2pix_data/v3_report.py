"""Validate v3 native scenes and export a standalone image comparison report."""

from collections import Counter
from hashlib import sha256
from html import escape
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .protocol import (
    NOMINAL_CAMERA_XYZ, PLAYER_AIR_OFFSETS, RejectedScene, V3_MATERIAL_VERSION, V3_RGB,
    require_air_camera, v3_lighting_images, v3_plan, v3_rgb, v3_version,
)
from .qa import (
    named_ids, number, require, validate_camera_protocol, validate_view,
    validate_visibility_lookup, vector,
)
from .storage import (
    EMPTY_FACE, EMPTY_NODE, array_schema, build_provenance, committed_paths, read_arrays,
    read_json, read_vocabulary, write_json,
)


CAVE_DARK_FRACTION = .5
FACE_COLORS = np.array([(223, 103, 104), (130, 48, 58), (111, 195, 121),
                        (49, 110, 71), (111, 162, 233), (63, 78, 167)], dtype="u1")
BACKGROUND = (17, 23, 39)


def brightness(rgb):
    """Display-encoded RGB8 Rec. 709 luma, measured directly on native PNGs."""
    luma = rgb @ np.array([.2126, .7152, .0722])
    return {"mean_luma": float(luma.mean()), "median_luma": float(np.median(luma)),
            "near_black_fraction": float((luma < 8).mean()),
            "near_white_fraction": float((luma >= 247).mean())}


def _validate_renderer_lighting(view, floor):
    light = view.get("renderer_lighting")
    require(isinstance(light, dict) and light.get("version") == "voxwm-light-floor-v1",
            "Missing native renderer light LUT evidence")
    require(type(light.get("render_tick")) is int and light["render_tick"] == view["render_simulation_tick"],
            "Native lighting evidence belongs to a different RGB tick")
    require(math.isclose(number(light, "lighting_floor"), floor, rel_tol=0, abs_tol=1e-6)
            and number(light, "display_gamma") == V3_RGB["display_gamma"]
            and number(light, "lut_zero") == round(floor * 255)
            and number(light, "lut_max") == 255
            and number(light, "pending_meshes") == 0,
            "Native renderer light LUT or mesh state differs from this exposure")


def _validate_material_lighting(view, ambient):
    probe = view.get("material_renderer_lighting")
    light = view["renderer_lighting"]
    frame_keys = ("render_tick", "display_gamma", "lighting_floor", "lut_zero", "lut_max", "pending_meshes")
    require(number(view, "solid_ambient_floor") == ambient and isinstance(probe, dict)
            and probe.get("version") == "voxwm-material-ambient-v1" and probe.get("shader_bound") is True
            and number(probe, "solid_ambient_floor") == ambient
            and all(probe.get(key) == light[key] for key in frame_keys),
            "Material ambient proof differs from the native RGB frame")


def validate_lighting_pair(meta, frames, official_rgb, config=None):
    """Check the two native images for the selected rendering protocol."""
    config = config or {}
    rgb = v3_rgb(config)
    ambient = rgb.get("solid_ambient_floor")
    names = v3_lighting_images(config)
    require(set(frames) == set(names), "Lighting comparison images differ from the selected profile")
    check = meta.get("lighting_comparison")
    require(isinstance(check, dict), "Missing native Lighting Comparison")
    for key in ("voxel_identity_unchanged", "light_identity_unchanged", "white_identity_unchanged"):
        require(check.get(key) is True, f"Native Lighting Comparison failed {key}")
    index = check.get("view_index")
    require(type(index) is int and 0 <= index < len(meta["views"]), "Invalid lighting view index")
    main_view = meta["views"][index]
    for label, expected_floor in (("control", 0.), ("proposed", rgb["lighting_floor"])):
        row = check[label]
        require(number(row, "lighting_floor") == expected_floor,
                "Lighting Comparison floor differs from the requested setting")
        require(number(row, "display_gamma") == main_view["gamma"], "Display gamma changed in Lighting Comparison")
        _validate_renderer_lighting(row, expected_floor)
        if ambient is not None:
            _validate_material_lighting(row, 0. if label == "control" else ambient)
            if label == "proposed":
                require(row["material_renderer_lighting"] == main_view["material_renderer_lighting"],
                        "Lighting Comparison has different material evidence from the dataset view")
        for key in ("actual_camera_world_xyz", "cam_dir_xyz"):
            require(np.allclose(vector(row, key), vector(main_view, key), rtol=0, atol=1e-6),
                    f"Lighting Comparison changed {key}")
        for key in ("fov_x_rad", "fov_y_rad"):
            require(math.isclose(number(row, key), number(main_view, key), rel_tol=0, abs_tol=1e-6),
                    f"Lighting Comparison changed {key}")
    eye_light = [check[key].get("native_eye_param1") for key in ("control", "proposed")]
    if any(value is not None for value in eye_light):
        require(all(type(value) is int and 0 <= value <= 255 for value in eye_light)
                and eye_light[0] == eye_light[1], "Native eye-cell param1 differs across captures")
    require(np.array_equal(frames[names[1]], official_rgb[index]),
            "Proposed lighting PNG differs from its official dataset view")
    control, proposed = (brightness(frames[label]) for label in names)
    status = "VERIFIED"
    if meta["kind"] == "cave":
        if control["near_black_fraction"] < CAVE_DARK_FRACTION:
            status = "CONTROL_NOT_DARK"
        elif proposed["mean_luma"] < control["mean_luma"] + 5.:
            status = "NO_MEASURABLE_BRIGHTENING"
        elif proposed["near_black_fraction"] >= CAVE_DARK_FRACTION:
            status = "CAVE_REMAINS_DARK"
    return dict(check, status=status, control_brightness=control, proposed_brightness=proposed)


def validate_payload(meta, config, vocabulary, slot, arrays):
    from .geometry import is_water_liquid, water_contains
    from .v3_sampling import cave_evidence, scene_visibility, settled_anchor

    array_schema(arrays, config)
    require(all(meta.get(key) == value for key, value in slot.items()), "Scene does not match its scheduled slot")
    require(type(meta.get("world_seed")) is int, "Invalid world seed")
    for key in ("surface_biome", "foot_biome", "engine_build", "game_build"):
        require(isinstance(meta.get(key), str) and bool(meta[key]), f"Missing {key}")
    require(meta.get("native_biome") == meta["surface_biome"], "Expected native_biome to use the surface label")
    kind = slot["kind"]
    foot = vector(meta, "foot_anchor_xyz", integer=True)
    origin = vector(meta, "region_origin_xyz", integer=True)
    origin_enu = vector(meta, "region_origin_enu", integer=True)
    require(np.array_equal(origin, foot - 24) and np.array_equal(origin_enu, origin[[0, 2, 1]]),
            "Invalid foot-centered 48³ coordinates")
    if kind != "underwater":
        y_min, y_max = config["survey_y"]
        require(y_min <= origin[1] <= y_max - 47, "The 48³ crop lies outside the v3 survey range")
    require(math.isclose(number(meta, "world_time"), .5, abs_tol=1e-5) and meta.get("weather") == "none"
            and number(meta, "gamma") == 3., "Invalid native daylight, weather or gamma")
    settings = meta.get("native_render_settings")
    expected_rgb = v3_rgb(config)
    ambient = expected_rgb.get("solid_ambient_floor")
    require(isinstance(settings, dict) and all(settings.get(key) == value for key, value in expected_rgb.items()),
            "Native viewing range, fog, gamma or brightness floor differs from the v3 protocol")
    if ambient is not None:
        require(meta.get("lighting_profile") == config["lighting_profile"]
                and number(meta, "solid_ambient_floor") == ambient
                and number(meta, "lighting_floor") == expected_rgb["lighting_floor"],
                "Scene material ambient metadata differs from the selected profile")
    validate_camera_protocol(meta)
    validate_visibility_lookup()

    nodes, params = (arrays[key] for key in ("scene_node_id", "scene_param2"))
    require(nodes.max() < len(vocabulary), "Scene node ID is outside its vocabulary")
    water, terrain = (set(named_ids(meta, key, vocabulary)) for key in ("water_nodes", "terrain_nodes"))
    require(water.isdisjoint(terrain), "Natural terrain classification includes liquid")
    air = vocabulary.index("air")
    require(air not in water and air not in terrain, "Natural terrain or water classification includes Air")
    native_water = meta.get("water_definitions", {})
    require(isinstance(native_water, dict) and set(native_water) == {vocabulary[ident] for ident in water}
            and all(isinstance(definition, dict) and is_water_liquid(definition) for definition in native_water.values()),
            "Missing native liquid rules for water-volume reconstruction")
    definitions = {ident: dict(native_water.get(name, {}), name=name, water=ident in water,
                               walkable=ident in terrain, terrain=ident in terrain)
                   for ident, name in enumerate(vocabulary)}
    views = meta.get("views")
    require(isinstance(views, list) and len(views) == config["views_per_scene"], "Missing scene views")
    actual_foot = None
    water_surface = kind == "surface" and meta.get("anchor_kind") == "surface_water"
    if kind == "underwater":
        descent = meta.get("underwater_descent")
        require(isinstance(descent, dict), "Missing native underwater descent")
        actual_foot = vector(meta, "settled_player_xyz")
        start = vector(descent, "spawn_player_xyz")
        require(np.array_equal(actual_foot, vector(descent, "settled_player_xyz"))
                and np.array_equal(foot, settled_anchor(actual_foot))
                and type(descent.get("ticks")) is int and descent["ticks"] > 0
                and start[1] > actual_foot[1], "Underwater anchor must follow actual native descent")
        support = vector(descent, "support_xyz", integer=True) - origin
        require(((0 <= support) & (support < 48)).all()
                and int(nodes[support[0], support[2], support[1]]) in terrain
                and math.isclose(number(descent, "support_top_y"), actual_foot[1], abs_tol=.03),
                "Settled player lacks native floor support")
        view_origin = vector(meta, "visibility_camera_local_xyz")
    else:
        view_origin = NOMINAL_CAMERA_XYZ
        body = nodes[24, 24, 24 + np.asarray(PLAYER_AIR_OFFSETS)]
        require((body == air).all(), "Player body intersects native solid geometry")
        require(int(nodes[24, 24, 23]) in (water if water_surface else terrain),
                "Foot anchor lacks the expected native water or terrain below it")
    try:
        center, full, active = scene_visibility(nodes, definitions, kind, camera_origin=view_origin)
        if kind == "cave":
            require(meta.get("cave_evidence") == cave_evidence(nodes, definitions),
                    "Recorded cave enclosure differs from the decoded raw geometry")
    except RejectedScene as exc:
        raise ValueError(str(exc)) from exc
    require(math.isclose(number(meta, "center_free_ratio"), center, abs_tol=1e-10)
            and math.isclose(number(meta, "full_free_ratio"), full, abs_tol=1e-10), "Changed free-space metadata")
    require(meta.get("active_view_regions") == active, "Recorded visibility mask differs from decoded geometry")

    def require_medium(grid, rules, eye):
        if kind == "underwater":
            if not water_contains(grid, params, rules, eye):
                raise RejectedScene("Actual RGB camera is outside the native water volume")
        else:
            require_air_camera(grid, rules, eye)

    ids, ticks = zip(*(validate_view(view, config, foot, active, meta, nodes, definitions,
                                    camera_check=require_medium, actual_foot=actual_foot) for view in views))
    require(len(set(ids)) == len(ids) and all(a < b for a, b in zip(ticks, ticks[1:])),
            "View regions must be distinct and native ticks increasing")
    for view in views:
        require(view.get("camera_medium") == ("water" if kind == "underwater" else "air"),
                "Recorded view medium differs from expected scene kind")
        require(number(view, "lighting_floor") == expected_rgb["lighting_floor"], "Unexpected native brightness floor")
        _validate_renderer_lighting(view, expected_rgb["lighting_floor"])
        if ambient is not None:
            _validate_material_lighting(view, ambient)
        if expected_rgb.get("voxwm_exact_draw_range"):
            from .fog_runtime import validate_fog

            validate_fog(view.get("renderer_fog", {}), ratio=expected_rgb["fog_start"],
                         tick=view["render_simulation_tick"])
        if water_surface:
            require(np.allclose(vector(view, "actual_player_world_xyz"), foot, atol=.003, rtol=0),
                    "Surface observer moved from its water-surface anchor")
        if actual_foot is not None:
            require(np.allclose(vector(view, "actual_player_world_xyz"), actual_foot, atol=.003, rtol=0),
                    "Player moved after settling underwater")
            require(np.allclose(vector(view, "projection_origin_local_xyz"), view_origin, atol=.003, rtol=0),
                    "Underwater visibility was evaluated at a different eye position")

    surface, state, faces, counts = (arrays[key] for key in
                                    ("surface_node_id", "surface_param2", "surface_face_dir", "water_count"))
    hits = surface != EMPTY_NODE
    require((faces[~hits] == EMPTY_FACE).all() and (state[~hits] == 0).all(), "Invalid BACKGROUND sentinels")
    require((faces[hits] < 6).all() and (surface[hits] < len(vocabulary)).all()
            and not np.isin(surface[hits], (air, *water)).any(), "Invalid Surface face, node ID, Air or Water hit")
    present = np.zeros((len(vocabulary), 256), dtype=bool)
    present[nodes, params] = True
    require(present[surface[hits], state[hits]].all(), "Surface node/state is absent from its own raw voxel map")
    require(np.isin(nodes, list(water)).any() or not counts.any(), "Water Count is nonzero in a dry voxel map")
    if kind == "underwater":
        require(counts.any(axis=(1, 2)).all(), "A submerged view has no rays that exit any water voxel")


def _read_png(path, config):
    with Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (config["width"], config["height"]),
                f"Invalid native PNG: {path.name}")
        return np.asarray(image)


def _scan(root, allow_partial, visit=None):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    version = v3_version(config)
    require(manifest.get("version") == version and manifest.get("axes") == "ENU" and config.get("mode") == "smoke"
            and config.get("views_per_scene") == 5 and (config.get("width"), config.get("height")) == (640, 360),
            "Expected a native Vox2Pix v3 smoke dataset at 640×360 with five views per scene")
    slots = {slot["scene_id"]: slot for slot in v3_plan(config)}
    paths = committed_paths(root)
    missing = sorted(slots.keys() - paths.keys())
    require(paths.keys() <= slots.keys() and (allow_partial or not missing),
            f"Expected {len(slots)} committed scenes; found {len(paths)}")
    vocabulary = read_vocabulary(root) if paths else []
    kinds, comparisons, seeds, scene_rows, view_rows = Counter(), {}, set(), [], []
    provenance = None
    for scene_id, path in paths.items():
        try:
            meta = read_json(path / "scene.json")
            arrays = read_arrays(path, config)
            require(meta.get("data_source") == "native", "Smoke requires native provenance")
            require(meta["world_seed"] not in seeds, "Duplicate native world seed")
            builds = build_provenance(meta)
            require(provenance is None or provenance == builds, "Per-scene native build or vocabulary rules changed")
            provenance = builds
            seeds.add(meta["world_seed"])
            validate_payload(meta, config, vocabulary, slots[scene_id], arrays)
            rgb_paths = [path / "rgb" / f"{index:03d}.png" for index in range(config["views_per_scene"])]
            require(sorted((path / "rgb").iterdir()) == rgb_paths, "Missing or extra dataset RGB frames")
            rgb = [_read_png(p, config) for p in rgb_paths]
            rgb_stats = [brightness(frame) for frame in rgb]
            qa_paths = {p.stem: p for p in (path / "lighting_comparison").glob("*.png")}
            pair = {label: _read_png(p, config) for label, p in qa_paths.items()}
            comparisons[scene_id] = validate_lighting_pair(meta, pair, rgb, config)
            kinds[meta["kind"]] += 1
            scene_rows.append({key: meta.get(key) for key in ("scene_id", "kind", "world_seed", "foot_anchor_xyz",
                                                            "settled_player_xyz", "underwater_descent",
                                                            "surface_biome", "foot_biome", "active_view_regions",
                                                            "native_render_settings", "cave_evidence",
                                                            "search_seconds", "search_stats")})
            rows = _view_stats(meta, arrays, rgb_stats)
            view_rows.extend(rows)
            if visit:
                visit(path, meta, arrays, vocabulary, rgb_paths, qa_paths, rows, comparisons[scene_id])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    verified = not missing and bool(comparisons) and all(value["status"] == "VERIFIED"
                                                         for value in comparisons.values())
    return dict(status="NATIVE VERIFIED" if verified else "NOT NATIVE VERIFIED", protocol_version=version,
                capture_status="INCOMPLETE" if missing else "NATIVE CAPTURE VERIFIED",
                scene_count=len(paths), frame_count=len(view_rows), expected_scene_count=len(slots),
                kind_counts=dict(kinds), missing_slots=[slots[index] for index in missing],
                engine_build=provenance[0] if provenance else None, game_build=provenance[1] if provenance else None,
                lighting_comparison=comparisons, scenes=scene_rows, views=view_rows)


def _view_stats(meta, arrays, rgb_stats):
    background = arrays["surface_node_id"] == EMPTY_NODE
    counts = arrays["water_count"]
    rows = []
    for index, (view, light) in enumerate(zip(meta["views"], rgb_stats, strict=True)):
        empty, water = background[index], counts[index]
        warning = []
        if light["near_black_fraction"] >= .95:
            warning.append("RGB is almost entirely dark")
        if empty.mean() >= .9 and meta["kind"] != "surface":
            warning.append("Local input produces very few Surface hits")
        row = dict(scene_id=meta["scene_id"], kind=meta["kind"], view_index=index, brightness=light,
                         grid_anchor_xyz=meta["foot_anchor_xyz"],
                         foot_xyz=view.get("actual_player_world_xyz", meta["foot_anchor_xyz"]),
                         camera_xyz=view["actual_camera_world_xyz"],
                         yaw_deg=view["yaw_deg"], pitch_deg=view["pitch_deg"], fov_x_rad=view["fov_x_rad"],
                         fov_y_rad=view["fov_y_rad"], gamma=view["gamma"], camera_medium=view["camera_medium"],
                         lighting_floor=view["lighting_floor"], renderer_lighting=view["renderer_lighting"],
                         valid_hit_fraction=float((~empty).mean()),
                         background_fraction=float(empty.mean()), water_min=int(water.min()), water_max=int(water.max()),
                         water_nonzero_fraction=float((water > 0).mean()),
                         background_with_water_fraction=float((empty & (water > 0)).mean()), warnings=warning)
        if "solid_ambient_floor" in view:
            row.update(solid_ambient_floor=view["solid_ambient_floor"],
                       material_renderer_lighting=view["material_renderer_lighting"])
        rows.append(row)
    return rows


def validate(root):
    return _scan(root, allow_partial=False)


def _node_colors(names):
    colors = np.zeros((EMPTY_NODE + 1, 3), dtype="u1")
    for index, name in enumerate(names):
        colors[index] = [52 + c % 164 for c in sha256(name.encode()).digest()[:3]]
    colors[EMPTY_NODE] = BACKGROUND
    return colors


def _image(path, label):
    return (f'<figure><img loading="lazy" src="{escape(str(path), quote=True)}" alt="{escape(label, quote=True)}">'
            f'<figcaption>{escape(label)}</figcaption></figure>')


def _view_block(scene_dir, row):
    def vector_text(values):
        return ", ".join(f"{value:.4g}" for value in values)

    index = row["view_index"]
    x, y = (math.degrees(row[key]) for key in ("fov_x_rad", "fov_y_rad"))
    legend = ("Native RGB", "Surface Node ID", "Surface face direction", "Water Count (log gray)",
              "Hit (white) / BACKGROUND (black)")
    images = "".join(_image(scene_dir / f"{index:03d}_{suffix}.png", label) for suffix, label in
                     zip(("rgb", "surface", "faces", "water", "hit"), legend, strict=True))
    warnings = "".join(f'<p class="warning">{escape(warning)}</p>' for warning in row["warnings"])
    light = row["brightness"]
    ambient = (f"<dt>Solid ambient floor</dt><dd>{row['solid_ambient_floor']:.2f}</dd>"
               if "solid_ambient_floor" in row else "")
    return f"""
<article><h3>View {index + 1} · Camera in {row['camera_medium']}</h3>
<dl><dt>Foot XYZ</dt><dd>{vector_text(row['foot_xyz'])}</dd>
<dt>Integer grid anchor XYZ</dt><dd>{vector_text(row['grid_anchor_xyz'])}</dd>
<dt>Actual camera XYZ</dt><dd>{vector_text(row['camera_xyz'])}</dd>
<dt>Yaw / pitch</dt><dd>{row['yaw_deg']:.2f}° / {row['pitch_deg']:.2f}°</dd>
<dt>FOV X × Y</dt><dd>{x:.2f}° × {y:.2f}°</dd><dt>Gamma</dt><dd>{row['gamma']:.1f}</dd>
<dt>Global LUT floor</dt><dd>{row['lighting_floor']:.2f}</dd>{ambient}
<dt>Mean / median luma</dt><dd>{light['mean_luma']:.3f} / {light['median_luma']:.3f}</dd>
<dt>Near-black</dt><dd>{light['near_black_fraction']:.2%}</dd>
<dt>Surface hit / BACKGROUND</dt><dd>{row['valid_hit_fraction']:.2%} / {row['background_fraction']:.2%}</dd>
<dt>Water Count min–max / nonzero</dt><dd>{row['water_min']}–{row['water_max']} / {row['water_nonzero_fraction']:.2%}</dd>
<dt>BACKGROUND with Water Count &gt; 0</dt><dd>{row['background_with_water_fraction']:.2%}</dd></dl>
{warnings}<div class="images">{images}</div></article>"""


def _lighting_block(scene_dir, check, kind, image_names):
    warnings = ' class="warning"' if check["status"] != "VERIFIED" else ""
    captions = []
    for key in ("control", "proposed"):
        light = check[f"{key}_brightness"]
        floor = check[key]["lighting_floor"]
        zero = check[key]["renderer_lighting"]["lut_zero"]
        solid = check[key].get("solid_ambient_floor")
        label = f"Global floor {floor:.2f}" + (f" · solid ambient {solid:.2f}" if solid is not None else "")
        captions.append(f"{label} · LUT[0] {zero} · mean / median {light['mean_luma']:.3f} / "
                        f"{light['median_luma']:.3f} · near-black {light['near_black_fraction']:.2%} · "
                        f"near-white {light['near_white_fraction']:.2%}")
    rows = "".join(_image(scene_dir / f"{label}.png", caption)
                   for label, caption in zip(image_names, captions, strict=True))
    normalized = "Luma uses Rec. 709 on encoded RGB8 (0–255). Near-black &lt; 8. Near-white ≥ 247."
    near_white_gain = check["proposed_brightness"]["near_white_fraction"] - check["control_brightness"]["near_white_fraction"]
    bright_warning = ('<p class="warning">Near-white coverage increased by more than 10 percentage points.</p>'
                      if near_white_gain > .1 else "")
    eye_light = check["control"].get("native_eye_param1")
    eye_note = f" Native eye-cell param1 was {eye_light} in both captures." if eye_light is not None else ""
    return f"""<article><h3{warnings}>Lighting Comparison · {escape(check['status'])}</h3>
<p>{escape(kind.title())} · View {check['view_index'] + 1} · Display gamma {check['proposed']['display_gamma']:.1f}.
Native Node IDs, param1, param2 and White Model agree across the two captures.{eye_note}</p>
<p>{normalized}</p>{bright_warning}<div class="images lighting">{rows}</div></article>"""


def _cave_block(meta):
    if meta["kind"] != "cave":
        return ""
    roof = meta["cave_evidence"]
    return (f"<p>Cave evidence (provisional): roof distance {roof['roof_distance']:.2f} voxels; "
            f"roof thickness {roof['roof_thickness']:.2f} voxels; roof coverage {roof['roof_coverage']:.1%}; "
            f"open sky rays {roof['sky_open_fraction']:.1%}; enclosed lateral rays {roof['side_enclosed_fraction']:.1%}.</p>")


def _underwater_block(meta):
    descent = meta.get("underwater_descent")
    if descent is None:
        return ""
    initial = descent["spawn_player_xyz"][1]
    settled = descent["settled_player_xyz"][1]
    return (f"<p>The player descended from Y {initial:.3f} to {settled:.3f} in {descent['ticks']} native ticks. "
            f"Terrain support ends at Y {descent['support_top_y']:.3f}. "
            "The 48³ input is centered after settling.</p>")


def report(root):
    """Write local HTML, labeled decodes of native arrays, and a JSON summary."""
    root = Path(root)
    output = root / "report"
    sections = []
    colors = None

    def export(path, meta, arrays, vocabulary, rgb_paths, qa_paths, rows, comparison):
        nonlocal colors
        if colors is None:
            colors = _node_colors(vocabulary)
        relative = Path(path.name)
        scene_dir = output / relative
        scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path / "scene.json", scene_dir / "scene.json")
        lighting = relative / "lighting_comparison"
        (output / lighting).mkdir(exist_ok=True)
        for key, frame in qa_paths.items():
            shutil.copyfile(frame, output / lighting / f"{key}.png")
        for index, rgb in enumerate(rgb_paths):
            shutil.copyfile(rgb, scene_dir / f"{index:03d}_rgb.png")
            nodes = arrays["surface_node_id"][index]
            hits = nodes != EMPTY_NODE
            face = np.empty((*nodes.shape, 3), dtype="u1")
            face[:] = BACKGROUND
            face[hits] = FACE_COLORS[arrays["surface_face_dir"][index][hits]]
            water = arrays["water_count"][index]
            water_gray = np.rint(255 * np.log1p(water.astype(np.float32)) / math.log(256)).astype("u1")
            for name, image in (("surface", colors[nodes]), ("faces", face), ("water", water_gray),
                                ("hit", hits.astype("u1") * 255)):
                Image.fromarray(image).save(scene_dir / f"{index:03d}_{name}.png")
        scene_kind = escape(meta["kind"].title())
        search = meta["search_stats"]
        views = "".join(_view_block(relative, row) for row in rows)
        search_count = (f"{search.get('candidate_water_columns', search['candidate_floors'])} submerged starts"
                        if meta["kind"] == "underwater" else f"{search['candidate_floors']} candidate floors")
        sections.append(f"""<section><h2>{scene_kind} · Scene {meta['scene_id']:04d}</h2>
<p>Seed {meta['world_seed']} · Surface biome {escape(meta['surface_biome'])} · Foot biome {escape(meta['foot_biome'])}
 · {sum(meta['active_view_regions'])}/40 active regions · <a href="{relative}/scene.json">Scene metadata</a></p>
<p>Search: {meta['search_seconds']:.1f} s · {search['world_starts']} worlds · {search['surveys']} surveys
 · {search['cave_hints']} cave hints · {search_count}.</p>
{_cave_block(meta)}{_underwater_block(meta)}{_lighting_block(lighting, comparison, meta['kind'], sorted(qa_paths))}
{views}</section>""")

    summary = _scan(root, allow_partial=True, visit=export)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    status_class = ' class="warning"' if summary["status"] != "NATIVE VERIFIED" else ""
    omitted = ", ".join(f"{slot['kind']} {slot['scene_id']:04d}" for slot in summary["missing_slots"])
    incomplete = f'<p class="warning">Missing scenes: {escape(omitted)}</p>' if omitted else ""
    lighting_description = (
        "The native Light LUT uses gamma 3.0 and global floor 0.0. Comparisons switch the solid material "
        "ambient between 0.0 and 0.35 at the same native camera, terrain and light state."
        if summary["protocol_version"] == V3_MATERIAL_VERSION else
        "The native Light LUT uses gamma 3.0. Comparisons switch its global floor between 0.0 and 0.35 "
        "at the same native camera, terrain and light state."
    )
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vox2Pix v3 native collection</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#111723;color:#eef3f8;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:24px}} a{{color:#a9ccff}}
section{{margin:36px 0}}article{{margin:16px 0;padding:16px;background:#1a2434;border:1px solid #35445b;border-radius:8px}}
h1,h2,h3,p{{margin:0 0 8px}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:12px 0}}
dt,figcaption{{color:#bbcad9}}dd{{margin:0;font-variant-numeric:tabular-nums}}.warning{{color:#ffc0b1;font-weight:600}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}}
.lighting{{grid-template-columns:repeat(auto-fit,minmax(215px,1fr));max-width:960px}}
figure{{margin:0}}img{{width:100%;display:block;background:#0a0d15;aspect-ratio:16/9}}figcaption{{margin-top:4px}}
</style><main><h1>Vox2Pix v3 · Native smoke report</h1>
<p{status_class}>{summary['status']} · {summary['capture_status']}</p>
<p>{summary['scene_count']}/{summary['expected_scene_count']} scenes · {summary['frame_count']} native RGB views · <a href="summary.json">Summary JSON</a></p>
{incomplete}<p>RGB retains native sky and fog. Distance fog limits far-field image mismatch; voxel projections cover the local 48³ input only. Surface imagery uses stored discrete attributes and approximate geometry for unsupported drawtypes.</p>
<p>{lighting_description} Cave verification uses a dark control (at least 50% near-black), a mean luma gain of at least 5, and below 50% near-black after applying the floor.</p>
<p>Water Count includes a starting water voxel only when a ray traverses and exits actual liquid. Black in the Water Count panel means count zero. The face colors represent Luanti +X, −X, +Y, −Y, +Z and −Z.</p>
{''.join(sections)}</main></html>"""
    temporary = output / ".index.html.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output / "index.html")
    return summary
