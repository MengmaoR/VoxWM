"""Validate v3 native scenes and export a standalone image comparison report."""

from collections import Counter
from hashlib import sha256
from html import escape
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .protocol import PLAYER_AIR_OFFSETS, RejectedScene, V3_RGB, V3_VERSION, require_air_camera, v3_plan
from .qa import (
    named_ids, number, require, validate_camera_protocol, validate_view,
    validate_visibility_lookup, vector,
)
from .storage import (
    EMPTY_FACE, EMPTY_NODE, array_schema, build_provenance, committed_paths, read_arrays,
    read_json, read_vocabulary, write_json,
)


TORCH_ITEM = "mcl_torches:torch"
TORCH_PNGS = ("torch_off", "torch_on")
TORCH_UNVERIFIED = ("MODULE_UNAVAILABLE", "NO_MEASURABLE_LIGHT")
FACE_COLORS = np.array([(223, 103, 104), (130, 48, 58), (111, 195, 121),
                        (49, 110, 71), (111, 162, 233), (63, 78, 167)], dtype="u1")
BACKGROUND = (17, 23, 39)


def brightness(rgb):
    luma = rgb @ np.array([.2126, .7152, .0722])
    return {"mean_luma": float(luma.mean()), "near_black_fraction": float((luma < 8).mean())}


def _validate_torch(state, on):
    require(isinstance(state, dict), "Missing native torch state")
    for key in ("module_available", "enabled", "setting_enabled", "hidden"):
        require(type(state.get(key)) is bool, f"Invalid torch {key}")
    require(state["hidden"], "Native first-person wielditem was visible")
    require(state["enabled"] == on and state["setting_enabled"], "Incorrect native torch control state")
    if on:
        require(state.get("item") == TORCH_ITEM and number(state, "item_light_source") > 0,
                "Native torch was not wielded or has no registered light")
    else:
        require(state.get("item") != TORCH_ITEM, "OFF comparison still held a torch")


def validate_torch_pair(meta, frames):
    """Validate the actual auxiliary PNG payload before commit and on readback."""
    if meta["kind"] == "surface":
        require(not frames, "Unexpected surface Torch A/B frames")
        return
    require(set(frames) == set(TORCH_PNGS), "Torch OFF/ON comparison requires both PNG frames")
    check = meta.get("torch_ab")
    require(isinstance(check, dict), "Missing native Torch A/B metadata")
    require(check.get("voxel_identity_unchanged") is True, "Torch A/B changed Node ID or param2")
    index = check.get("view_index")
    require(type(index) is int and 0 <= index < len(meta["views"]), "Invalid Torch A/B view index")
    _validate_torch(check.get("off"), False)
    _validate_torch(check.get("on"), True)
    require(check["off"]["module_available"] == check["on"]["module_available"],
            "Torch module availability changed across the comparison")
    actual = {label: brightness(frames[f"torch_{label}"]) for label in ("off", "on")}
    for label, values in actual.items():
        recorded = check.get(f"{label}_brightness")
        require(isinstance(recorded, dict) and all(math.isclose(number(recorded, key), value, abs_tol=1e-8)
                                                   for key, value in values.items()),
                f"Torch {label.upper()} brightness differs from its saved PNG")
    status = check.get("status")
    require(status in (*TORCH_UNVERIFIED, "VERIFIED"), "Unknown Torch A/B status")
    if not check["on"]["module_available"]:
        require(status == "MODULE_UNAVAILABLE", "Missing module was misreported as effective")
    if status == "VERIFIED":
        require(check["on"]["module_available"] and actual["on"]["mean_luma"] > actual["off"]["mean_luma"],
                "Torch was reported as verified without native brightness improvement")


def validate_payload(meta, config, vocabulary, slot, arrays):
    from .geometry import is_water_liquid, water_contains
    from .v3_sampling import cave_evidence, scene_visibility

    array_schema(arrays, config)
    require(all(meta.get(key) == value for key, value in slot.items()), "Scene does not match its scheduled slot")
    require(type(meta.get("world_seed")) is int, "Invalid world seed")
    for key in ("surface_biome", "foot_biome", "engine_build", "game_build"):
        require(isinstance(meta.get(key), str) and bool(meta[key]), f"Missing {key}")
    kind = slot["kind"]
    foot = vector(meta, "foot_anchor_xyz", integer=True)
    origin = vector(meta, "region_origin_xyz", integer=True)
    origin_enu = vector(meta, "region_origin_enu", integer=True)
    require(np.array_equal(origin, foot - 24) and np.array_equal(origin_enu, origin[[0, 2, 1]]),
            "Invalid foot-centered 48³ coordinates")
    require(math.isclose(number(meta, "world_time"), .5, abs_tol=1e-5) and meta.get("weather") == "none"
            and number(meta, "gamma") == 3., "Invalid native daylight, weather or gamma")
    settings = meta.get("native_render_settings")
    require(isinstance(settings, dict) and all(settings.get(key) == value for key, value in V3_RGB.items()),
            "Native viewing range, fog or gamma differs from the v3 protocol")
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
    try:
        center, full, active = scene_visibility(nodes, definitions, kind)
        if kind == "cave":
            require(meta.get("cave_evidence") == cave_evidence(nodes, definitions),
                    "Recorded cave enclosure differs from the decoded raw geometry")
    except RejectedScene as exc:
        raise ValueError(str(exc)) from exc
    require(math.isclose(number(meta, "center_free_ratio"), center, abs_tol=1e-10)
            and math.isclose(number(meta, "full_free_ratio"), full, abs_tol=1e-10), "Changed free-space metadata")
    require(meta.get("active_view_regions") == active, "Recorded visibility mask differs from decoded geometry")
    body = nodes[24, 24, 24 + np.asarray(PLAYER_AIR_OFFSETS)]
    require(np.isin(body, (air, *water) if kind == "underwater" else (air,)).all(),
            "Player body intersects native solid geometry")
    support = int(nodes[24, 24, 23])
    require(support in terrain or (kind == "surface" and meta.get("anchor_kind") == "surface_water" and support in water),
            "Foot anchor lacks native terrain support")

    def require_medium(grid, rules, eye):
        if kind == "underwater":
            if not water_contains(grid, params, rules, eye):
                raise RejectedScene("Actual RGB camera is outside the native water volume")
        else:
            require_air_camera(grid, rules, eye)

    views = meta.get("views")
    require(isinstance(views, list) and len(views) == config["views_per_scene"], "Missing scene views")
    ids, ticks = zip(*(validate_view(view, config, foot, active, meta, nodes, definitions,
                                    camera_check=require_medium) for view in views))
    require(len(set(ids)) == len(ids) and all(a < b for a, b in zip(ticks, ticks[1:])),
            "View regions must be distinct and native ticks increasing")
    for view in views:
        require(view.get("camera_medium") == ("water" if kind == "underwater" else "air"),
                "Recorded view medium differs from expected scene kind")
        _validate_torch(view.get("torch"), True)

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
    require(manifest.get("version") == V3_VERSION and manifest.get("axes") == "ENU" and config.get("mode") == "smoke"
            and config.get("views_per_scene") == 5 and (config.get("width"), config.get("height")) == (640, 360),
            "Expected a native Vox2Pix v3 smoke dataset at 640×360 with five views per scene")
    slots = {slot["scene_id"]: slot for slot in v3_plan(config)}
    paths = committed_paths(root)
    missing = sorted(slots.keys() - paths.keys())
    require(paths.keys() <= slots.keys() and (allow_partial or not missing),
            f"Expected {len(slots)} committed scenes; found {len(paths)}")
    vocabulary = read_vocabulary(root) if paths else []
    kinds, torch, seeds, view_rows = Counter(), {}, set(), []
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
            rgb_stats = [brightness(_read_png(p, config)) for p in rgb_paths]
            qa_paths = {p.stem: p for p in (path / "qa").glob("*.png")}
            validate_torch_pair(meta, {label: _read_png(p, config) for label, p in qa_paths.items()})
            if qa_paths:
                torch[scene_id] = meta["torch_ab"]
            kinds[meta["kind"]] += 1
            rows = _view_stats(meta, arrays, rgb_stats)
            view_rows.extend(rows)
            if visit:
                visit(path, meta, arrays, vocabulary, rgb_paths, qa_paths, rows)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    verified = not missing and bool(torch) and all(value["status"] == "VERIFIED" for value in torch.values())
    return dict(status="NATIVE VERIFIED" if verified else "NOT NATIVE VERIFIED", protocol_version=V3_VERSION,
                capture_status="INCOMPLETE" if missing else "NATIVE CAPTURE VERIFIED",
                scene_count=len(paths), frame_count=len(view_rows), expected_scene_count=len(slots),
                kind_counts=dict(kinds), missing_slots=[slots[index] for index in missing],
                engine_build=provenance[0] if provenance else None, game_build=provenance[1] if provenance else None,
                torch_ab=torch, views=view_rows)


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
        rows.append(dict(scene_id=meta["scene_id"], kind=meta["kind"], view_index=index, brightness=light,
                         foot_xyz=meta["foot_anchor_xyz"], camera_xyz=view["actual_camera_world_xyz"],
                         yaw_deg=view["yaw_deg"], pitch_deg=view["pitch_deg"], fov_x_rad=view["fov_x_rad"],
                         fov_y_rad=view["fov_y_rad"], gamma=view["gamma"], camera_medium=view["camera_medium"],
                         torch_item=view["torch"]["item"], valid_hit_fraction=float((~empty).mean()),
                         background_fraction=float(empty.mean()), water_min=int(water.min()), water_max=int(water.max()),
                         water_nonzero_fraction=float((water > 0).mean()),
                         background_with_water_fraction=float((empty & (water > 0)).mean()), warnings=warning))
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
    return f"""
<article><h3>View {index + 1} · Camera in {row['camera_medium']}</h3>
<dl><dt>Foot XYZ</dt><dd>{vector_text(row['foot_xyz'])}</dd>
<dt>Actual camera XYZ</dt><dd>{vector_text(row['camera_xyz'])}</dd>
<dt>Yaw / pitch</dt><dd>{row['yaw_deg']:.2f}° / {row['pitch_deg']:.2f}°</dd>
<dt>FOV X × Y</dt><dd>{x:.2f}° × {y:.2f}°</dd><dt>Gamma</dt><dd>{row['gamma']:.1f}</dd>
<dt>Wielded item</dt><dd>{escape(row['torch_item'])}</dd>
<dt>Mean luma / near-black</dt><dd>{light['mean_luma']:.3f} / {light['near_black_fraction']:.2%}</dd>
<dt>Surface hit / BACKGROUND</dt><dd>{row['valid_hit_fraction']:.2%} / {row['background_fraction']:.2%}</dd>
<dt>Water Count min–max / nonzero</dt><dd>{row['water_min']}–{row['water_max']} / {row['water_nonzero_fraction']:.2%}</dd>
<dt>BACKGROUND with Water Count &gt; 0</dt><dd>{row['background_with_water_fraction']:.2%}</dd></dl>
{warnings}<div class="images">{images}</div></article>"""


def _torch_block(scene_dir, check):
    warnings = ' class="warning"' if check["status"] != "VERIFIED" else ""
    off, on = (check[f"{key}_brightness"] for key in ("off", "on"))
    return f"""<article><h3{warnings}>Torch A/B · {escape(check['status'])}</h3>
<p>View {check['view_index'] + 1}. Node ID and param2 unchanged across OFF and ON.</p>
<div class="images torch">{_image(scene_dir / 'torch_off.png', f"OFF · luma {off['mean_luma']:.3f} · near-black {off['near_black_fraction']:.2%}")}
{_image(scene_dir / 'torch_on.png', f"ON · luma {on['mean_luma']:.3f} · near-black {on['near_black_fraction']:.2%}")}</div></article>"""


def _cave_block(meta):
    if meta["kind"] != "cave":
        return ""
    roof = meta["cave_evidence"]
    return (f"<p>Cave evidence (provisional): roof distance {roof['roof_distance']:.2f} voxels; "
            f"roof thickness {roof['roof_thickness']:.2f} voxels; roof coverage {roof['roof_coverage']:.1%}; "
            f"open sky rays {roof['sky_open_fraction']:.1%}; enclosed lateral rays {roof['side_enclosed_fraction']:.1%}.</p>")


def report(root):
    """Write local HTML, labeled decodes of native arrays, and a JSON summary."""
    root = Path(root)
    output = root / "report"
    sections = []
    colors = None

    def export(path, meta, arrays, vocabulary, rgb_paths, qa_paths, rows):
        nonlocal colors
        if colors is None:
            colors = _node_colors(vocabulary)
        relative = Path(path.name)
        scene_dir = output / relative
        scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path / "scene.json", scene_dir / "scene.json")
        for key, frame in qa_paths.items():
            shutil.copyfile(frame, scene_dir / f"{key}.png")
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
        torch = _torch_block(relative, meta["torch_ab"]) if qa_paths else ""
        views = "".join(_view_block(relative, row) for row in rows)
        sections.append(f"""<section><h2>{scene_kind} · Scene {meta['scene_id']:04d}</h2>
<p>Seed {meta['world_seed']} · Surface biome {escape(meta['surface_biome'])} · Foot biome {escape(meta['foot_biome'])}
 · {sum(meta['active_view_regions'])}/40 active regions · <a href="{relative}/scene.json">Scene metadata</a></p>
{_cave_block(meta)}{torch}{views}</section>""")

    summary = _scan(root, allow_partial=True, visit=export)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    status_class = ' class="warning"' if summary["status"] != "NATIVE VERIFIED" else ""
    omitted = ", ".join(f"{slot['kind']} {slot['scene_id']:04d}" for slot in summary["missing_slots"])
    incomplete = f'<p class="warning">Missing scenes: {escape(omitted)}</p>' if omitted else ""
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
.torch{{grid-template-columns:repeat(auto-fit,minmax(215px,1fr));max-width:960px}}
figure{{margin:0}}img{{width:100%;display:block;background:#0a0d15;aspect-ratio:16/9}}figcaption{{margin-top:4px}}
</style><main><h1>Vox2Pix v3 · Native smoke report</h1>
<p{status_class}>{summary['status']} · {summary['capture_status']}</p>
<p>{summary['scene_count']}/{summary['expected_scene_count']} scenes · {summary['frame_count']} native RGB views · <a href="summary.json">Summary JSON</a></p>
{incomplete}<p>RGB retains native sky and fog. Distance fog limits far-field image mismatch; voxel projections cover the local 48³ input only. Surface imagery uses stored discrete attributes and approximate geometry for unsupported drawtypes.</p>
<p>Node ID and param2 do not encode the original light field. The torch comparison measures whether the fixed rendering setup increases brightness; learning its effect requires a training experiment.</p>
<p>Water Count includes a starting water voxel only when a ray traverses and exits actual liquid. Black in the Water Count panel means count zero. The face colors represent Luanti +X, −X, +Y, −Y, +Z and −Z.</p>
{''.join(sections)}</main></html>"""
    temporary = output / ".index.html.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output / "index.html")
    return summary
