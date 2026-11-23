"""Offline RGB diagnostics for matched native material-lighting captures."""

from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import zarr

from .storage import EMPTY_NODE, write_json
from .v3_report import _node_colors


MODES = {
    "A": ("Native control", 0.0, 0.0),
    "B": ("Global LUT baseline", 0.35, 0.0),
    "C": ("Material ambient", 0.0, 0.35),
}
PAIRS = ("A-C", "B-C")
REGIONS = {
    "direct_geometry": "Direct geometry",
    "geometry_through_water": "Geometry through water",
    "water_background": "Water traversal · local BACKGROUND",
    "dry_background": "Local BACKGROUND · no water",
}
REC709 = np.array([.2126, .7152, .0722])
DELTA_BINS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def luma_stats(values):
    if not values.size:
        return None
    p10, median = np.percentile(values, [10., 50.])
    return dict(mean_luma=float(values.mean()), median_luma=float(median), p10_luma=float(p10),
                near_black_fraction=float((values < 8).mean()),
                near_white_fraction=float((values >= 247).mean()))


def delta_stats(values):
    if not values.size:
        return None
    return dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)),
                unchanged_fraction=float((values == 0).mean()),
                histogram=np.histogram(values, bins=DELTA_BINS)[0].tolist())


def region_masks(surface, water):
    hit = surface != EMPTY_NODE
    wet = water > 0
    return dict(zip(REGIONS, (hit & ~wet, hit & wet, ~hit & wet, ~hit & ~wet), strict=True))


def measures(lumas, differences, mask):
    return dict(pixel_count=int(mask.sum()), pixel_ratio=float(mask.mean()),
                image_metrics={mode: luma_stats(luma[mask]) for mode, luma in lumas.items()},
                rgb_difference={pair: delta_stats(delta[mask]) for pair, delta in differences.items()})


def neighbor_luma(lumas, masks):
    """Adjacent-pixel contrast measured along exactly the masks supplied."""
    contrast = {}
    for mode, luma in lumas.items():
        horizontal = np.abs(np.diff(luma, axis=1))[masks[0]]
        vertical = np.abs(np.diff(luma, axis=0))[masks[1]]
        values = np.concatenate((horizontal, vertical))
        contrast[mode] = (dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)))
                          if values.size else None)
    return contrast


def cave_features(surface, face, vocabulary, images, lumas):
    """Use approximate hits to inspect material texture independently of silhouettes."""
    hit = surface != EMPTY_NODE
    id_match = (surface[:, 1:] == surface[:, :-1], surface[1:] == surface[:-1])
    face_match = (face[:, 1:] == face[:, :-1], face[1:] == face[:-1])
    materials = {}
    for ident in np.unique(surface[hit]):
        name = vocabulary[int(ident)]
        node = name.split(":", 1)[-1]
        if node not in ("stone", "diorite") and "ore" not in node and "_with_" not in node:
            continue
        mask = surface == ident
        same_side = (mask[:, 1:] & mask[:, :-1] & face_match[0],
                     mask[1:] & mask[:-1] & face_match[1])
        materials[name] = dict(
            pixel_count=int(mask.sum()),
            image_metrics={mode: luma_stats(luma[mask]) for mode, luma in lumas.items()},
            median_rgb={mode: np.median(rgb[mask], axis=0).tolist() for mode, rgb in images.items()},
            within_material_face_luma_step=neighbor_luma(lumas, same_side))
    edges = (hit[:, 1:] & hit[:, :-1] & ~(id_match[0] & face_match[0]),
             hit[1:] & hit[:-1] & ~(id_match[1] & face_match[1]))
    return dict(materials=materials, material_or_face_edge_pair_count=sum(int(mask.sum()) for mask in edges),
                material_or_face_edge_luma_step=neighbor_luma(lumas, edges))


def underwater_view2(images, groups, lumas, differences, water):
    blue = {mode: (rgb[..., 2] > rgb[..., 0]) & (rgb[..., 2] > rgb[..., 1]) for mode, rgb in images.items()}
    original_blue_background = blue["A"] & groups["water_background"]
    return dict(
        local_background_ratio=float((groups["water_background"] | groups["dry_background"]).mean()),
        blue_pixel_fraction={mode: float(mask.mean()) for mode, mask in blue.items()},
        blue_background=measures(lumas, differences, original_blue_background),
        covered_by_water_count_fraction={mode: float((water[mask] > 0).mean()) if mask.any() else None
                                         for mode, mask in blue.items()},
        attribution_limit="The 48³ projection does not identify far geometry, the water surface, fog or "
                          "the underwater overlay at each RGB pixel. Their separate contributions require "
                          "native pass-level buffers.",
    )


def _save(path, array):
    Image.fromarray(array).save(path)


def _difference_panels(path, delta, groups):
    height, width = delta.shape
    # Fixed 8-bit scale across views and mode pairs, with a separate cell per mask.
    values = np.rint(delta).astype("u1")
    grid = np.zeros((height * 2, width * 2), dtype="u1")
    for index, mask in enumerate(groups.values()):
        row, col = divmod(index, 2)
        panel = grid[row * height:(row + 1) * height, col * width:(col + 1) * width]
        panel[mask] = values[mask]
    _save(path, grid)


def write_comparison(root, scene_path, config, vocabulary, meta, index, images, evidence):
    """Write images and compute measures for one verified A/B/C view."""
    expected = (config["height"], config["width"], 3)
    if set(images) != set(MODES) or any(rgb.shape != expected or rgb.dtype != np.uint8 for rgb in images.values()):
        raise ValueError("Every lighting mode needs an aligned RGB8 frame")
    store = zarr.open_group(str(Path(scene_path) / "voxels.zarr"), mode="r")
    surface, face, water = (store[name][index] for name in ("surface_node_id", "surface_face_dir", "water_count"))
    if any(array.shape != expected[:2] for array in (surface, face, water)):
        raise ValueError("White Model does not align with its RGB view")
    groups = region_masks(surface, water)
    hit = surface != EMPTY_NODE
    lumas = {mode: images[mode] @ REC709 for mode in MODES}
    differences = {pair: np.abs(images[pair[0]].astype(np.int16) - images[pair[2]]).mean(axis=2)
                   for pair in PAIRS}
    relative = Path(f"scene_{meta['scene_id']:04d}_view_{index + 1:02d}")
    output = Path(root) / relative
    output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        _save(output / f"{mode}.png", images[mode])
    _save(output / "surface.png", _node_colors(vocabulary)[surface])
    _save(output / "surface_hit.png", hit.astype("u1") * 255)
    _save(output / "background.png", (~hit).astype("u1") * 255)
    _save(output / "water_count.png", np.rint(np.log1p(water.astype(np.float32)) * 255 / math.log(256)).astype("u1"))
    for pair, delta in differences.items():
        _difference_panels(output / f"{pair}.png", delta, groups)
    view = meta["views"][index]
    return dict(
        scene_id=meta["scene_id"], kind=meta["kind"], world_seed=meta["world_seed"],
        view_index=index, relative_assets=relative.as_posix(), pose=view,
        original_render_settings=meta.get("native_render_settings", {}),
        image_metrics={mode: luma_stats(luma) for mode, luma in lumas.items()},
        rgb_difference={pair: delta_stats(delta) for pair, delta in differences.items()},
        regions={group: measures(lumas, differences, mask) for group, mask in groups.items()},
        surface_hit_ratio=float(hit.mean()), water_count_histogram=np.bincount(water.ravel(), minlength=256).tolist(),
        cave_materials=cave_features(surface, face, vocabulary, images, lumas) if meta["kind"] == "cave" else None,
        underwater_view2=(underwater_view2(images, groups, lumas, differences, water)
                          if meta["kind"] == "underwater" and index == 1 else None),
        evidence=evidence,
    )


def _figure(source, caption):
    src, title = escape(str(source), quote=True), escape(caption, quote=True)
    return (f'<figure><a href="{src}"><img loading="lazy" src="{src}" alt="{title}"></a>'
            f"<figcaption>{title}</figcaption></figure>")


def _values(values, fields):
    return " / ".join(f"{values[key]:.3f}" for key in fields) if values else "—"


def _mode_row(mode, caption, global_floor, solid_floor, stats, probe):
    global_floor = probe.get("lighting_floor", global_floor)
    solid_floor = probe.get("solid_ambient_floor", solid_floor)
    tick = probe.get("render_tick", "—")
    lut = probe.get("lut_zero", "—")
    return (f"<tr><th>{mode} {caption}</th><td>{global_floor:.2f} / {solid_floor:.2f}</td>"
            f"<td>{_values(stats, ('mean_luma', 'median_luma', 'p10_luma'))}</td>"
            f"<td>{stats['near_black_fraction']:.2%} / {stats['near_white_fraction']:.2%}</td>"
            f"<td>{tick} / {lut}</td></tr>")


def _region_row(group, row):
    lumas = "".join(f"<td>{_values(row['image_metrics'][mode], ('mean_luma', 'median_luma'))}</td>" for mode in MODES)
    deltas = "".join(f"<td>{_values(row['rgb_difference'][pair], ('mean', 'p95'))}</td>" for pair in PAIRS)
    return f"<tr><th>{group}</th><td>{row['pixel_count']:,} / {row['pixel_ratio']:.2%}</td>{lumas}{deltas}</tr>"


def _table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{escape(header)}</th>" for header in headers) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _cave_block(cave):
    if cave is None:
        return ""
    materials = []
    for name, row in cave["materials"].items():
        lumas = " · ".join(f"{mode} {_values(row['image_metrics'][mode], ('mean_luma', 'median_luma'))}"
                          for mode in MODES)
        steps = " · ".join(f"{mode} {_values(row['within_material_face_luma_step'][mode], ('mean', 'p95'))}"
                           for mode in MODES)
        colors = " · ".join(f"{mode} " + "/".join(f"{value:.0f}" for value in row["median_rgb"][mode])
                            for mode in MODES)
        materials.append(f"<tr><th>{escape(name)}</th><td>{row['pixel_count']:,}</td><td>{lumas}</td>"
                         f"<td>{steps}</td><td>{colors}</td></tr>")
    edge = " · ".join(f"{mode} {_values(cave['material_or_face_edge_luma_step'][mode], ('mean', 'p95'))}"
                      for mode in MODES)
    return ("<h4>Cave texture and edges</h4><p>Same-material, same-face adjacent luma steps describe "
            "local texture contrast under the approximate White Model.</p>" +
            _table(("Material", "Pixels", "Mean / median luma", "Mean / P95 luma step", "Median RGB"), materials) +
            f"<p>{cave['material_or_face_edge_pair_count']:,} adjacent material or face boundaries. "
            f"Mean / P95 luma step {edge}.</p>")


def _underwater_block(row):
    if row is None:
        return ""
    blue = row["blue_background"]
    delta = blue["rgb_difference"]["A-C"]
    unchanged = f" {delta['unchanged_fraction']:.2%} of those pixels match exactly." if delta else ""
    water_covered = ", ".join(f"{mode} {value:.2%}" if value is not None else f"{mode} —"
                              for mode, value in row["covered_by_water_count_fraction"].items())
    return (f"<h4>Underwater view 2</h4><p>Local BACKGROUND covers {row['local_background_ratio']:.2%}. "
            f"The blue region in A with Water Count &gt; 0 and a local BACKGROUND comprises "
            f"{blue['pixel_count']:,} pixels. Its A–C RGB mean / P95 absolute difference is "
            f"{_values(delta, ('mean', 'p95'))}.{unchanged}</p><p>Water Count &gt; 0 among each mode's blue pixels "
            f"{water_covered}. {escape(row['attribution_limit'])}</p>")


def _view_block(row):
    pose = row["pose"]
    base = Path(row["relative_assets"])
    camera = ", ".join(f"{value:.5g}" for value in pose["actual_camera_world_xyz"])
    yaw, pitch = pose["yaw_deg"], pose["pitch_deg"]
    fov = " × ".join(f"{math.degrees(pose[key]):.2f}°" for key in ("fov_x_rad", "fov_y_rad"))
    water_nonzero = 1 - row["water_count_histogram"][0] / sum(row["water_count_histogram"])
    evidence = row["evidence"]
    probes = evidence.get("probes", {})
    versions = ", ".join(sorted({probe["version"] for probe in probes.values() if "version" in probe}))
    eye = evidence.get("native_eye_param1", "—")
    invariant_fields = ("node_id_identical", "raw_param1_identical", "raw_param2_identical",
                        "camera_pose_identical", "white_identical")
    invariants = ('<p class="proof">Native Node IDs, raw param1/param2, camera and saved White data agree.</p>'
                  if all(evidence.get(key) is True for key in invariant_fields) else "")
    info = json.dumps({"evidence": evidence, "render_settings": row["original_render_settings"]}, indent=2)
    rgb = "".join(_figure(base / f"{mode}.png", f"{mode} · {label}") for mode, (label, _, _) in MODES.items())
    masks = "".join(_figure(base / f"{name}.png", caption) for name, caption in (
        ("surface", "Surface node classes"), ("surface_hit", "SurfaceHit"), ("background", "Local BACKGROUND"),
        ("water_count", "Water Count · log gray")))
    differences = "".join(_figure(base / f"{pair}.png", f"{pair} · masked mean absolute RGB difference")
                          for pair in PAIRS)
    modes = _table(("Mode", "Global / solid floor", "Mean / median / P10 luma", "Near-black / near-white",
                    "Render tick / zero-light LUT"),
                   (_mode_row(mode, *value, row["image_metrics"][mode], probes.get(mode, {}))
                    for mode, value in MODES.items()))
    regions = _table(("White Model region", "Count / ratio", "A mean / median", "B mean / median",
                      "C mean / median", "A–C mean / P95", "B–C mean / P95"),
                     (_region_row(REGIONS[name], row["regions"][name]) for name in REGIONS))
    whole = " · ".join(f"{pair} {_values(row['rgb_difference'][pair], ('mean', 'p95'))}" for pair in PAIRS)
    return f"""<article id="{base.name}"><h3>View {row['view_index'] + 1}</h3>
<p>Camera ({camera}) · yaw {yaw:.2f}° · pitch {pitch:.2f}° · FOV {fov} · Gamma {pose['gamma']:.1f}</p>
<p>Native eye param1 {eye} · Renderer {escape(versions or "see evidence")}</p>{invariants}
<div class="rgb">{rgb}</div>{modes}
<p>Whole-image mean / P95 absolute RGB difference {whole}. SurfaceHit {row['surface_hit_ratio']:.2%}.
Water Count &gt; 0 {water_nonzero:.2%}.</p>
{_cave_block(row['cave_materials'])}{_underwater_block(row['underwater_view2'])}
<details><summary>White Model masks and stratified differences</summary>
<div class="masks">{masks}</div>{regions}
<p>Difference panels show direct geometry, geometry through water, water traversal with
local BACKGROUND and dry BACKGROUND in row-major order. Black denotes zero or masked-out pixels.
All panels use the same 0–255 scale.</p><div class="differences">{differences}</div></details>
<details><summary>Renderer, native eye light, camera and voxel/White proof</summary><pre>{escape(info)}</pre></details>
</article>"""


def finish_report(root, views, metadata):
    """Publish a standalone HTML page and exact machine-readable metrics."""
    root = Path(root)
    grouped = defaultdict(list)
    for row in sorted(views, key=lambda entry: (entry["scene_id"], entry["view_index"])):
        grouped[row["scene_id"], row["kind"], row["world_seed"]].append(row)
    sections = []
    for (scene, kind, seed), rows in grouped.items():
        sections.append(f'<section id="scene-{scene}"><h2>{escape(kind.title())} · seed {seed}</h2>' +
                        "".join(_view_block(row) for row in rows) + "</section>")
    summary = dict(metadata=metadata, modes=MODES, luma_definition="Rec.709 on native RGB8 (0–255)",
                   rgb_delta_definition="Per-pixel mean of absolute R, G and B differences",
                   rgb_delta_histogram_edges=DELTA_BINS, views=views)
    write_json(root / "summary.json", summary)
    navigation = " · ".join(f'<a href="#scene-{scene}">{escape(kind.title())}</a>' for scene, kind, _ in grouped)
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vox2Pix material lighting comparison</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#14191f;color:#e8edf4;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1560px;margin:auto;padding:24px}}article{{background:#1d2734;padding:16px;margin:20px 0}}
h1,h2,h3,h4,p{{margin:0 0 10px}}section{{margin:36px 0}}
.proof{{color:#a5d6bb}}
figure{{margin:0;min-width:0}}img{{width:100%;display:block;background:#11151c}}a,summary{{color:#a9ccff}}
figcaption,th,p{{color:#c5d1df}}p{{margin-top:8px}}
.rgb,.masks,.differences{{display:grid;gap:10px}}.rgb{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.masks{{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}.differences{{grid-template-columns:repeat(2,minmax(0,1fr))}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums}}
th,td{{padding:4px 7px;border-bottom:1px solid #354355;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
details{{margin:10px 0}}summary{{cursor:pointer}}
@media(max-width:760px){{main{{padding:12px}}.rgb{{grid-template-columns:1fr}}table{{font-size:.8rem}}}}
</style>
<main><h1>Vox2Pix · Native material lighting comparison</h1>
<p>{len(grouped)} scenes · {len(views)} matched views · <a href="summary.json">Measured JSON</a></p>
<nav>{navigation}</nav>
<p>Rec.709 luma on native RGB8 (0–255). Near-black means luma &lt; 8 and near-white
means luma ≥ 247. RGB difference is the per-pixel mean absolute channel difference.</p>
<p>A/B/C use the same saved 48³ inputs and camera poses. White Model regions approximate RGB
visibility. The material branch leaves native water, sky, fog and the underwater overlay on
their existing paths. Background beyond the crop can still include terrain affected by ambient lighting.</p>
{''.join(sections)}</main></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    return summary
