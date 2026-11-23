"""Offline, same-camera diagnostics for native map range and distance fog."""

from collections import defaultdict
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import zarr

from .material_report import REC709, delta_stats, luma_stats, measures, region_masks
from .storage import EMPTY_NODE, write_json


MODES = {"A": "Fog off", "B": "Old fog · 24–30", "C": "Proposed fog · 20–30"}
DISTANCES = (20, 22, 24, 25, 27, 29, 30, 32)


def fog_factor(distance, start, end, enabled=True):
    if not enabled:
        return 0.
    if end <= start:
        raise ValueError("Native fog end must be after its start")
    return float(np.clip((distance - start) / (end - start), 0., 1.))


def native_fog_target(color, factor):
    """Pinned native node shader colors the fog target according to clarity."""
    color = np.asarray(color[:3], dtype=np.float64) / 255.
    maximum = color.max(initial=0.)
    if maximum < 1e-7:
        maximum = 1.
    return color * (color / maximum) ** (2. * (1. - factor)) * 255.


def image_measures(rgb):
    return dict(mean_rgb=rgb.reshape(-1, 3).mean(axis=0).tolist(), **luma_stats(rgb @ REC709))


def _images(images):
    if set(images) != set(MODES):
        raise ValueError("Aligned fog-off, old and proposed images are required")
    shape = images["A"].shape
    if len(shape) != 3 or shape[-1] != 3 or any(
            rgb.shape != shape or rgb.dtype != np.uint8 for rgb in images.values()):
        raise ValueError("Fog comparisons require aligned native RGB8 images")
    return shape[:2]


def _save(path, array):
    Image.fromarray(array).save(path)


def _sample_box(row, height, width):
    if "pixel_bbox" in row:
        left, top, right, bottom = map(int, row["pixel_bbox"])
    else:
        x, y = row.get("pixel_center", (width / 2, height / 2))
        left, top = int(x) - 3, int(y) - 3
        right, bottom = left + 7, top + 7
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("Marker sample lies outside the native RGB frame")
    return left, top, right, bottom


def write_marker(output, row, images):
    """Store one actual native marker triplet and measure its core pixels."""
    height, width = _images(images)
    row = dict(row)
    mask = row.pop("marker_mask", None)
    distance = float(row.get("measured_distance", row.get("actual_distance", row["distance"])))
    relative = Path("calibration") / f"distance_{row['distance']:g}"
    folder = Path(output) / relative
    folder.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = box = _sample_box(row, height, width)
    crop = _sample_box(dict(pixel_bbox=row.get("crop_bbox", box)), height, width)
    if mask is None:
        mask = np.zeros((height, width), dtype=bool)
        mask[top:bottom, left:right] = True
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (height, width) or not mask.any():
            raise ValueError("The marker core mask must select native RGB pixels")
    samples = {mode: image[mask].astype(np.float64) for mode, image in images.items()}
    probes = row["probes"]
    stats = {}
    for mode, rgb in images.items():
        probe = probes[mode]
        color = np.asarray(probe["actual_fog_color"][:3], dtype=float)
        factor = fog_factor(distance, probe["actual_fog_start"], probe["actual_fog_end"],
                            probe["fog_enabled"])
        target = native_fog_target(color, factor)
        current, baseline = samples[mode], samples["A"]
        native_expected = baseline * (1. - factor) + target * factor
        endpoint_expected = baseline * (1. - factor) + color * factor
        span = target - baseline
        denominator = float(np.square(span).sum())
        fitted = float(((current - baseline) * span).sum() / denominator) if denominator > 1e-9 else None
        stats[mode] = dict(
            mean_rgb=current.mean(axis=0).tolist(),
            mean_luma=float((current @ REC709).mean()),
            rgb_distance_from_fog=float(np.abs(current - color).mean()),
            native_fog_factor=factor,
            native_tinted_target_rgb=target.tolist(),
            native_model_mean_abs_error=float(np.abs(current - native_expected).mean()),
            fixed_color_linear_mean_abs_error=float(np.abs(current - endpoint_expected).mean()),
            fitted_native_fog_factor=fitted,
            marker_texture_std=float(current.std(axis=0).mean()),
            residual_color_fraction=(float(np.abs(current - color).mean() / np.abs(baseline - color).mean())
                                     if np.abs(baseline - color).mean() > 1e-9 else None),
        )
        _save(folder / f"{mode}.png", rgb)
        _save(folder / f"{mode}_crop.png", rgb[crop[1]:crop[3], crop[0]:crop[2]])
    proposed = probes["C"]
    runtime_ok = (proposed.get("fog_enabled") is True and proposed.get("directional_colored_fog") is True
                  and proposed.get("fog_type") == "linear" and proposed.get("shader_uniforms_bound") is True
                  and abs(proposed["actual_fog_start"] - 20.) <= .001
                  and abs(proposed["actual_fog_end"] - 30.) <= .001)
    row.update(relative_assets=relative.as_posix(), measured_distance=distance, pixel_bbox=list(box),
               sample_pixel_count=int(mask.sum()), image_metrics=stats,
               expected_proposed_factor=fog_factor(distance, 20., 30.),
               checks=dict(runtime_fog=runtime_ok,
                           native_color_equation=stats["C"]["native_model_mean_abs_error"] <= 3.,
                           endpoint_hidden=(distance < 30. or stats["C"]["rgb_distance_from_fog"] <= 3.)))
    return row


def write_existing(output, row, images, source_scene_path):
    """Use the saved White Model only for region diagnostics of matched RGB."""
    height, width = _images(images)
    row = dict(row)
    relative = Path(f"scene_{row['scene_id']:04d}_view_{row['view_index'] + 1:02d}")
    folder = Path(output) / relative
    folder.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(Path(source_scene_path) / "voxels.zarr"), mode="r")
    index = row["view_index"]
    surface, water = (store[field][index] for field in ("surface_node_id", "water_count"))
    if surface.shape != (height, width) or water.shape != (height, width):
        raise ValueError("Saved White Model dimensions differ from the native RGB")
    masks = region_masks(surface, water)
    lumas = {mode: rgb @ REC709 for mode, rgb in images.items()}
    diffs = {pair: np.abs(images[pair[0]].astype(np.int16) - images[pair[2]]).mean(axis=2)
             for pair in ("A-C", "B-C")}
    for mode, rgb in images.items():
        _save(folder / f"{mode}.png", rgb)
    for pair, delta in diffs.items():
        _save(folder / f"{pair}.png", np.rint(delta).astype("u1"))
    hit = surface != EMPTY_NODE
    _save(folder / "surface_hit.png", hit.astype("u1") * 255)
    _save(folder / "background.png", (~hit).astype("u1") * 255)
    _save(folder / "water_count.png", np.rint(np.log1p(water.astype(float)) * 255 / np.log(256)).astype("u1"))
    row.update(relative_assets=relative.as_posix(),
               image_metrics={mode: image_measures(rgb) for mode, rgb in images.items()},
               rgb_difference={pair: delta_stats(delta) for pair, delta in diffs.items()},
               regions={name: measures(lumas, diffs, mask) for name, mask in masks.items()},
               surface_hit_ratio=float(hit.mean()))
    return row


def _figure(path, label):
    source, label = escape(str(path), quote=True), escape(label)
    return f'<figure><a href="{source}"><img loading="lazy" src="{source}" alt="{label}"></a><figcaption>{label}</figcaption></figure>'


def _table(headers, rows):
    cells = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return "<table><thead><tr>" + "".join(f"<th>{escape(name)}</th>" for name in headers) + "</tr></thead><tbody>" + cells + "</tbody></table>"


def _details(label, data):
    return f'<details><summary>{escape(label)}</summary><pre>{escape(json.dumps(data, indent=2))}</pre></details>'


def _marker_block(row):
    base = Path(row["relative_assets"])
    frames = "".join(_figure(base / f"{mode}.png", f"{mode} · {label}") for mode, label in MODES.items())
    crops = "".join(_figure(base / f"{mode}_crop.png", f"{mode} · marker sample") for mode in MODES)
    rows = []
    for mode in MODES:
        stats = row["image_metrics"][mode]
        fitted = stats["fitted_native_fog_factor"]
        rows.append((mode, "/".join(f"{c:.2f}" for c in stats["mean_rgb"]), f"{stats['mean_luma']:.3f}",
                     f"{stats['rgb_distance_from_fog']:.3f}", f"{stats['native_fog_factor']:.2%}",
                     "—" if fitted is None else f"{fitted:.2%}",
                     f"{stats['native_model_mean_abs_error']:.3f}"))
    return f"""<article><h3>{row['distance']:g} blocks · measured {row['measured_distance']:.6f}</h3>
<div class="triple">{frames}</div><div class="triple crops">{crops}</div>
{_table(('Mode', 'Mean RGB', 'Mean luma', 'RGB distance to fog', 'Native factor', 'RGB-fitted factor', 'Native equation MAE'), rows)}
{_details('Marker position, camera and runtime fog evidence', row)}</article>"""


def _scene_block(row):
    base = Path(row["relative_assets"])
    frames = "".join(_figure(base / f"{mode}.png", f"{mode} · {label}") for mode, label in MODES.items())
    diagnostics = "".join(_figure(base / f"{name}.png", label) for name, label in (
        ("A-C", "Fog off – proposed absolute RGB difference"), ("B-C", "Old – proposed absolute RGB difference"),
        ("surface_hit", "Saved SurfaceHit"), ("background", "Saved local BACKGROUND"),
        ("water_count", "Saved Water Count · log gray")))
    rows = []
    for mode in MODES:
        stats = row["image_metrics"][mode]
        rows.append((mode, f"{stats['mean_luma']:.3f}", f"{stats['median_luma']:.3f}",
                     f"{stats['p10_luma']:.3f}", f"{stats['near_black_fraction']:.2%}"))
    return f"""<article><h3>View {row['view_index'] + 1}</h3><div class="triple">{frames}</div>
{_table(('Mode', 'Mean luma', 'Median luma', 'P10 luma', 'Near-black'), rows)}
<details><summary>Saved masks and RGB differences</summary><div class="panels">{diagnostics}</div></details>
{_details('Camera, unchanged voxel data and actual native fog', row)}</article>"""


def _montage(root, rows):
    thumb = (320, 180)
    canvas = Image.new("RGB", (thumb[0] * 3, (thumb[1] + 24) * len(rows)), "#131a22")
    draw = ImageDraw.Draw(canvas)
    for line, row in enumerate(rows):
        for column, mode in enumerate(MODES):
            source = root / row["relative_assets"] / f"{mode}.png"
            with Image.open(source) as image:
                image = image.resize(thumb)
                x, y = thumb[0] * column, (thumb[1] + 24) * line
                canvas.paste(image, (x, y + 24))
                draw.text((x + 6, y + 5), f"{row['distance']:g} blocks - {mode}", fill="white")
    canvas.save(root / "calibration_montage.png")


def finish_report(output, marker_rows, scene_rows, metadata):
    """Publish one relative-link HTML report with measured calibration status."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted(marker_rows, key=lambda row: row["distance"])
    complete = tuple(row["distance"] for row in rows) == DISTANCES
    checks = [ok for row in rows for ok in row["checks"].values()]
    status = "PASS" if complete and checks and all(checks) else "FAIL"
    summary = dict(status=status, metadata=metadata, modes=MODES,
                   luma_definition="Rec.709 on encoded native RGB8, range 0–255",
                   calibration=rows, existing_scenes=scene_rows,
                   observation="Fixed 48³ cube with half extent 24, independent of native RGB range 30")
    write_json(root / "summary.json", summary)
    _montage(root, rows)
    groups = defaultdict(list)
    for row in sorted(scene_rows, key=lambda row: (row["scene_id"], row["view_index"])):
        groups[row["scene_id"], row["kind"]].append(row)
    table_rows = []
    for row in rows:
        stats = row["image_metrics"]["C"]
        table_rows.append((f"{row['distance']:g}", f"{row['measured_distance']:.6f}",
                           f"{row['expected_proposed_factor']:.0%}", f"{stats['native_fog_factor']:.1%}",
                           f"{stats['rgb_distance_from_fog']:.3f}",
                           "PASS" if all(row["checks"].values()) else "FAIL"))
    sections = "".join(f"<section><h2>{escape(kind.title())} · scene {scene:04d}</h2>" +
                       "".join(_scene_block(row) for row in views) + "</section>"
                       for (scene, kind), views in groups.items())
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vox2Pix native fog calibration</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#131a22;color:#e6edf5;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:24px}}article{{padding:16px;background:#1e2937;margin:18px 0}}section{{margin:32px 0}}
h1,h2,h3,p{{margin:8px 0}}a,summary{{color:#a7d7ff}}figure{{margin:0;min-width:0}}img{{display:block;width:100%;background:#111}}
.triple,.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.crops img{{height:120px;object-fit:contain;image-rendering:pixelated}}
figcaption{{font-size:.85rem;color:#c0cfdf}}table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{border-bottom:1px solid #405064;text-align:left;padding:5px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}summary{{cursor:pointer}}@media(max-width:780px){{.triple,.panels{{grid-template-columns:1fr}}main{{padding:10px}}table{{font-size:.7rem}}}}
</style><main><h1>Vox2Pix native fog calibration · {status}</h1>
<p>{len(rows)} measured marker distances · {len(scene_rows)} saved camera poses · <a href="summary.json">Measured JSON</a></p>
<p>A disables fog. B uses the native start ratio 0.8. C uses the native start ratio 0.6666667. All modes use viewing range 30, Gamma 3.0, global floor 0.0 and solid ambient 0.35.</p>
<p>The fog factor is linear in Euclidean eye distance. The pinned native shader also tints the fog target according to clarity. The RGB-fitted factor and native equation error account for that target. Whole-image metrics alone cannot establish local visibility.</p>
{_table(('Target distance', 'Measured distance', 'Expected C factor', 'Actual C factor', 'C RGB distance to fog', 'Checks'), table_rows)}
{_figure(Path('calibration_montage.png'), 'All distances · A / B / C')}
{''.join(_marker_block(row) for row in rows)}{sections}
{_details('Run metadata', metadata)}</main></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    return summary
