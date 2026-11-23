"""Offline evidence for independent 48-block RGB drawing and 20–30 fog."""

from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import zarr

from . import fog_report
from .material_report import delta_stats
from .storage import EMPTY_NODE, read_vocabulary, write_json
from .v3_report import _node_colors


DISTANCES = (12, 18, 20, 22, 24, 27, 29, 30, 32, 36, 40, 44, 48, 52)
MODES = {"A": "Fog off · draw 48", "B": "Fog 20–30 · draw 30", "C": "Fog 20–30 · draw 48"}
PANEL_BLOCK = [0, 0, 2]


def write_marker(output, row, images):
    """Reuse native RGB measurements and test actual map submission separately."""
    result = fog_report.write_marker(output, row, images)
    probe = result["probes"]["C"]
    runtime = result["checks"]["runtime_fog"]
    range_ok = (abs(probe.get("draw_control_wanted_range", -1) - 48.) <= .001
                and abs(probe.get("actual_map_draw_range", -1) - 48.) <= .001)
    loaded = PANEL_BLOCK in probe.get("loaded_mapblocks_xyz", [])
    submitted = PANEL_BLOCK in probe.get("submitted_mapblocks_xyz", [])
    result["checks"].update(runtime_fog=runtime and range_ok,
                            marker_geometry_submitted=(result["distance"] > 48 or submitted))
    result["marker_mapblock"] = dict(xyz=PANEL_BLOCK, loaded_in_C=loaded, submitted_in_C=submitted,
                                      submitted_in_B=PANEL_BLOCK in result["probes"]["B"].get(
                                          "submitted_mapblocks_xyz", []))
    result["b_c_marker_rgb_mean_abs_difference"] = float(np.abs(
        np.array(result["image_metrics"]["B"]["mean_rgb"]) -
        np.array(result["image_metrics"]["C"]["mean_rgb"])).mean())
    return result


def write_existing(output, row, images, source_scene_path):
    """Compare the actual B/C pair with saved White Model and Water Count."""
    if set(images) != {"B", "C"}:
        raise ValueError("Existing scene replay requires exactly old B and proposed C")
    before, after = images["B"], images["C"]
    if before.shape != after.shape or before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("Old and proposed native RGB must be aligned uint8 frames")
    result = dict(row)
    relative = Path(row["kind"]) / f"scene_{row['scene_id']:04d}_view_{row['view_index'] + 1:02d}"
    folder = Path(output) / relative
    folder.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(Path(source_scene_path) / "voxels.zarr"), mode="r")
    surface = store["surface_node_id"][row["view_index"]]
    water = store["water_count"][row["view_index"]]
    if surface.shape != before.shape[:2] or water.shape != surface.shape:
        raise ValueError("Saved White Model shape differs from its matching RGB")
    vocabulary = read_vocabulary(Path(source_scene_path).parent.parent)
    white = _node_colors(vocabulary)[surface]
    delta = np.abs(before.astype(np.int16) - after).mean(axis=2)
    for label, image in images.items():
        Image.fromarray(image).save(folder / f"{label}.png")
    for label, image in (("B-C", np.rint(delta).astype("u1")), ("white", white),
                         ("water_count", np.rint(np.log1p(water.astype(float)) * 255 / np.log(256)).astype("u1"))):
        Image.fromarray(image).save(folder / f"{label}.png")
    groups = fog_report.region_masks(surface, water)
    result.update(relative_assets=relative.as_posix(),
                  image_metrics={mode: fog_report.image_measures(image) for mode, image in images.items()},
                  rgb_difference=delta_stats(delta),
                  regions={name: dict(pixel_count=int(mask.sum()),
                                       rgb_difference=delta_stats(delta[mask])) for name, mask in groups.items()},
                  surface_hit_ratio=float((surface != EMPTY_NODE).mean()),
                  saved_white_data_used=True)
    return result


def _montage(root, rows):
    canvas = Image.new("RGB", (960, 204 * len(rows)), "#131a22")
    text = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        for column, mode in enumerate(MODES):
            x, y = column * 320, index * 204
            text.text((x + 5, y + 5), f"{row['distance']:g} blocks - {mode}", fill="white")
            with Image.open(root / row["relative_assets"] / f"{mode}.png") as source:
                canvas.paste(source.resize((320, 180)), (x, y + 24))
    canvas.save(root / "calibration_montage.png")


def _marker_section(row):
    base = Path(row["relative_assets"])
    full = "".join(fog_report._figure(base / f"{mode}.png", f"{mode} · {label}") for mode, label in MODES.items())
    crops = "".join(fog_report._figure(base / f"{mode}_crop.png", f"{mode} · marker crop") for mode in MODES)
    table = []
    for mode in MODES:
        metrics = row["image_metrics"][mode]
        probe = row["probes"][mode]
        submitted = PANEL_BLOCK in probe.get("submitted_mapblocks_xyz", [])
        table.append((mode, f"{probe['actual_map_draw_range']:g}",
                      f"{probe['actual_fog_start']:.4f} / {probe['actual_fog_end']:.4f}",
                      f"{metrics['native_fog_factor']:.1%}",
                      "/".join(f"{value:.2f}" for value in metrics["mean_rgb"]),
                      f"{metrics['rgb_distance_from_fog']:.3f}", str(submitted)))
    return (f"<article><h3>{row['distance']:g} blocks · measured {row['measured_distance']:.6f}</h3>"
            f'<div class="triple">{full}</div><div class="triple crops">{crops}</div>' +
            fog_report._table(("Mode", "Native draw range", "Actual fog start / end", "Fog factor", "Marker mean RGB",
                               "RGB distance to fog", "Panel MapBlock submitted"), table) +
            fog_report._details("Marker and native renderer evidence", row) + "</article>")


def _scene_section(row):
    base = Path(row["relative_assets"])
    images = "".join(fog_report._figure(base / f"{mode}.png", MODES[mode]) for mode in ("B", "C"))
    previews = "".join(fog_report._figure(base / f"{field}.png", label) for field, label in (
        ("B-C", "Absolute RGB difference · fixed 0–255 scale"), ("white", "Saved single-surface node preview"),
        ("water_count", "Saved Water Count · log gray")))
    return (f"<article><h3>View {row['view_index'] + 1}</h3>"
            f'<div class="pair">{images}</div><div class="triple">{previews}</div>' +
            fog_report._details("Camera pose, voxel identity and native fog", row) + "</article>")


def finish_report(output, marker_rows, scene_rows, metadata):
    """Keep unit, native measurements and human visual review separately labeled."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    markers = sorted(marker_rows, key=lambda row: row["distance"])
    expected_distances = tuple(row["distance"] for row in markers) == DISTANCES
    native = expected_distances and all(all(row["checks"].values()) for row in markers)
    compared = len(scene_rows) == 15 and {row["kind"] for row in scene_rows} == {"surface", "cave", "underwater"}
    status = "PASS" if native and compared else "FAIL"
    verification = dict(unit_verified=metadata.get("unit_verified", False),
                        native_renderer_verified=bool(native),
                        visual_calibration_verified=metadata.get("visual_calibration_verified", False),
                        production_dataset_collected=False)
    summary = dict(status=status, verification=verification, metadata=metadata, modes=MODES,
                   calibration=markers, existing_scenes=scene_rows,
                   voxel_observation_shape=[48, 48, 48], voxel_observation_half_extent=24,
                   rgb_viewing_range=48, fog_start_distance=20, fog_end_distance=30,
                   fog_type="linear", luma_definition="Rec.709 on native RGB8, range 0–255")
    write_json(root / "summary.json", summary)
    _montage(root, markers)
    cells = []
    for row in markers:
        probe = row["probes"]["C"]
        stats = row["image_metrics"]["C"]
        block = row["marker_mapblock"]
        cells.append((f"{row['distance']:g}", f"{row['measured_distance']:.6f}",
                      f"{stats['native_fog_factor']:.1%}", f"{stats['mean_luma']:.3f}",
                      f"{stats['rgb_distance_from_fog']:.3f}",
                      f"{probe['actual_map_draw_range']:g}", str(block["submitted_in_B"]), str(block["submitted_in_C"])))
    groups = defaultdict(list)
    for row in sorted(scene_rows, key=lambda row: (row["scene_id"], row["view_index"])):
        groups[row["kind"]].append(row)
    sections = "".join(f"<section><h2>{escape(kind.title())}</h2>" +
                       "".join(_scene_section(row) for row in groups[kind]) + "</section>"
                       for kind in ("surface", "cave", "underwater"))
    verification_table = fog_report._table(("Check", "Verified"),
                                            ((escape(key.replace("_", " ")), str(value))
                                             for key, value in verification.items()))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Vox2Pix RGB range 48</title>
<style>body{{background:#131a22;color:#e6edf5;font:15px/1.5 system-ui;margin:0}}main{{max-width:1600px;margin:auto;padding:24px}}
a,summary{{color:#a7d7ff}}article{{background:#1e2937;padding:16px;margin:20px 0}}section{{margin:32px 0}}
figure{{margin:0;min-width:0}}img{{width:100%;display:block}}.triple,.pair{{display:grid;gap:12px;grid-template-columns:repeat(3,minmax(0,1fr))}}
.pair{{grid-template-columns:repeat(2,minmax(0,1fr))}}.crops img{{height:120px;object-fit:contain;image-rendering:pixelated}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}td,th{{text-align:left;padding:5px;border-bottom:1px solid #43556a}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}summary{{cursor:pointer}}figcaption{{color:#c4d0df}}
@media(max-width:780px){{.triple,.pair{{grid-template-columns:1fr}}main{{padding:10px}}table{{font-size:.7rem}}}}</style>
<main><h1>Vox2Pix · Draw range 48 with independent fog 20–30</h1>
<p>Native measurement checks {status} · <a href="summary.json">Measured JSON</a></p>{verification_table}
<p>A draws to 48 with fog disabled. B draws to 30 and C draws to 48. B and C share absolute fog distances 20–30, Gamma 3.0, global light floor 0 and solid ambient 0.35.</p>
<p>The calibrated native panel is the sole world geometry. Its MapBlock submission proves geometry reaches the renderer even when fog conceals its color. The 52-block case records native MapBlock bounding-volume leakage. It is excluded from a strict nominal-range submission requirement.</p>
{fog_report._table(('Target blocks', 'Measured blocks', 'C fog blend', 'C luma', 'C RGB distance to fog', 'C draw range', 'B submitted', 'C submitted'), cells)}
{fog_report._figure(Path('calibration_montage.png'), 'All measured distances · A / B / C')}
<p>Fog factor is linear in native Euclidean camera distance. Native directional fog also tints the fog target with clarity. RGB residuals use that existing Shader equation. Saved White Model previews cover the stored 48³ volume only.</p>
{''.join(_marker_section(row) for row in markers)}{sections}{fog_report._details('Run metadata', metadata)}</main></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    return summary
