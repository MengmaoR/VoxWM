"""Offline evidence for the final draw32, absolute Fog20–30 RGB profile."""

from html import escape
from html.parser import HTMLParser
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import zarr

from . import fog_report
from .fog_runtime import validate_fog_absolute
from .material_report import REC709
from .storage import EMPTY_NODE, RAW_FIELDS, V3_WHITE_FIELDS, read_json, read_vocabulary, write_json
from .v3_report import _node_colors


DISTANCES = (10, 18, 20, 22, 24, 25, 27, 29, 30, 31, 32, 34)
MODES = {"A": "Fog OFF · draw32", "C": "Final Fog20–30 · draw32"}


def _runtime(probe, enabled=True):
    try:
        validate_fog_absolute(probe, viewing_range=32, enabled=enabled)
    except ValueError:
        return False
    return True


def _material(probe):
    light = probe.get("material_lighting", {})
    return (light.get("display_gamma") == 3. and light.get("lighting_floor") == 0.
            and light.get("solid_ambient_floor") == .35 and light.get("lut_zero") == 0
            and light.get("lut_max") == 255 and light.get("shader_bound") is True)


def write_marker(output, row, images):
    """Measure actual A/C captures; outside-range sky is not a marker fragment."""
    if set(images) != set(MODES):
        raise ValueError("The final calibration requires exactly A and C")
    height, width, channels = images["A"].shape
    if channels != 3 or any(image.shape != (height, width, 3) or image.dtype != np.uint8
                            for image in images.values()):
        raise ValueError("Marker captures must be aligned native RGB8")
    result = dict(row)
    distance = float(row["measured_distance"])
    block = row["marker_mapblock_xyz"]
    relative = Path("calibration") / f"distance_{row['distance']:g}"
    folder = Path(output) / relative
    folder.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = box = fog_report._sample_box(row, height, width)
    crop = fog_report._sample_box(dict(pixel_bbox=row.get("crop_bbox", box)), height, width)
    baseline = images["A"][top:bottom, left:right].astype(float)
    baseline_submitted = block in row["probes"]["A"].get("submitted_mapblocks_xyz", [])
    metrics = {}
    for mode, image in images.items():
        probe = row["probes"][mode]
        current = image[top:bottom, left:right].astype(float)
        color = np.asarray(probe["actual_fog_color"][:3], dtype=float)
        factor = fog_report.fog_factor(distance, 20, 30, probe["fog_enabled"])
        submitted = block in probe.get("submitted_mapblocks_xyz", [])
        applicable = submitted and baseline_submitted
        target = fog_report.native_fog_target(color, factor)
        expected = baseline * (1 - factor) + target * factor
        span = target - baseline
        denominator = float(np.square(span).sum())
        metrics[mode] = dict(mean_rgb=current.mean(axis=(0, 1)).tolist(),
                             mean_luma=float((current @ REC709).mean()),
                             rgb_distance_from_fog=float(np.abs(current - color).mean()),
                             theoretical_fog_factor=factor,
                             native_fog_factor=factor if applicable else None,
                             native_model_mean_abs_error=float(np.abs(current - expected).mean()) if applicable else None,
                             fitted_native_fog_factor=float(((current - baseline) * span).sum() / denominator)
                             if applicable and denominator > 1e-9 else None,
                             marker_texture_std=float(current.std(axis=(0, 1)).mean()) if submitted else None,
                             marker_geometry_submitted=submitted,
                             measurement_status="MEASURED" if applicable else "INAPPLICABLE — marker not submitted")
        Image.fromarray(image).save(folder / f"{mode}.png")
        Image.fromarray(image[crop[1]:crop[3], crop[0]:crop[2]]).save(folder / f"{mode}_crop.png")
    proposed = metrics["C"]
    submitted = proposed["marker_geometry_submitted"]
    equation = proposed["native_model_mean_abs_error"]
    result.update(relative_assets=relative.as_posix(), image_metrics=metrics, pixel_bbox=list(box),
                  checks=dict(runtime_fog=all(_runtime(row["probes"][mode], mode == "C") for mode in MODES),
                              material_ambient=all(_material(row["probes"][mode]) for mode in MODES),
                              marker_geometry_submitted=submitted and baseline_submitted if distance <= 32 else None,
                              native_color_equation=equation <= 3 if equation is not None else None,
                              endpoint_hidden=proposed["rgb_distance_from_fog"] <= 3 if distance >= 30 and submitted else None),
                  marker_mapblock=dict(xyz=block, loaded_in_C=block in row["probes"]["C"].get("loaded_mapblocks_xyz", []),
                                       submitted_in_A=baseline_submitted, submitted_in_C=submitted),
                  outside_range_assessment=("MapBlock bounds retain geometry beyond nominal32; inspect fully fogged pixels."
                                            if distance > 32 and submitted else
                                            "Outside nominal32; no marker geometry submitted." if distance > 32 else None))
    return result


def write_existing(output, row, images, source_scene_path):
    """Preserve the saved source arrays verbatim and render only the new RGB."""
    if set(images) != {"C"} or images["C"].dtype != np.uint8:
        raise ValueError("Existing-scene export requires the final native RGB8 image C")
    source = Path(source_scene_path)
    root = Path(output)
    relative = Path(row["kind"]) / f"scene_{row['scene_id']:04d}_view_{row['view_index'] + 1:02d}"
    folder = root / relative
    folder.mkdir(parents=True, exist_ok=True)
    data_folder = root / row["kind"] / "saved_scene"
    if not data_folder.exists():
        data_folder.mkdir()
        shutil.copytree(source / "voxels.zarr", data_folder / "voxels.zarr")
        shutil.copy2(source / "scene.json", data_folder / "scene.json")
    before = zarr.open_group(str(source / "voxels.zarr"), mode="r")
    after = zarr.open_group(str(data_folder / "voxels.zarr"), mode="r")
    equal = {field: bool(np.array_equal(before[field][:], after[field][:])) for field in RAW_FIELDS | V3_WHITE_FIELDS}
    if not all(equal.values()):
        raise ValueError(f"Saved-source regression failed: {equal}")
    index = row["view_index"]
    surface, water = (after[field][index] for field in ("surface_node_id", "water_count"))
    if images["C"].shape != (*surface.shape, 3) or water.shape != surface.shape:
        raise ValueError("RGB and original White Model dimensions differ")
    vocabulary = read_vocabulary(source.parent.parent)
    Image.fromarray(images["C"]).save(folder / "C.png")
    Image.fromarray(_node_colors(vocabulary)[surface]).save(folder / "white.png")
    Image.fromarray(np.rint(np.log1p(water.astype(float)) * 255 / np.log(256)).astype("u1")).save(folder / "water_count.png")
    background = surface == EMPTY_NODE
    Image.fromarray(background.astype("u1") * 255).save(folder / "background.png")
    exported_background = np.asarray(Image.open(folder / "background.png")) != 0
    result = dict(row, relative_assets=relative.as_posix(),
                  image_metrics={"C": fog_report.image_measures(images["C"])},
                  source_scene_metadata=read_json(source / "scene.json"),
                  regression=dict(saved_source_arrays_identical=equal,
                                  background_identical=bool(np.array_equal(background, exported_background)),
                                  method="Verbatim copies of saved arrays; White Model is not recomputed"),
                  source_data_copy=(data_folder.relative_to(root)).as_posix())
    write_json(folder / "frame.json", result)
    return result


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "img":
            self.paths.append(attributes["src"])
        elif tag == "a" and attributes.get("href"):
            self.paths.append(attributes["href"])


def check_links(root):
    missing = []
    for page in Path(root).rglob("*.html"):
        links = _Links()
        links.feed(page.read_text(encoding="utf-8"))
        for target in links.paths:
            if not target.startswith(("#", "https://", "http://")) and not (page.parent / target).is_file():
                missing.append(f"{page.relative_to(root)}: {target}")
    if missing:
        raise ValueError("Missing offline assets: " + ", ".join(missing))
    return True


def _page(title, contents):
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>body{{background:#131a22;color:#e6edf5;font:15px/1.5 system-ui;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}
a,summary{{color:#a7d7ff}}article{{background:#1e2937;padding:16px;margin:20px 0}}figure{{margin:0;min-width:0}}img{{width:100%;display:block}}
.panels{{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}}.crops img{{height:130px;object-fit:contain;image-rendering:pixelated}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}td,th{{text-align:left;padding:6px;border-bottom:1px solid #43556a}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media(max-width:780px){{.panels{{grid-template-columns:1fr}}}}</style><main>{contents}</main></html>'''


def _marker_section(row, prefix):
    base = Path(prefix) / row["relative_assets"]
    images = "".join(fog_report._figure(base / f"{mode}.png", MODES[mode]) for mode in MODES)
    crops = "".join(fog_report._figure(base / f"{mode}_crop.png", f"{mode} · marker crop") for mode in MODES)
    table = []
    for mode in MODES:
        metrics = row["image_metrics"][mode]
        table.append((mode, "/".join(f"{value:.2f}" for value in metrics["mean_rgb"]),
                      f"{metrics['mean_luma']:.2f}", f"{metrics['rgb_distance_from_fog']:.2f}",
                      str(metrics["marker_geometry_submitted"]), metrics["measurement_status"]))
    return (f"<article><h3>{row['distance']:g} blocks · measured {row['measured_distance']:.6f}</h3>"
            f'<div class="panels">{images}</div><div class="panels crops">{crops}</div>' +
            fog_report._table(("Mode", "Mean RGB", "Rec.709 luma", "RGB distance to Fog Color", "Submitted", "Measurement"), table) +
            fog_report._details("Fixed camera, marker coordinates and native probes", row) + "</article>")


def _montage(root, rows):
    canvas = Image.new("RGB", (640, 204 * len(rows)), "#131a22")
    draw = ImageDraw.Draw(canvas)
    for line, row in enumerate(rows):
        for column, mode in enumerate(MODES):
            x, y = column * 320, line * 204
            draw.text((x + 5, y + 5), f"{row['distance']:g} blocks · {mode}", fill="white")
            with Image.open(root / row["relative_assets"] / f"{mode}.png") as image:
                canvas.paste(image.resize((320, 180)), (x, y + 24))
    canvas.save(root / "calibration" / "montage.png")


def finish_report(output, marker_rows, scene_rows, metadata):
    """Native evidence and explicit human visual review have separate statuses."""
    root = Path(output)
    (root / "report").mkdir(parents=True, exist_ok=True)
    (root / "calibration").mkdir(exist_ok=True)
    markers = sorted(marker_rows, key=lambda row: row["distance"])
    scenes = sorted(scene_rows, key=lambda row: (row["scene_id"], row["view_index"]))
    expected_markers = tuple(row["distance"] for row in markers) == DISTANCES
    fixed_camera = bool(markers) and all(
        np.allclose(row["camera_world_xyz"], markers[0]["camera_world_xyz"], rtol=0, atol=1e-6)
        and row["pose"] == markers[0]["pose"] for row in markers)
    frame_counts = all({row["view_index"] for row in scenes if row["kind"] == kind} == set(range(5))
                       for kind in ("surface", "cave", "underwater")) and len(scenes) == 15
    calibration_ok = (expected_markers and fixed_camera and
                      all(value is not False for row in markers for value in row["checks"].values()))
    regression_ok = frame_counts and all(
        all(row["regression"]["saved_source_arrays_identical"].values())
        and row["regression"]["background_identical"]
        and all(row.get("evidence", {}).get(key) is True
                for key in ("node_id_identical", "raw_param2_identical", "camera_pose_identical")) for row in scenes)
    native_ok = bool(calibration_ok and frame_counts and all(
        _runtime(row["probes"]["C"]) and _material(row["probes"]["C"]) for row in scenes))
    fixture = bool(metadata.get("fixture", False))
    visual = bool(metadata.get("visual_verified", False)) and not fixture
    verification = dict(unit_verified=bool(metadata.get("unit_verified", False)),
                        native_renderer_verified=native_ok and not fixture, white_model_regression_verified=bool(regression_ok),
                        visual_verified=visual, dataset_collection_started=False)
    status = ("UNIT_FIXTURE" if fixture else "PASS" if native_ok and regression_ok and visual
              and verification["unit_verified"] else "NATIVE_PASS_VISUAL_PENDING" if native_ok and regression_ok else "FAIL")
    summary = dict(status=status, verification=verification, metadata=metadata,
                   fixed_calibration_camera=fixed_camera, calibration=markers, existing_scenes=scenes,
                   final_configuration=dict(voxel_observation=[48, 48, 48], voxel_observation_half_extent=24,
                       rgb_viewing_range=32, fog_start_distance=20, fog_end_distance=30, fog_type="native_linear",
                       display_gamma=3., lighting_floor=0., solid_ambient_floor=.35),
                   outside_range_observations=[dict(distance=row["distance"], **row["marker_mapblock"])
                                               for row in markers if row["distance"] > 32],
                   luma_definition="Rec.709 on display-encoded native RGB8, range 0–255")
    write_json(root / "manifest.json", metadata)
    write_json(root / "summary.json", summary)
    _montage(root, markers)
    description = ("<p>The camera stays fixed. One colored native panel is placed at each measured distance in sequence, "
                   "then removed before the next distance, so near markers cannot obscure distant ones. "
                   "Each panel is rendered with Fog OFF and the final absolute Fog20–30 at draw32.</p>"
                   "<p>Native MapBlock bounding-volume culling can retain geometry beyond nominal32. Such retention is reported, "
                   "and a retained marker must be fully hidden by fog. Sky-only pixels have no marker-fragment fog equation.</p>")
    calibration_html = "".join(_marker_section(row, "..") for row in markers)
    (root / "calibration" / "index.html").write_text(_page("Final Fog Calibration", "<h1>Fixed-camera native fog calibration</h1>" + description +
        fog_report._figure(Path("montage.png"), "All 12 distances · Fog OFF / final Fog20–30") + calibration_html), encoding="utf-8")
    probes = [row["probes"]["C"] for row in markers + scenes]
    fields = ("viewing_range_config", "draw_control_wanted_range", "actual_map_draw_range", "actual_fog_start_distance",
              "actual_fog_end_distance", "camera_far_plane", "max_block_send_distance_nodes", "max_block_generate_distance_nodes")
    runtime_table = fog_report._table(("Actual native measurement · node units", "Observed values"),
        ((field, ", ".join(str(value) for value in sorted({p.get(field) for p in probes if p.get(field) is not None}))) for field in fields))
    scene_html = []
    for kind in ("surface", "cave", "underwater"):
        scene_html.append(f"<h2>{kind.title()}</h2>")
        for row in scenes:
            if row["kind"] != kind:
                continue
            base = Path("..") / row["relative_assets"]
            panels = "".join(fog_report._figure(base / f"{name}.png", label) for name, label in (
                ("C", "Final native RGB"), ("white", "Original White Model · semantic node colors"),
                ("water_count", "Original Water Count · log gray"), ("background", "Original BACKGROUND mask")))
            scene_html.append(f"<article><h3>View {row['view_index'] + 1}</h3><div class='panels'>{panels}</div>" +
                              fog_report._details("Original camera, scene metadata and native runtime", row) + "</article>")
    verification_html = fog_report._table(("Verification", "Result"), ((key, str(value)) for key, value in verification.items()))
    conclusions = dict(native_draw32=native_ok, fog_start20=native_ok, fog_end30=native_ok,
                       geometry_30_to_32_submitted=all(row["marker_mapblock"]["submitted_in_C"] for row in markers if 30 <= row["distance"] <= 32),
                       mapblock_culling_beyond32="See outside_range_observations; no per-voxel hard clipping is added",
                       no_hard_cut_at24_or30=bool(calibration_ok), material_ambient=native_ok,
                       native_underwater_rendering="Original shaders and post-effect retained; visual review remains explicit")
    contents = (f"<h1>Vox2Pix · final RGB32 / Fog20–30</h1><p>Result: {status}</p>" + verification_html +
                "<h2>Final configuration</h2>" + fog_report._details("Locked configuration", summary["final_configuration"]) +
                "<h2>Native runtime verification</h2>" + runtime_table +
                "<h2>Fog calibration</h2><p><a href='../calibration/index.html'>Full fixed-camera calibration</a> · "
                "<a href='../summary.json'>Machine-readable measurements</a> · <a href='../manifest.json'>Run manifest</a></p>" + description +
                fog_report._figure(Path("../calibration/montage.png"), "All 12 measured distances") +
                "".join(scene_html) + "<h2>Final summary</h2>" + fog_report._details("Measured conclusions and limits", conclusions) +
                fog_report._details("Run metadata", metadata))
    (root / "report" / "index.html").write_text(_page("Vox2Pix final native rendering32", contents), encoding="utf-8")
    (root / "index.html").write_text(_page("Vox2Pix final32", "<h1>Final RGB32</h1><a href='report/index.html'>Open native rendering report</a>"), encoding="utf-8")
    summary["offline_links_verified"] = check_links(root)
    write_json(root / "summary.json", summary)
    return summary
