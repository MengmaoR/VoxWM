"""Measure saved v3 payloads without recompressing or changing source data."""
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from .qa import _sizes
from .protocol import v3_plan, v3_version
from .storage import RAW_FIELDS, V3_WHITE_FIELDS, array_schema, committed_paths, read_json, write_json


def compression_report(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    if config.get("mode") not in ("smoke", "production"):
        from .qa import compression_report as legacy_report

        return legacy_report(root)
    if manifest.get("version") != v3_version(config) or manifest.get("axes") != "ENU":
        raise ValueError("Unsupported Vox2Pix v3 dataset manifest")
    slots = {slot["scene_id"]: slot for slot in v3_plan(config)}
    paths = committed_paths(root)
    if not paths.keys() <= slots.keys():
        raise ValueError("Committed scenes do not belong to this dataset plan")
    fields = RAW_FIELDS | V3_WHITE_FIELDS
    raw, stored = Counter(), Counter()
    png_sizes = []
    fixed_metadata = sum((root / name).stat().st_size for name in ("manifest.json", "node_vocabulary.json")
                         if (root / name).is_file())
    metadata = fixed_metadata
    auxiliary = 0
    biomes, splits, kinds = Counter(), Counter(), Counter()
    for scene_id, path in paths.items():
        meta = read_json(path / "scene.json")
        if any(meta.get(key) != value for key, value in slots[scene_id].items()):
            raise ValueError(f"Scene {scene_id} differs from its scheduled slot")
        store = zarr.open_group(str(path / "voxels.zarr"), mode="r")
        arrays = {name: store[name] for name in store.array_keys()}
        array_schema(arrays, config)
        rgb_paths = [path / "rgb" / f"{index:03d}.png" for index in range(config["views_per_scene"])]
        if sorted((path / "rgb").iterdir()) != rgb_paths:
            raise ValueError(f"Scene {scene_id} does not contain its expected RGB frames")
        png_sizes.extend(p.stat().st_size for p in rgb_paths)
        auxiliary += sum(p.stat().st_size for p in (path / "lighting_comparison").glob("*.png"))
        metadata += (path / "scene.json").stat().st_size
        for file in (path / "voxels.zarr").rglob("*"):
            if file.is_file():
                if file.parent.name in fields and not file.name.startswith("."):
                    stored[file.parent.name] += file.stat().st_size
                else:
                    metadata += file.stat().st_size
        raw.update({name: arrays[name].nbytes for name in fields})
        kinds[meta["kind"]] += 1
        biomes[meta["surface_biome"]] += 1
        splits[meta.get("split", "not_recorded")] += 1

    count = len(png_sizes)
    if not count:
        raise ValueError("No committed scenes to measure; collect data before requesting a compression report")

    def group(names):
        result = {name: _sizes(raw[name], stored[name]) for name in names}
        result["total"] = _sizes(sum(raw[name] for name in names), sum(stored[name] for name in names))
        return result

    rgb = _sizes(config["width"] * config["height"] * 3 * count, sum(png_sizes))
    rgb.update(mean_bytes_per_frame=float(np.mean(png_sizes)),
               median_bytes_per_frame=float(np.median(png_sizes)), p95_bytes_per_frame=float(np.percentile(png_sizes, 95)))
    total = sum(stored.values()) + sum(png_sizes) + metadata + auxiliary
    scale = 50000 / count
    projected = round((total - fixed_metadata - auxiliary) * scale + fixed_metadata)
    result = dict(protocol_version=manifest["version"],
                  status="INCOMPLETE" if len(paths) != len(slots) else "COMPLETE",
                  scene_count=len(paths), frame_count=count,
                  expected_scene_count=len(slots),
                  expected_frame_count=len(slots) * config["views_per_scene"],
                  missing_scene_count=len(slots) - len(paths), kind_counts=dict(kinds),
                  rgb=rgb, white_model=group(V3_WHITE_FIELDS), raw_voxel_maps=group(RAW_FIELDS),
                  metadata=dict(actual_bytes=metadata, fixed_dataset_bytes=fixed_metadata),
                  auxiliary_lighting_pngs=dict(actual_bytes=auxiliary),
                  pilot_total_bytes=total, pilot_total_GB=total / 1e9, pilot_total_GiB=total / 2**30,
                  projected_50k_total_bytes=projected, projected_50k_GB=projected / 1e9,
                  projected_50k_GiB=projected / 2**30, projection_frame_scale=scale,
                  projection_note="Estimate from measured scenes; partial or unrepresentative coverage can bias it. Shared root metadata counted once; auxiliary lighting PNGs excluded from the production projection.",
                  coverage=dict(surface_biome_counts=dict(biomes), split_scene_counts=dict(splits)),
                  compression=dict(arrays="Lossless Blosc/Zstd", rgb="Lossless PNG", recompressed=False),
                  size_definition="Logical bytes of committed scene payloads, root manifest and vocabulary; excludes filesystem block padding, logs, generated reports and runtime progress. No archive was created.")
    write_json(root / "compression_report.json", result)
    (root / "compression_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(report):
    lines = ["# Vox2Pix v3 compression report", "",
             f"**{report['status']}**: {report['scene_count']}/{report['expected_scene_count']} scenes, "
             f"{report['frame_count']}/{report['expected_frame_count']} RGB/White Model pairs.", "",
             f"Protocol: `{report['protocol_version']}`. Sizes are read from files and Zarr metadata without decoding arrays or recompressing them.", "",
             "| Payload | Raw bytes | Stored bytes | Raw / stored |", "| --- | ---: | ---: | ---: |"]
    payloads = {"RGB PNG": report["rgb"]}
    for name in ("white_model", "raw_voxel_maps"):
        for field, values in report[name].items():
            payloads[f"{name} total" if field == "total" else field] = values
    for name, values in payloads.items():
        ratio = values["compression_ratio"]
        label = f"{ratio:.2f}×" if ratio is not None else "—"
        lines.append(f"| {name} | {values['raw_bytes']:,} | {values['actual_bytes']:,} | {label} |")
    lines.extend([f"| Metadata | — | {report['metadata']['actual_bytes']:,} | — |",
                  f"| Auxiliary lighting PNGs | — | {report['auxiliary_lighting_pngs']['actual_bytes']:,} | — |", "",
                  f"Measured dataset payload: **{report['pilot_total_GB']:.3f} GB / {report['pilot_total_GiB']:.3f} GiB**.",
                  f"50K-frame estimate (scale {report['projection_frame_scale']:g}× for per-scene payloads): "
                  f"**{report['projected_50k_GB']:.3f} GB / {report['projected_50k_GiB']:.3f} GiB**.",
                  report["projection_note"], "", report["size_definition"], "",
                  "## Coverage", "", "| Scene kind | Count |", "| --- | ---: |"])
    lines.extend(f"| {kind} | {count} |" for kind, count in report["kind_counts"].items())
    lines.extend(["",
                  "Biome and split counts are recorded in `compression_report.json`.",
                  "The RGB resolution remains 640×360. No legacy storage gate or automatic resolution recommendation is applied.", ""])
    return "\n".join(lines)
