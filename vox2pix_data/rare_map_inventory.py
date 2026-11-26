"""Locate exact Vox2Pix-only node states in saved 48³ input maps, once per scene."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
from numcodecs import blosc


def targets_from_csv(path):
    with open(path) as file:
        return [
            [r["node_name"], int(r["raw_param2"]), int(r["vox2pix_48_voxels"])]
            for r in csv.DictReader(file)
            if int(r["vox2pix_48_voxels"]) and not int(r["vae_48_voxels"])
        ]


def histogram(codes):
    values, counts = np.unique(codes, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def capture_details(codes, meta, names):
    blocks = codes.reshape(3, 16, 3, 16, 3, 16).transpose(0, 2, 4, 1, 3, 5).reshape(27, 4096)
    descent = meta.get("underwater_descent", {})
    column = [names[int(code) // 256] for code in codes[24, 24]]
    low = meta["region_origin_enu"][2]
    # Terrain and liquids are a subset of the nodes accepted by VAE locate().
    liquids = {
        name for name, d in meta["water_definitions"].items() if d["liquidtype"] in ("source", "flowing")
    }
    qualifying = set(meta["terrain_nodes"]) | liquids | {"mcl_core:lava_source", "mcl_core:lava_flowing"}
    observed_top = max((low + j for j, name in enumerate(column) if name in qualifying), default=None)
    s = meta["surface_y"]
    return {
        "center_column": column,
        "observed_column_top": observed_top,
        "candidate_surface_lower_y": max(s, observed_top) if observed_top is not None else s,
        "block_counts": [histogram(block) for block in blocks],
        "survey_foot_xyz": descent.get("spawn_player_xyz", meta["foot_anchor_xyz"]),
    }


def scan(root, targets):
    names = json.loads((root / "node_vocabulary.json").read_text())
    expected = {names.index(name) * 256 + p: n for name, p, n in targets}
    wanted = np.zeros(len(names) * 256, dtype=bool)
    wanted[list(expected)] = True
    scenes = sorted((root / "scenes").glob("scene_*"))
    manifest = json.loads((root / "manifest.json").read_text())
    if len(scenes) != manifest["config"]["scenes"]:
        raise ValueError("Incomplete inventory of scene directories")
    air, ignore = names.index("air"), names.index("ignore")
    void = names.index("mcl_core:void")
    rows = []
    observed = Counter()
    blosc.set_nthreads(1)
    for i, scene in enumerate(scenes, 1):
        arrays = zarr.open_group(str(scene / "voxels.zarr"), mode="r")
        nodes = arrays["scene_node_id"][:]
        params = arrays["scene_param2"][:]
        if nodes.shape != (48, 48, 48) or params.shape != nodes.shape:
            raise ValueError(f"Incomplete 48³ input at {scene}")
        codes = nodes.astype(np.int32) * 256 + params
        hit = wanted[codes]
        if hit.any():
            m = json.loads((scene / "scene.json").read_text())
            counts = histogram(codes[hit])
            observed.update(counts)
            center = (slice(16, 32),) * 3
            origin = m["region_origin_enu"]
            foot = m["foot_anchor_xyz"]
            if [origin[0] + 24, origin[2] + 24, origin[1] + 24] != foot:
                raise ValueError(f"Survey and stored map centers differ in {scene}")
            rows.append(
                dict(
                    scene=scene.name,
                    **{
                        k: m[k]
                        for k in (
                            "world_seed",
                            "kind",
                            "surface_biome",
                            "foot_biome",
                            "anchor_kind",
                            "surface_y",
                            "region_origin_enu",
                            "foot_anchor_xyz",
                        )
                    },
                    center_air=int(np.sum(nodes[center] == air)),
                    center_void=int(np.sum(nodes[center] == void)),
                    ignore=int(np.sum(nodes == ignore)),
                    counts=histogram(codes),
                    hits=counts,
                    center_hits=histogram(codes[center][hit[center]]),
                    **capture_details(codes, m, names),
                )
            )
        if i % 5000 == 0:
            print(f"{i}/{len(scenes)} maps, {len(rows)} hits", file=sys.stderr, flush=True)
    if observed != expected:
        raise ValueError("Per-map hit counts differ from the full input-map inventory")
    keys = {(r["world_seed"], *r["region_origin_enu"]) for r in rows}
    if len(rows) != len(keys):
        raise ValueError("Duplicate physical maps among the hits")
    return {"total_maps": len(scenes), "names": names, "targets": expected, "maps": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--coverage-csv", type=Path)
    source.add_argument("--targets-json", help="Semantic targets for execution on a read-only dataset host")
    args = parser.parse_args()
    targets = targets_from_csv(args.coverage_csv) if args.coverage_csv else json.loads(args.targets_json)
    json.dump(scan(args.root, targets), sys.stdout, separators=(",", ":"))
