"""Focused report checks for drawn geometry hidden by independent distance fog."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from .fog48_report import DISTANCES, finish_report, write_existing, write_marker
from .fog_report import fog_factor, native_fog_target
from .storage import write_json
from .test_fog_report import Links


def marker(distance):
    baseline = np.full((8, 8, 3), [105, 40, 20], dtype="u1")
    color = [191, 215, 254, 255]
    images, probes = {}, {}
    for mode, enabled, draw in (("A", False, 48), ("B", True, 30), ("C", True, 48)):
        factor = fog_factor(distance, 20., 30., enabled)
        probes[mode] = dict(actual_fog_start=20., actual_fog_end=30., actual_fog_color=color,
                            fog_enabled=enabled, directional_colored_fog=True,
                            fog_type="linear", shader_uniforms_bound=True,
                            draw_control_wanted_range=draw, actual_map_draw_range=draw,
                            loaded_mapblocks_xyz=[[0, 0, 2]],
                            submitted_mapblocks_xyz=[[0, 0, 2]] if distance <= draw else [])
        images[mode] = np.rint(baseline * (1. - factor) + native_fog_target(color, factor) * factor).astype("u1")
    row = dict(distance=distance, measured_distance=float(distance), probes=probes,
               pixel_bbox=[3, 3, 5, 5], crop_bbox=[1, 1, 7, 7])
    return row, images


class Fog48ReportTests(unittest.TestCase):
    def test_far_geometry_must_be_submitted_even_when_the_rgb_is_fog_color(self):
        with TemporaryDirectory() as temp:
            row, images = marker(40)
            result = write_marker(Path(temp), row, images)
            self.assertEqual(result["image_metrics"]["C"]["rgb_distance_from_fog"], 0.)
            self.assertTrue(result["checks"]["marker_geometry_submitted"])
            self.assertFalse(result["marker_mapblock"]["submitted_in_B"])
            row["probes"]["C"]["submitted_mapblocks_xyz"] = []
            result = write_marker(Path(temp), row, images)
            self.assertFalse(result["checks"]["marker_geometry_submitted"])

    def test_relative_report_assets_and_separate_verification_status(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            markers = [write_marker(root, *marker(distance)) for distance in DISTANCES]
            source = root / "source"
            (source / "scenes").mkdir(parents=True)
            write_json(source / "node_vocabulary.json", ["air", "mcl_core:stone"])
            existing = []
            _, images = marker(24)
            for scene_id, kind in enumerate(("surface", "cave", "underwater")):
                path = source / "scenes" / f"scene_{scene_id:04d}"
                store = zarr.open_group(str(path / "voxels.zarr"), mode="w")
                store.array("surface_node_id", np.ones((5, 8, 8), dtype="u2"))
                store.array("water_count", np.zeros((5, 8, 8), dtype="u1"))
                for index in range(5):
                    row = dict(scene_id=scene_id, kind=kind, view_index=index)
                    existing.append(write_existing(root, row, {mode: images[mode] for mode in ("B", "C")}, path))
            result = finish_report(root, markers, existing, {"unit_verified": True, "fixture": True})
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["verification"]["native_renderer_verified"])
            self.assertFalse(result["verification"]["visual_calibration_verified"])
            self.assertFalse(result["verification"]["production_dataset_collected"])
            links = Links()
            links.feed((root / "index.html").read_text())
            self.assertTrue(links.images)
            self.assertTrue(all((root / path).is_file() for path in links.images))
            self.assertTrue((root / "surface" / "scene_0000_view_01" / "white.png").is_file())


if __name__ == "__main__":
    unittest.main()
