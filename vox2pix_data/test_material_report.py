"""Small deterministic checks for material-ambient diagnostics and offline assets."""

from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from .material_report import finish_report, region_masks, write_comparison
from .storage import EMPTY_NODE


class ImageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.sources.append(dict(attrs)["src"])


class MaterialReportTests(unittest.TestCase):
    def test_region_partition_is_exhaustive_and_ordered(self):
        surface = np.array([[0, 1], [EMPTY_NODE, EMPTY_NODE]], dtype=np.uint16)
        water = np.array([[0, 2], [1, 0]], dtype=np.uint8)
        groups = region_masks(surface, water)
        np.testing.assert_array_equal(np.stack(tuple(groups.values())).sum(axis=0), 1)
        self.assertEqual([np.flatnonzero(mask).tolist() for mask in groups.values()], [[0], [1], [2], [3]])

    def test_measures_and_relative_report_assets(self):
        surface = np.array([[0, 0, 1, 1], [EMPTY_NODE, EMPTY_NODE, 0, 1]], dtype=np.uint16)
        water = np.array([[0, 0, 1, 1], [3, 0, 0, 1]], dtype=np.uint8)
        positions = [0, 0, 0]
        pose = dict(actual_camera_world_xyz=positions, yaw_deg=0, pitch_deg=0,
                    fov_x_rad=1.57, fov_y_rad=1.25, gamma=3.0)
        rgb_a = np.zeros((2, 4, 3), dtype="u1")
        rgb_c = rgb_a.copy()
        rgb_c[0, 0] = 9
        rgb_c[0, 1] = 21
        images = {"A": rgb_a, "B": rgb_a, "C": rgb_c}
        with TemporaryDirectory() as temp:
            root = Path(temp) / "report"
            scene_path = Path(temp) / "scene"
            store = zarr.open_group(str(scene_path / "voxels.zarr"), mode="w")
            for name, values in (("surface_node_id", surface), ("water_count", water),
                                 ("surface_face_dir", np.zeros_like(water))):
                store.array(name, np.stack([values, values]))
            kwargs = (root, scene_path, {"width": 4, "height": 2}, ["mcl_core:stone", "mcl_core:diorite"])
            cave = write_comparison(*kwargs, dict(scene_id=0, kind="cave", world_seed=42, views=[pose]),
                                    0, images, {})
            underwater = write_comparison(*kwargs, dict(scene_id=1, kind="underwater", world_seed=43,
                                                         views=[pose, pose]), 1, images, {})
            finish_report(root, [cave, underwater], {"fixture": True})
            links = ImageLinks()
            links.feed((root / "index.html").read_text(encoding="utf-8"))
            self.assertEqual(len(links.sources), 18)
            self.assertTrue(all((root / path).is_file() for path in links.sources))
            self.assertAlmostEqual(cave["rgb_difference"]["A-C"]["mean"], 30 / 8)
            self.assertAlmostEqual(cave["image_metrics"]["C"]["near_black_fraction"], .75)
            self.assertEqual(cave["regions"]["water_background"]["pixel_count"], 1)
            self.assertEqual(cave["regions"]["water_background"]["rgb_difference"]["A-C"]["mean"], 0)
            stone = cave["cave_materials"]["materials"]["mcl_core:stone"]
            self.assertAlmostEqual(stone["within_material_face_luma_step"]["C"]["mean"], 12)
            self.assertAlmostEqual(underwater["underwater_view2"]["local_background_ratio"], .25)


if __name__ == "__main__":
    unittest.main()
