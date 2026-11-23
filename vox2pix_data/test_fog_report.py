"""Focused checks for native fog factors, measured residuals and offline assets."""

from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from .fog_report import DISTANCES, finish_report, fog_factor, native_fog_target, write_existing, write_marker
from .storage import EMPTY_NODE


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs)["src"])


class FogReportTests(unittest.TestCase):
    def test_native_linear_factor_and_directional_tint(self):
        self.assertEqual([fog_factor(d, 20., 30.) for d in (10, 20, 22, 24, 25, 27, 29, 30, 32)],
                         [0., 0., .2, .4, .5, .7, .9, 1., 1.])
        np.testing.assert_allclose(native_fog_target([191, 215, 254, 255], 1.), [191, 215, 254])
        tinted = native_fog_target([191, 215, 254, 255], .4)
        self.assertTrue(np.all(tinted <= [191, 215, 254]))
        self.assertEqual(fog_factor(32., 20., 30., False), 0.)

    def test_native_metrics_and_offline_report_reject_wrong_runtime_range(self):
        baseline = np.full((8, 8, 3), [105, 40, 20], dtype="u1")
        color = [191, 215, 254, 255]
        with TemporaryDirectory() as temp:
            output = Path(temp)
            rows = []
            for distance in DISTANCES:
                probes = {}
                images = {}
                for mode, enabled, start in (("A", False, 20.), ("B", True, 24.), ("C", True, 20.)):
                    factor = fog_factor(distance, start, 30., enabled)
                    probes[mode] = dict(actual_fog_start=start, actual_fog_end=30.,
                                        actual_fog_color=color, fog_enabled=enabled,
                                        directional_colored_fog=True, fog_type="linear", shader_uniforms_bound=True)
                    image = baseline * (1. - factor) + native_fog_target(color, factor) * factor
                    images[mode] = np.rint(image).astype("u1")
                row = dict(distance=distance, measured_distance=float(distance), pixel_bbox=[3, 3, 5, 5],
                           crop_bbox=[1, 1, 7, 7], probes=probes)
                rows.append(write_marker(output, row, images))
            self.assertAlmostEqual(rows[2]["image_metrics"]["C"]["fitted_native_fog_factor"], .4, places=2)
            self.assertEqual(rows[-1]["image_metrics"]["C"]["rgb_distance_from_fog"], 0.)
            scene = output / "source"
            store = zarr.open_group(str(scene / "voxels.zarr"), mode="w")
            surface = np.zeros((1, 8, 8), dtype="u2")
            surface[0, 1:3] = EMPTY_NODE
            water = np.zeros((1, 8, 8), dtype="u1")
            store.array("surface_node_id", surface)
            store.array("water_count", water)
            existing = write_existing(output, dict(scene_id=0, kind="surface", view_index=0), images, scene)
            self.assertEqual(existing["regions"]["direct_geometry"]["pixel_count"], 48)
            self.assertEqual(finish_report(output, rows, [existing], {"fixture": True})["status"], "PASS")
            links = Links()
            links.feed((output / "index.html").read_text())
            self.assertTrue(links.images)
            self.assertTrue(all((output / path).is_file() for path in links.images))
            rows[-1]["checks"]["runtime_fog"] = False
            self.assertEqual(finish_report(output, rows, [existing], {"fixture": True})["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
