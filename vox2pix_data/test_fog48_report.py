"""Focused report checks for drawn geometry hidden by independent distance fog."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from .fog48_report import DISTANCES, finish_report, write_existing, write_marker
from .fog_report import fog_factor, native_fog_target
from .fog_runtime import validate_fog_absolute
from .protocol import V3_FOG48_GEOMETRY, V3_FOG48_PROFILE, V3_FOG48_VERSION, v3_rgb, v3_version
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
    def test_absolute_runtime_keeps_map_range_independent_from_fog_end(self):
        probe = dict(version="voxwm-native-fog-v2", render_tick=73, fog_enabled=True,
                     directional_colored_fog=True, fog_type="linear", viewing_range_config=48,
                     draw_control_wanted_range=48, actual_map_draw_range=48, runData_fog_range=48,
                     actual_fog_start_distance=20., actual_fog_end_distance=30.,
                     shader_fog_distance=30., shader_fog_shading_parameter=3.,
                     shader_uniforms_bound=True, loaded_mapblocks_xyz=[[0, 0, 2]],
                     submitted_mapblocks_xyz=[[0, 0, 2]], camera_far_plane=3000.,
                     max_block_send_distance_nodes=128, max_block_generate_distance_nodes=128)
        self.assertIs(validate_fog_absolute(probe, tick=73), probe)
        culled = dict(probe, viewing_range_config=30, draw_control_wanted_range=30,
                      actual_map_draw_range=30, runData_fog_range=30, shader_uniforms_bound=False,
                      submitted_mapblocks_xyz=[])
        self.assertIs(validate_fog_absolute(culled, viewing_range=30, tick=73), culled)
        for updates, reason in (
                ({"actual_fog_end_distance": 48}, "Unexpected native actual_fog_end_distance"),
                ({"actual_map_draw_range": 30}, "Unexpected native actual_map_draw_range"),
                ({"render_tick": 72}, "another RGB frame"),
                ({"max_block_generate_distance_nodes": 32}, "cannot cover draw range")):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, reason):
                validate_fog_absolute(dict(probe, **updates), tick=73)
        config = {"lighting_profile": V3_FOG48_PROFILE}
        self.assertEqual(v3_version(config), V3_FOG48_VERSION)
        self.assertEqual(V3_FOG48_GEOMETRY["voxel_observation_shape"], [48, 48, 48])
        self.assertEqual(V3_FOG48_GEOMETRY["voxel_observation_half_extent"], 24)
        self.assertEqual(v3_rgb(config)["viewing_range"], 48)
        self.assertEqual(v3_rgb(config)["fog_start_distance"], 20.)
        self.assertEqual(v3_rgb(config)["fog_end_distance"], 30.)
        self.assertNotIn("fog_start", v3_rgb(config))
        legacy = v3_rgb({"lighting_profile": "material-ambient"})
        self.assertEqual(legacy["viewing_range"], 30)
        self.assertEqual(legacy["fog_start"], .8)
        self.assertEqual(legacy["solid_ambient_floor"], .35)

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

    def test_culled_52_block_marker_is_background_without_a_fabricated_fog_fit(self):
        with TemporaryDirectory() as temp:
            row, images = marker(52)
            row["probes"]["C"]["shader_uniforms_bound"] = False
            images["C"][:] = [30, 60, 100]
            result = write_marker(Path(temp), row, images)
            self.assertTrue(result["checks"]["runtime_fog"])
            self.assertIsNone(result["checks"]["endpoint_hidden"])
            self.assertIsNone(result["checks"]["native_color_equation"])
            self.assertIsNone(result["image_metrics"]["C"]["native_fog_factor"])
            self.assertIsNone(result["image_metrics"]["C"]["fitted_native_fog_factor"])
            self.assertEqual(result["image_metrics"]["C"]["mean_rgb"], [30., 60., 100.])
            self.assertIn("INAPPLICABLE", result["image_metrics"]["C"]["measurement_status"])

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
