"""Small synthetic checks of final32 measurements, regression and offline links."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zarr

from .final_report import DISTANCES, check_links, finish_report, write_existing, write_marker
from .fog_report import fog_factor, native_fog_target
from .fog_runtime import validate_fog_absolute
from .storage import EMPTY_NODE, write_json


def probe(*, enabled=True, submitted=True, block=None):
    blocks = [block or [0, 0, 1]]
    return dict(version="voxwm-native-fog-v2", fog_enabled=enabled, directional_colored_fog=True,
                fog_type="linear", viewing_range_config=32, draw_control_wanted_range=32,
                actual_map_draw_range=32, actual_fog_start_distance=20, actual_fog_end_distance=30,
                actual_fog_color=[191, 215, 254, 255], camera_far_plane=2000,
                max_block_send_distance_nodes=128, max_block_generate_distance_nodes=128,
                shader_uniforms_bound=submitted, shader_fog_distance=30, shader_fog_shading_parameter=3,
                loaded_mapblocks_xyz=blocks, submitted_mapblocks_xyz=blocks if submitted else [],
                material_lighting=dict(display_gamma=3., lighting_floor=0., solid_ambient_floor=.35,
                                       lut_zero=0, lut_max=255, shader_bound=True))


def marker(distance, submitted=True):
    block = [0, 0, (distance + 1) // 16]
    base = np.full((8, 8, 3), [105, 40, 20], dtype="u1")
    factor = fog_factor(distance, 20, 30)
    rgb = np.rint(base * (1 - factor) + native_fog_target([191, 215, 254], factor) * factor).astype("u1")
    if not submitted:
        base[:] = [30, 60, 100]
        rgb[:] = [30, 60, 100]
    return (dict(distance=distance, measured_distance=distance, pixel_bbox=[3, 3, 5, 5], crop_bbox=[1, 1, 7, 7],
                 marker_mapblock_xyz=block, camera_world_xyz=[0, 2, .5],
                 pose=dict(camera_world_xyz=[0, 2, .5], cam_dir_xyz=[0, 0, 1]),
                 probes={"A": probe(enabled=False, submitted=submitted, block=block),
                         "C": probe(submitted=submitted, block=block)}), {"A": base, "C": rgb})


class FinalReportTests(unittest.TestCase):
    def test_runtime_draw32_keeps_absolute_fog_and_far_plane(self):
        self.assertIsInstance(validate_fog_absolute(probe(), viewing_range=32), dict)
        for change in ({"actual_map_draw_range": 30}, {"actual_fog_end_distance": 32}, {"camera_far_plane": 32}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_fog_absolute(dict(probe(), **change), viewing_range=32)
        self.assertEqual([fog_factor(distance, 20, 30) for distance in DISTANCES],
                         [0, 0, 0, .2, .4, .5, .7, .9, 1, 1, 1, 1])

    def test_marker_requires_30_to_32_submission_and_reports_bounding_retention(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, images = marker(31, submitted=False)
            result = write_marker(root, row, images)
            self.assertFalse(result["checks"]["marker_geometry_submitted"])
            row, images = marker(34)
            result = write_marker(root, row, images)
            self.assertIsNone(result["checks"]["marker_geometry_submitted"])
            self.assertTrue(result["checks"]["endpoint_hidden"])
            self.assertIn("retain geometry", result["outside_range_assessment"])
            row, images = marker(34, submitted=False)
            result = write_marker(root, row, images)
            self.assertIsNone(result["checks"]["native_color_equation"])
            self.assertIsNone(result["checks"]["endpoint_hidden"])
            self.assertIsNone(result["image_metrics"]["C"]["native_fog_factor"])

    def test_saved_arrays_offline_assets_and_visual_status(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            markers = [write_marker(root, *marker(distance)) for distance in DISTANCES]
            source_root = root / "original"
            source_root.mkdir(exist_ok=True)
            write_json(source_root / "node_vocabulary.json", ["air", "stone"])
            scenes = []
            rgb = np.full((8, 8, 3), 90, dtype="u1")
            for scene_id, kind in enumerate(("surface", "cave", "underwater")):
                path = source_root / "scenes" / f"scene_{scene_id:04d}"
                path.mkdir(parents=True)
                write_json(path / "scene.json", dict(scene_id=scene_id, kind=kind))
                group = zarr.open_group(str(path / "voxels.zarr"), mode="w")
                group.array("scene_node_id", np.ones((4, 4, 4), dtype="u2"))
                group.array("scene_param2", np.zeros((4, 4, 4), dtype="u1"))
                surface = np.ones((5, 8, 8), dtype="u2")
                surface[:, :2] = EMPTY_NODE
                group.array("surface_node_id", surface)
                for field in ("surface_param2", "surface_face_dir", "water_count"):
                    group.array(field, np.zeros((5, 8, 8), dtype="u1"))
                for index in range(5):
                    scenes.append(write_existing(root, dict(scene_id=scene_id, kind=kind, view_index=index,
                        probes={"C": probe()}, evidence=dict(node_id_identical=True, raw_param2_identical=True,
                                                               camera_pose_identical=True)), {"C": rgb}, path))
            result = finish_report(root, markers, scenes, dict(unit_verified=True, fixture=True))
            self.assertEqual(result["status"], "UNIT_FIXTURE")
            self.assertFalse(result["verification"]["native_renderer_verified"])
            self.assertFalse(result["verification"]["visual_verified"])
            self.assertFalse(result["verification"]["dataset_collection_started"])
            self.assertTrue(result["verification"]["white_model_regression_verified"])
            self.assertTrue(check_links(root))
            self.assertTrue((root / "report" / "index.html").is_file())
            self.assertTrue((root / "calibration" / "index.html").is_file())
            (root / scenes[0]["relative_assets"] / "C.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing offline assets"):
                check_links(root)


if __name__ == "__main__":
    unittest.main()
