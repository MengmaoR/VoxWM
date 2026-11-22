"""Lightweight fixtures. No Craftium process or native data is used."""
from collections import Counter
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from .geometry import Projector, build_library
from .protocol import (
    BIOMES, SETTINGS, RejectedScene, camera_angles, direction_for_angles, plan, region_id,
    sample_views, scene_visibility, site_candidates,
)


DEFS = {
    0: {"name": "air", "drawtype": "airlike"},
    1: {"name": "stone", "walkable": True, "terrain": True},
    2: {"name": "glass", "drawtype": "glasslike"},
    3: {"name": "water", "drawtype": "liquid", "water": True},
    4: {"name": "slab", "drawtype": "nodebox"},
}
EMPTY = 65535


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.projector = Projector(build_library(Path(temp.name)))

    def project(self, nodes, boxes=None, direction=(0., 0., 1.)):
        params = nodes.astype("u1") * 7
        scene = self.projector.scene(nodes, params, DEFS, boxes or {})
        return tuple(field[:, 0, 0].tolist() for field in
                     scene.render((0., 2., 0.), direction, 1.2, 1.2, 1, 1))

    def test_single_wall(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[:, 26, :] = 1
        self.assertEqual(self.project(nodes), ([1, EMPTY, EMPTY], [7, 0, 0], [5, 255, 255]))

    def test_separated_and_adjacent_distinct_voxels(self):
        for locations in ([26, 29, 34], [26, 27, 28]):
            with self.subTest(locations=locations):
                nodes = np.zeros((48, 48, 48), dtype=np.uint16)
                nodes[24, locations, 26] = 1
                self.assertEqual(self.project(nodes), ([1, 1, 1], [7, 7, 7], [5, 5, 5]))

    def test_transparent_layers(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, [28, 29], 26] = [2, 1]
        self.assertEqual(self.project(nodes)[0], [2, 1, EMPTY])
        nodes[24, 27, 26] = 3
        self.assertEqual(self.project(nodes), ([3, 2, 1], [21, 14, 7], [5, 5, 5]))

    def test_nodebox_misses_empty_half_of_cell(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 26, 26] = 4
        boxes = {int(np.ravel_multi_index((24, 26, 26), nodes.shape)):
                 [[-.5, -.5, -.5, .5, -.25, .5]]}
        self.assertEqual(self.project(nodes, boxes)[0], [EMPTY] * 3)
        self.assertEqual(self.project(nodes, boxes, (0., -.19, 1.)), ([4, EMPTY, EMPTY], [28, 0, 0], [5, 255, 255]))

    def test_pixel_axes(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 26, 27] = 1
        nodes[24, 26, 25] = 2
        nodes[23, 26, 26] = 3
        nodes[25, 26, 26] = 1
        scene = self.projector.scene(nodes, nodes.astype("u1"), DEFS, {})
        node, _, face = scene.render((0, 2, 0), (0, 0, 1), math.pi / 2, math.pi / 2, 3, 3)
        self.assertEqual(node[0, :, 1].tolist(), [1, EMPTY, 2])
        self.assertEqual(node[0, 1, :].tolist(), [3, EMPTY, 1])
        self.assertEqual(int(face[0, 1, 0]), 5)


class SamplingTests(unittest.TestCase):
    def test_integer_origin_native_capture_layout(self):
        from .native import RAW, decode_region

        raw = np.zeros((3, 3, 3, 16, 16, 16), dtype=RAW)
        raw[2, 0, 1, 5, 6, 7] = (1234, 221, 17)
        observed = decode_region(raw.tobytes())
        self.assertEqual(observed[37, 6, 23]["node"], 1234)
        self.assertEqual(observed[37, 6, 23]["param2"], 17)
        self.assertEqual(np.count_nonzero(observed["node"]), 1)

    def test_seeded_quotas_and_region_angles(self):
        slots = plan(42)
        self.assertEqual(len(slots), 200)
        self.assertEqual(set(Counter(slot["biome"] for slot in slots).values()), {8})
        self.assertEqual(Counter(slot["height"] for slot in slots), {"surface": 120, "shallow": 40, "deep": 40})
        self.assertNotEqual(slots, plan(43))
        views = sample_views([False] * 20 + [True] * 20, np.random.default_rng(42))
        self.assertEqual(len({v["region_id"] for v in views}), 5)
        for view in views:
            actual = camera_angles(direction_for_angles(view["requested_yaw_deg"], view["requested_pitch_deg"]))
            self.assertEqual(region_id(*actual), view["region_id"])
            self.assertGreaterEqual(view["region_id"], 20)

    def test_water_eye_is_rejected(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 24, 26] = 3
        with self.assertRaisesRegex(RejectedScene, "not air"):
            scene_visibility(nodes, DEFS)

    def test_natural_water_surface_and_shallow_floor(self):
        rng = np.random.default_rng(42)
        survey = np.zeros((63, 63, 256), dtype=np.uint16)
        survey[:, :, :64] = 1
        survey[:, :, 64:66] = 3
        foot, surface, kind = next(site_candidates(survey, DEFS, "surface", rng))
        self.assertEqual((foot[1], surface, kind), (64, 1, "surface_water"))
        survey.fill(1)
        survey[:, :, 45:60] = 0
        survey[:, :, 68:] = 0
        foot, surface, kind = next(site_candidates(survey, DEFS, "shallow", rng))
        self.assertEqual((foot[1], surface, kind), (45, 5, "cave_floor"))
        self.assertEqual(list(site_candidates(survey, DEFS, "deep", rng)), [])


class StorageTests(unittest.TestCase):
    def test_round_trip_resume_and_measured_report(self):
        from PIL import Image
        from . import qa
        from .storage import DatasetWriter, read_arrays, read_vocabulary

        config = dict(qa.PILOT, seed=42, renderer="software", width=8, height=6)
        slot = plan(config["seed"])[0]
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[:, :, :24] = 1
        center, full, active = scene_visibility(nodes, DEFS)
        views = sample_views(active, np.random.default_rng(42))
        fov_y = math.radians(72)
        fov_x = 2 * math.atan(math.tan(fov_y / 2) * config["width"] / config["height"])
        for tick, view in enumerate(views, start=1):
            yaw, pitch = view["requested_yaw_deg"], view["requested_pitch_deg"]
            view.update(cam_pos_xyz=[0, 3, 0], cam_dir_xyz=direction_for_angles(yaw, pitch).tolist(),
                        yaw_deg=yaw, pitch_deg=pitch, fov_x_rad=fov_x, fov_y_rad=fov_y,
                        resolution=[8, 6], world_time=.5, weather="none", gamma=3.,
                        simulation_tick=tick + 1, render_simulation_tick=tick)
        meta = dict(slot, world_seed=7, native_biome=BIOMES[slot["biome"]][0],
                    foot_anchor_xyz=[0, 1, 0], region_origin_xyz=[-24, -23, -24],
                    region_origin_enu=[-24, -24, -23], anchor_kind="surface_floor", surface_y=0,
                    center_free_ratio=center, full_free_ratio=full, active_view_regions=active,
                    views=views, engine_build="fixture", game_build="fixture", engine_version={},
                    render_settings=dict(SETTINGS, renderer="software", fixed24=True,
                                         rgb_observations=True, hud=False),
                    geometry_counts={}, water_nodes=[], glass_nodes=[], world_time=.5, weather="none", gamma=3.,
                    data_source="synthetic")
        white_shape = (5, 3, 6, 8)
        arrays = dict(scene_node_id=nodes, scene_param2=nodes.astype("u1") * 7,
                      white_node_id=np.full(white_shape, EMPTY, dtype="u2"),
                      white_param2=np.zeros(white_shape, dtype="u1"),
                      white_face=np.full(white_shape, 255, dtype="u1"))
        for field, value in (("white_node_id", 1), ("white_param2", 7), ("white_face", 2)):
            arrays[field][:, 0, 0, 0] = value
        rgbs = np.random.default_rng(42).integers(0, 256, size=(5, 6, 8, 3), dtype="u1")
        vocab = ["air", "stone"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with DatasetWriter(root, config) as writer:
                scene = writer.commit(meta, vocab, arrays, rgbs)
            with DatasetWriter(root, config) as writer:
                self.assertEqual(writer.completed, {slot["scene_id"]})
            self.assertEqual(read_vocabulary(root), vocab)
            observed = read_arrays(scene, config)
            for key, array in arrays.items():
                np.testing.assert_array_equal(observed[key], array)
            with Image.open(scene / "rgb/000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), rgbs[0])
            self.assertEqual(qa.validate_scene(scene, config, vocab, slot), meta)
            with patch.dict(qa.PILOT, width=8, height=6):
                with self.assertRaisesRegex(ValueError, "Incomplete pilot"):
                    qa.check(root)

                def visit_fixture(dataset, visit):
                    read_meta, payload, pngs = qa._validate_scene(scene, config, vocab, slot)
                    coverage = qa.Coverage(vocab)
                    coverage.add(read_meta, payload)
                    visit(scene, payload, pngs)
                    return coverage.report()

                with patch.object(qa, "_scan", side_effect=visit_fixture):
                    report = qa.compression_report(root)
                self.assertEqual(report["rgb"]["raw_bytes"], rgbs.nbytes)
                self.assertEqual(report["projected_50k_total_bytes"], 50 * report["pilot_total_bytes"])
                self.assertEqual(report["coverage"]["layers"][0]["valid_hits"], 5)
                self.assertTrue((root / "compression_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
