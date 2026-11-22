"""Lightweight fixtures. No Craftium process or native data is used."""
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .geometry import Projector, build_library
from .protocol import (
    BIOMES, NOMINAL_CAMERA_HEIGHT, NOMINAL_CAMERA_XYZ, SETTINGS, VERSION,
    VISIBILITY_LOOKUP_VERSION, VISIBILITY_RADIUS, RejectedScene, camera_angles, camera_local_xyz,
    direction_for_angles, plan, region_id, require_air_camera, sample_views, scene_visibility,
    site_candidates, visibility_lookup,
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
                     scene.render(NOMINAL_CAMERA_XYZ, direction, 1.2, 1.2, 1, 1))

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
                 [[-.5, -.25, -.5, .5, 0., .5]]}
        self.assertEqual(self.project(nodes, boxes)[0], [EMPTY] * 3)
        self.assertEqual(self.project(nodes, boxes, (0., .12, 1.)), ([4, EMPTY, EMPTY], [28, 0, 0], [5, 255, 255]))

    def test_pixel_axes(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 26, 27] = 1
        nodes[24, 26, 25] = 2
        nodes[23, 26, 26] = 3
        nodes[25, 26, 26] = 1
        scene = self.projector.scene(nodes, nodes.astype("u1"), DEFS, {})
        node, _, face = scene.render(NOMINAL_CAMERA_XYZ, (0, 0, 1), math.pi / 2, math.pi / 2, 3, 3)
        self.assertEqual(node[0, :, 1].tolist(), [1, EMPTY, 2])
        self.assertEqual(node[0, 1, :].tolist(), [3, EMPTY, 1])
        self.assertEqual(int(face[0, 1, 0]), 5)

    def test_white_model_alignment_fixture(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 30, 26] = 1
        scene = self.projector.scene(nodes, np.zeros_like(nodes, dtype=np.uint8), DEFS, {})
        width, height = 320, 240
        fov_y = 1.2
        fov_x = 2 * math.atan(math.tan(fov_y / 2) * width / height)
        camera_y = 1.625
        node, _, _ = scene.render((0, camera_y, 0), (0, 0, 1), fov_x, fov_y, width, height)

        # Independent pinhole raster of the front face of a cube centered at (0, 2, 6).
        # All silhouette hits enter this face since the eye lies within its x/y bounds.
        front_z = 5.5
        x = (2 * (np.arange(width) + .5) / width - 1) * front_z * math.tan(fov_x / 2)
        dy = (1 - 2 * (np.arange(height) + .5) / height) * front_z * math.tan(fov_y / 2)

        def geometry_boundary(eye_y):
            y = eye_y + dy
            return ((1.5 < y) & (y < 2.5))[:, None] & (np.abs(x) < .5)[None, :]

        expected = geometry_boundary(camera_y)
        self.assertTrue(expected.any())
        np.testing.assert_array_equal(node[0] != EMPTY, expected)
        self.assertFalse(np.array_equal(expected, geometry_boundary(NOMINAL_CAMERA_HEIGHT)))
        self.assertFalse(np.array_equal(expected, geometry_boundary(2.0)))


class CameraTests(unittest.TestCase):
    def test_camera_nominal_height(self):
        self.assertEqual(VERSION, "vox2pix-pilot-v2-camera16")
        self.assertEqual(NOMINAL_CAMERA_HEIGHT, 1.6)
        foot = (12, -9, 21)
        np.testing.assert_array_equal(camera_local_xyz((12, -7.375, 21), foot), (0, 1.625, 0))
        for invalid_y in (-7., math.nan):
            with self.subTest(height=invalid_y), self.assertRaises(RejectedScene):
                camera_local_xyz((12, invalid_y, 21), foot)

    def test_actual_camera_used_for_projection(self):
        from .native import NativeScene

        foot = (12, -9, 21)
        camera_world = (12., -7.375, 21.)
        config = dict(width=8, height=6, renderer="software")
        fov_y = 1.2
        fov_x = 2 * math.atan(math.tan(fov_y / 2) * config["width"] / config["height"])
        rgb = np.full((6, 8, 3), 137, dtype=np.uint8)
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[:, :, :24] = 1
        params = np.zeros_like(nodes, dtype=np.uint8)
        tick = 0
        direction = None

        def pose(_command, **settings):
            nonlocal direction
            direction = direction_for_angles(settings["yaw"], settings["pitch"])

        def snapshot(_origin, _kind, boxes=False):
            nonlocal tick
            tick += 1
            capture = dict(native_clock=dict(server_tick=tick, patch_version="fixture",
                                           region_capture_version="fixture"),
                           world_time=.5, weather="none", gamma=3.)
            packet = dict(cam_pos=camera_world, cam_dir=direction,
                          cam_fov_x=fov_x, cam_fov_y=fov_y)
            return (nodes, params), capture, rgb, packet

        render = Mock(return_value=(np.full((3, 6, 8), EMPTY, dtype=np.uint16),
                                    np.zeros((3, 6, 8), dtype=np.uint8),
                                    np.full((3, 6, 8), 255, dtype=np.uint8)))
        projector = Mock()
        projector.scene.return_value = SimpleNamespace(render=render, geometry_counts={})
        with TemporaryDirectory() as tmp, patch("vox2pix_data.native.Projector", return_value=projector):
            world = Path(tmp)
            (world / "vox2pix_engine.json").write_text(json.dumps({
                "mapgen": "v7", "version": {}, "definitions": [
                    dict(DEFS[0], id=0, water=False, glass=False),
                    dict(DEFS[1], id=1, water=False, glass=False),
                ],
            }))
            (world / "vox2pix_boxes.json").write_text('{"boxes":[]}')
            native = NativeScene(SimpleNamespace(world=world))
            with (patch.object(native, "command", side_effect=pose),
                  patch.object(native, "step"),
                  patch.object(native, "geometry_snapshot", side_effect=snapshot)):
                meta, _, _, rgbs = native.acquire(foot, "surface_floor", config, np.random.default_rng(9), None)
        self.assertEqual(len(render.call_args_list), 5)
        for view, frame, call in zip(meta["views"], rgbs, render.call_args_list, strict=True):
            origin, cast_direction, cast_fov_x, cast_fov_y, width, height = call.args
            np.testing.assert_array_equal(frame, rgb)
            np.testing.assert_array_equal(origin, (0, 1.625, 0))
            np.testing.assert_array_equal(cast_direction, view["cam_dir_xyz"])
            self.assertEqual((cast_fov_x, cast_fov_y, width, height), (fov_x, fov_y, 8, 6))
            self.assertEqual(view["foot_anchor_world_xyz"], list(foot))
            self.assertEqual(view["actual_camera_world_xyz"], list(camera_world))
            self.assertEqual(view["actual_camera_height_above_foot"], 1.625)
            self.assertEqual(view["projection_origin_local_xyz"], [0, 1.625, 0])

    def test_visibility_radius(self):
        self.assertEqual(VISIBILITY_RADIUS, 1.6)
        self.assertEqual(VISIBILITY_LOOKUP_VERSION, "voxel-centers-r1.6-v2")

    def test_flat_floor_not_blocking(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[:, :, :24] = 1
        self.assertNotIn((0, 0, -1), visibility_lookup())
        self.assertEqual(scene_visibility(nodes, DEFS)[2], [True] * 40)

    def test_lookup_rebuilt(self):
        lookup = visibility_lookup()
        expected = {(x, z, y) for x, z, y in product(range(-2, 3), range(-2, 3), range(-1, 5))
                    if math.dist((x, y, z), NOMINAL_CAMERA_XYZ) <= VISIBILITY_RADIUS}
        self.assertEqual(set(lookup), expected)
        self.assertEqual(Counter(y for _, _, y in lookup), {0: 1, 1: 9, 2: 9, 3: 1})
        self.assertEqual(set(lookup[(0, 0, 2)]), set(range(40)))
        visibility_lookup.cache_clear()
        rebuilt = visibility_lookup()
        for offset, bins in lookup.items():
            np.testing.assert_array_equal(rebuilt[offset], bins)

    def test_active_regions(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        for x, z, y in ((-1, -1, 1), (-1, 1, 1), (0, 0, 3)):
            nodes[24 + x, 24 + z, 24 + y] = 1
        self.assertEqual(sum(scene_visibility(nodes, DEFS)[2]), 20)
        nodes[23, 25, 25] = 0
        nodes[23, 24, 25] = nodes[24, 25, 26] = 1
        with self.assertRaises(RejectedScene):
            scene_visibility(nodes, DEFS)

    def test_water_camera_rejected(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[24, 24, 26] = 3
        with self.assertRaisesRegex(RejectedScene, "not air"):
            scene_visibility(nodes, DEFS)
        with self.assertRaisesRegex(RejectedScene, "not air"):
            require_air_camera(nodes, DEFS, (0, 1.625, 0))


class SamplingTests(unittest.TestCase):
    def test_unaligned_native_response_discards_the_world(self):
        from .native import NativeScene

        with TemporaryDirectory() as temp:
            world = Path(temp)
            scene = NativeScene.__new__(NativeScene)
            scene.session = SimpleNamespace(world=world)
            scene.seq = 0
            response = dict(request_id=1, native_clock=dict(patch_version="voxwm-fixed24-v1",
                            server_tick=6078, client_request_tick=6079), lua_tick=6078)
            (world / "vox2pix_response.json").write_text(json.dumps(response))
            with patch.object(scene, "step", return_value=(None, {"native_client_tick": 6079})):
                with self.assertRaisesRegex(ConnectionError, "Unaligned native tick during survey"):
                    scene.command("survey")

    def test_lua_json_node_ids_are_valid_lookup_indices(self):
        from .native import NativeScene

        with TemporaryDirectory() as tmp:
            world = Path(tmp)
            boot = world / "vox2pix_engine.json"
            definitions = [{"id": 412.0, "name": "stone"}, {"id": 126.0, "name": "air"}]
            boot.write_text(json.dumps({"mapgen": "v7", "definitions": definitions}))
            scene = NativeScene(SimpleNamespace(world=world))
            self.assertEqual(scene.vocabulary, ["air", "stone"])
            self.assertEqual(scene.lookup[[412, 126, 999]].tolist(), [1, 0, EMPTY])

            definitions[0]["id"] = 412.25
            boot.write_text(json.dumps({"mapgen": "v7", "definitions": definitions}))
            with self.assertRaisesRegex(ValueError, "Invalid native content ID"):
                NativeScene(SimpleNamespace(world=world))

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

    def test_natural_water_surface_and_shallow_floor(self):
        rng = np.random.default_rng(42)
        survey = np.zeros((63, 63, 256), dtype=np.uint16)
        survey[:, :, :64] = 1
        survey[:, :, 64:66] = 3
        foot, surface, kind = next(site_candidates(survey, DEFS, "surface", rng))
        self.assertEqual((foot[1], surface, kind), (66, 1, "surface_water"))
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
        heights = [1.6, 1.625, 1.6, 1.625, 1.6]
        for tick, (view, eye_height) in enumerate(zip(views, heights, strict=True), start=1):
            yaw, pitch = view["requested_yaw_deg"], view["requested_pitch_deg"]
            view.update(foot_anchor_world_xyz=[0, 1, 0],
                        actual_camera_world_xyz=[0, 1 + eye_height, 0],
                        actual_camera_height_above_foot=eye_height, nominal_camera_height=NOMINAL_CAMERA_HEIGHT,
                        projection_origin_local_xyz=[0, eye_height, 0], visibility_radius=VISIBILITY_RADIUS,
                        visibility_lookup_version=VISIBILITY_LOOKUP_VERSION,
                        cam_dir_xyz=direction_for_angles(yaw, pitch).tolist(),
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
                    data_source="synthetic", nominal_camera_height=NOMINAL_CAMERA_HEIGHT,
                    visibility_radius=VISIBILITY_RADIUS, visibility_lookup_version=VISIBILITY_LOOKUP_VERSION)
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
            with (patch.dict(views[1], projection_origin_local_xyz=[0, 1.6, 0]),
                  self.assertRaisesRegex(ValueError, "White Model origin")):
                qa.validate_payload(meta, config, vocab, slot, arrays)
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
                stats = report["actual_camera_height"]
                for label, aggregate in (("min", min), ("mean", np.mean), ("max", max), ("std", np.std)):
                    self.assertAlmostEqual(stats[label], aggregate(heights))
                self.assertTrue((root / "compression_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
