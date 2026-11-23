"""Synthetic ray fixtures for the single-surface Vox2Pix v3 format."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .geometry import Projector, build_library, water_contains


DEFINITIONS = {
    0: {"name": "air"},
    1: {"name": "stone"},
    2: {"name": "water_source", "drawtype": "liquid", "water": True},
    3: {"name": "water_flowing", "drawtype": "flowingliquid", "water": True,
        "liquid_alternative_source": "water_source"},
    4: {"name": "glass", "drawtype": "glasslike"},
    5: {"name": "waterlogged_roots", "drawtype": "allfaces_optional", "water": True},
    6: {"name": "nodebox", "drawtype": "nodebox"},
}


class V3ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.projector = Projector(build_library(Path(directory.name)))

    def setUp(self):
        self.nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        self.params = np.full(self.nodes.shape, 7, dtype=np.uint8)

    def trace(self, origin=(0., 0., 0.), direction=(0., 0., 1.), boxes=None):
        scene = self.projector.scene(self.nodes, self.params, DEFINITIONS, boxes or {})
        return tuple(field.item() for field in scene.render_v3(origin, direction, 1., 1., 1, 1))

    def test_first_surface_after_air_and_water(self):
        self.nodes[24, 28, 24] = 1
        self.assertEqual(self.trace(), (1, 7, 5, 0))
        self.nodes[24, 27, 24] = 2
        self.assertEqual(self.trace(), (1, 7, 5, 1))

    def test_three_water_background(self):
        self.nodes[24, 26:29, 24] = 2
        self.assertEqual(self.trace(), (65535, 0, 255, 3))

    def test_cube_corner_surface_beyond_radius24_remains_available(self):
        self.nodes[44, 44, 24] = 1
        self.assertEqual(self.trace(direction=(1., 0., 1.))[:2], (1, 7))

    def test_water_count_stops_at_the_stored_cube_boundary(self):
        self.nodes[24, 44:48, 24] = 2
        self.assertEqual(self.trace(), (65535, 0, 255, 4))

    def test_initial_water_voxel_exited_before_near_plane(self):
        self.nodes[24, 24, 24] = 2
        origin = (0, 0, .45)
        self.assertTrue(water_contains(self.nodes, self.params, DEFINITIONS, origin))
        self.assertEqual(self.trace(origin)[3], 1)

    def test_glass_and_waterlogged_roots_are_surfaces(self):
        self.nodes[24, 26:29, 24] = [2, 4, 1]
        self.assertEqual(self.trace(), (4, 7, 5, 1))
        self.nodes[24, 27, 24] = 5
        self.assertEqual(self.trace(), (5, 7, 5, 1))
        self.assertFalse(water_contains(self.nodes, self.params, DEFINITIONS, (0, 0, 3)))

    def test_flow_volume_and_grazing(self):
        self.nodes[24, 26, 24] = 3
        self.assertEqual(self.trace()[3], 0)
        self.assertEqual(self.trace((0, -.49, 0))[3], 1)
        self.assertEqual(self.trace((-.5, -.49, 0))[3], 0)
        self.assertFalse(water_contains(self.nodes, self.params, DEFINITIONS, (0, 0, 2)))
        self.assertTrue(water_contains(self.nodes, self.params, DEFINITIONS, (0, -.49, 2)))

    def test_flow_triangles_do_not_double_count(self):
        self.nodes[24, 26, 24] = 3
        self.nodes[25, 27, 24] = 2  # Raise one corner of the adjacent flowing node.
        self.assertEqual(self.trace((0, -.49, 0))[3], 1)
        self.assertEqual(self.trace((.4, 0., 0))[3], 1)
        self.assertTrue(water_contains(self.nodes, self.params, DEFINITIONS, (.4, 0., 2.4)))
        self.assertFalse(water_contains(self.nodes, self.params, DEFINITIONS, (.4, 0., 1.6)))

    def test_water_cut_by_protruding_surface_has_not_exited(self):
        self.nodes[24, 26, 24] = 2
        self.nodes[25, 26, 24] = 6
        index = np.ravel_multi_index((25, 26, 24), self.nodes.shape)
        box = [-1.25, -.25, .1, -.1, .25, .25]
        self.assertEqual(self.trace(boxes={index: [box]}), (6, 7, 5, 0))


if __name__ == "__main__":
    unittest.main()
