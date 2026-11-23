"""Small synthetic dry-cave and wet-camera controls for the v3 search."""

from collections import Counter
import unittest

import numpy as np

from .protocol import NOMINAL_CAMERA_XYZ, RejectedScene
from .v3_sampling import cave_evidence, require_camera, scene_visibility, settled_anchor, site_candidates


DEFINITIONS = {
    0: dict(name="air"),
    1: dict(name="stone", terrain=True, walkable=True),
    2: dict(name="water", water=True),
    3: dict(name="leaves", walkable=True),
}


class NativeKindGates(unittest.TestCase):
    def test_floor_is_not_a_visibility_occluder(self):
        nodes = np.zeros((48, 48, 48), dtype=np.uint16)
        nodes[..., :24] = 1
        self.assertEqual(sum(scene_visibility(nodes, DEFINITIONS, "surface")[2]), 40)

    def test_enclosed_cave_passes_even_with_small_free_ratios(self):
        nodes = np.ones((48, 48, 48), dtype=np.uint16)
        nodes[19:29, 19:29, 24:30] = 0
        evidence = cave_evidence(nodes, DEFINITIONS)
        self.assertEqual(evidence["sky_open_fraction"], 0)
        center, full, active = scene_visibility(nodes, DEFINITIONS, "cave")
        self.assertLess(center, 1 / 3)
        self.assertLess(full, 1 / 4)
        self.assertGreaterEqual(sum(active), 20)

    def test_open_and_overhung_ravine_fail_cave(self):
        nodes = np.ones((48, 48, 48), dtype=np.uint16)
        nodes[16:32, :, 24:] = 0
        with self.assertRaisesRegex(RejectedScene, "roof"):
            cave_evidence(nodes, DEFINITIONS)
        nodes[24, 24, 28:30] = 3
        with self.assertRaisesRegex(RejectedScene, "roof"):
            cave_evidence(nodes, DEFINITIONS)
        nodes[24, 24, 28:30] = 1
        with self.assertRaisesRegex(RejectedScene, "roof coverage"):
            cave_evidence(nodes, DEFINITIONS)

    def test_water_medium_needs_submersion(self):
        nodes = np.full((48, 48, 48), 2, dtype=np.uint16)
        nodes[..., :24] = 1
        self.assertEqual(sum(scene_visibility(nodes, DEFINITIONS, "underwater")[2]), 40)
        require_camera(nodes, DEFINITIONS, NOMINAL_CAMERA_XYZ, "underwater", lambda _: True)
        with self.assertRaisesRegex(RejectedScene, "actual liquid volume"):
            require_camera(nodes, DEFINITIONS, NOMINAL_CAMERA_XYZ, "underwater", lambda _: False)
        with self.assertRaisesRegex(RejectedScene, "not air"):
            require_camera(nodes, DEFINITIONS, NOMINAL_CAMERA_XYZ, "surface")

    def test_site_search_starts_from_native_floors_without_vae_height_band(self):
        survey = np.ones((64, 64, 96), dtype=np.uint16)
        survey[28:38, 28:38, 32:39] = 0
        counters = Counter()
        found = next(site_candidates(survey, DEFINITIONS, "cave", np.random.default_rng(42),
                                     counters, -112, hints=[(33, 34, 33)]))
        foot, surface, kind = found
        self.assertEqual((foot, kind), ((33, 32, 33), "cave_floor"))
        self.assertLess(foot[1] - 112, -38)
        self.assertGreater(counters["candidate_floors"], 1)
        survey[..., 40:] = 2
        foot, _, kind = next(site_candidates(survey, DEFINITIONS, "underwater",
                                            np.random.default_rng(42), Counter(), -112))
        self.assertEqual((foot[1], kind), (93, "underwater_floor"))

    def test_underwater_discovery_has_no_bed_or_crop_requirement(self):
        survey = np.full((64, 64, 96), 2, dtype=np.uint16)
        start, _, _ = next(site_candidates(survey, DEFINITIONS, "underwater",
                                          np.random.default_rng(42), Counter(), -112))
        self.assertEqual(start[1], 93)
        self.assertEqual(settled_anchor((-90.2, -112.501, 17.2)), (-90, -112, 17))
