"""Production quotas, seed independence and unchanged single-surface storage."""
from collections import Counter
import unittest

import numpy as np

from .native import scene_id_for_world_seed, scene_world_seed
from .protocol import BIOMES, v3_plan
from .storage import V3_WHITE_FIELDS, array_schema, scene_shapes
from .v3_report import validate_lighting_pair


class ProductionFormatTests(unittest.TestCase):
    def test_full_plan_keeps_kind_and_biome_quotas(self):
        slots = v3_plan(dict(mode="production", scenes=10000, seed=42))
        self.assertEqual(Counter(s["kind"] for s in slots), dict(surface=6000, cave=3000, underwater=1000))
        for kind, target in (("surface", 240), ("cave", 120)):
            counts = Counter(s["biome"] for s in slots if s["kind"] == kind)
            self.assertEqual(counts, dict.fromkeys(BIOMES, target))
        self.assertEqual([s["scene_id"] for s in slots], list(range(10000)))

    def test_new_seed_stride_does_not_reuse_another_slots_world(self):
        seeds = []
        for scene in (0, 1, 199, 200, 9999):
            for attempt in (0, 7, 8, 15, 16):
                seed = scene_world_seed(scene, 42, attempt, scene_count=10000)
                seeds.append(seed)
                self.assertEqual(scene_id_for_world_seed(seed, 42, scene_count=10000), scene)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(scene_world_seed(199, 42, 15), scene_world_seed(199, 42, 15, scene_count=200))

    def test_production_keeps_smoke_array_schema(self):
        config = dict(mode="production", views_per_scene=5, width=640, height=360)
        self.assertEqual(scene_shapes(config), scene_shapes(dict(config, mode="smoke")))
        types = dict(scene_node_id="uint16", scene_param2="uint8", **V3_WHITE_FIELDS)
        arrays = {name: np.zeros(shape, dtype=types[name]) for name, shape in scene_shapes(config).items()}
        array_schema(arrays, config)
        self.assertEqual(arrays["surface_node_id"].shape, (5, 360, 640))

    def test_production_does_not_require_repeating_lighting_experiment(self):
        config = dict(mode="production")
        meta = dict(lighting_comparison=dict(status="NOT_COLLECTED_PRODUCTION"))
        self.assertEqual(validate_lighting_pair(meta, {}, [], config)["status"], "VERIFIED")
        with self.assertRaises(ValueError):
            validate_lighting_pair(meta, {"unexpected_control": None}, [], config)


if __name__ == "__main__":
    unittest.main()
