"""Small checks for the active rendering profile and historical metadata."""

import unittest

from .__main__ import parser
from .protocol import (
    V3_FINAL_GEOMETRY, V3_FINAL_PROFILE, V3_FINAL_RGB, V3_FINAL_VERSION,
    V3_FOG48_GEOMETRY, V3_FOG48_PROFILE, V3_FOG48_RGB, V3_FOG48_VERSION,
    V3_FOG_GEOMETRY, V3_FOG_PROFILE, V3_VERSION, v3_fog_geometry, v3_rgb, v3_version,
)


class FinalProtocolTests(unittest.TestCase):
    def test_active_profile_changes_only_draw_range(self):
        config = {"lighting_profile": V3_FINAL_PROFILE}
        self.assertEqual(v3_rgb(config), dict(V3_FOG48_RGB, viewing_range=32))
        self.assertIs(v3_rgb(config), V3_FINAL_RGB)
        self.assertNotIn("fog_start", V3_FINAL_RGB)
        self.assertEqual(V3_FINAL_RGB["fog_start_distance"], 20)
        self.assertEqual(V3_FINAL_RGB["fog_end_distance"], 30)
        self.assertEqual(V3_FINAL_RGB["display_gamma"], 3)
        self.assertEqual(V3_FINAL_RGB["lighting_floor"], 0)
        self.assertEqual(V3_FINAL_RGB["solid_ambient_floor"], .35)
        self.assertEqual(v3_version(config), V3_FINAL_VERSION)

    def test_cli_default_and_explicit_legacy_profile(self):
        args = parser().parse_args(["collect", "--dataset", "unused", "--mode", "smoke"])
        self.assertEqual(args.lighting_profile, V3_FINAL_PROFILE)
        args = parser().parse_args(["collect", "--dataset", "unused", "--mode", "smoke",
                                    "--lighting-profile", V3_FOG48_PROFILE])
        self.assertEqual(args.lighting_profile, V3_FOG48_PROFILE)

    def test_new_and_legacy_geometry_metadata_stay_distinct(self):
        final = v3_fog_geometry({"lighting_profile": V3_FINAL_PROFILE})
        self.assertEqual(final, V3_FINAL_GEOMETRY)
        self.assertEqual(final["voxel_observation_shape"], [48, 48, 48])
        self.assertEqual(final["voxel_observation_half_extent"], 24)
        self.assertEqual(final["rgb_viewing_range"], 32)
        self.assertEqual(final["fog_type"], "native_linear")
        self.assertEqual(v3_fog_geometry({"lighting_profile": V3_FOG48_PROFILE}), V3_FOG48_GEOMETRY)
        self.assertEqual(v3_fog_geometry({"lighting_profile": V3_FOG_PROFILE}), V3_FOG_GEOMETRY)
        self.assertEqual(v3_version({"lighting_profile": V3_FOG48_PROFILE}), V3_FOG48_VERSION)
        self.assertEqual(v3_version({}), V3_VERSION)
        final["rgb_viewing_range"] = 999
        self.assertEqual(V3_FINAL_GEOMETRY["rgb_viewing_range"], 32)


if __name__ == "__main__":
    unittest.main()
