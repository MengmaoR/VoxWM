"""Same-frame underwater player/camera metadata, without a Native run."""
import copy
import math
import unittest

import numpy as np

from .protocol import (NOMINAL_CAMERA_HEIGHT, VISIBILITY_LOOKUP_VERSION, VISIBILITY_RADIUS,
                       camera_angles, direction_for_angles, sample_views)
from .qa import validate_view
from .v3_report import _captured_player_foot


class CameraHeightTests(unittest.TestCase):
    def setUp(self):
        self.foot = np.array([0, 0, 0])
        self.settled = np.array([0., -.4999, 0.])
        player = self.settled + [0, .0001, 0]
        camera = player + [0, 1.6, 0]
        self.config = dict(width=640, height=360)
        self.meta = dict(world_time=.5, weather="none", gamma=3.)
        view = sample_views([True] * 40, np.random.default_rng(42))[0]
        direction = direction_for_angles(view["requested_yaw_deg"], view["requested_pitch_deg"])
        yaw, pitch = camera_angles(direction)
        view.update(foot_anchor_world_xyz=self.foot.tolist(), actual_camera_world_xyz=camera.tolist(),
                    actual_player_world_xyz=player.tolist(), actual_camera_height_above_foot=1.6,
                    nominal_camera_height=NOMINAL_CAMERA_HEIGHT, visibility_radius=VISIBILITY_RADIUS,
                    visibility_lookup_version=VISIBILITY_LOOKUP_VERSION,
                    projection_origin_local_xyz=camera.tolist(), cam_dir_xyz=direction.tolist(),
                    yaw_deg=yaw, pitch_deg=pitch, resolution=[640, 360], fov_y_rad=math.radians(72),
                    fov_x_rad=2 * math.atan(math.tan(math.radians(72)/2) * 640/360),
                    simulation_tick=10, **self.meta)
        self.view = view

    def validate(self, view, player):
        return validate_view(view, self.config, self.foot, [True] * 40, self.meta,
                             np.zeros((48, 48, 48), dtype="u2"), {0: dict(name="air")},
                             actual_foot=player)

    def test_small_settling_uses_the_captured_player_position(self):
        with self.assertRaisesRegex(ValueError, "Recorded camera height"):
            self.validate(self.view, self.settled)
        self.validate(self.view, _captured_player_foot(self.view, self.settled))

    def test_real_motion_and_incorrect_height_are_still_rejected(self):
        moved = copy.deepcopy(self.view)
        moved["actual_player_world_xyz"][1] += .004
        with self.assertRaisesRegex(ValueError, "moved after settling"):
            _captured_player_foot(moved, self.settled)
        invalid = dict(self.view, actual_camera_height_above_foot=1.61)
        with self.assertRaisesRegex(ValueError, "Recorded camera height"):
            self.validate(invalid, _captured_player_foot(invalid, self.settled))


if __name__ == "__main__":
    unittest.main()
