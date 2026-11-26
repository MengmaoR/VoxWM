from unittest import TestCase

from voxel_vae.scene_overlap_geometry import summarize


class SceneOverlapTests(TestCase):
    def test_discrete_intersection_and_same_seed(self):
        a = (17, (-10, -2, -31), 'Static', 'clip1', 'a')
        b = (17, (37, 44, 14), 'Static', 'clip2', 'b')
        for axis in range(3):
            c = list(a[1])
            c[axis] += 48
            r = summarize([a, (17, tuple(c), 'Static', 'clip1', 'touch')])
            self.assertEqual(r['overlap_pairs'], 0)
        r = summarize([a, b, (19, a[1], 'Static', 'clip1', 'c')])
        self.assertEqual((r['eligible_pairs'], r['partial_pairs'], r['across_clip_pairs']), (1, 1, 1))
        self.assertEqual((r['pair_volume_sum'], r['affected_scenes'], r['center16_overlap_pairs']), (6, 2, 0))

    def test_distinct_scenes_at_one_center_and_cross_filter(self):
        scenes = [(8, (0, 0, 0), 'D1', 'a', '0'), (8, (0, 0, 0), 'D4', 'b', '1'),
                  (8, (15, 0, 0), 'Static', 'a', '2'), (9, (0, 0, 0), 'D1', 'a', '3')]
        r = summarize(scenes)
        self.assertEqual((r['overlap_pairs'], r['identical_center_pairs'], r['center16_overlap_pairs']), (3, 1, 3))
        self.assertEqual(r['pair_volume_max'], 48 ** 3)
        c = summarize(scenes, cross=True)
        self.assertEqual((c['eligible_pairs'], c['overlap_pairs'], c['partial_pairs']), (2, 2, 2))
        self.assertEqual(c['affected_by_category'], {'Static': 1, 'D1': 1, 'D4': 1})
        self.assertEqual(c['affected_seeds'], 1)
