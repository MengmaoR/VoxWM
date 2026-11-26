"""Synthetic checks for the peripheral-chunk coverage calculation."""
from unittest import TestCase

import numpy as np

from voxel_vae.outer_chunk_rank import blocks, motion_scores, select_frame


class OuterChunkRankTests(TestCase):
    def test_blocks_partition_the_frame_in_spatial_order(self):
        frame = (np.arange(48 ** 3).reshape(48, 48, 48) % 1580).astype(np.uint16)
        parts = blocks(frame)
        self.assertEqual(parts.shape, (27, 16 ** 3))
        for index, position in enumerate(np.ndindex(3, 3, 3)):
            region = tuple(slice(16 * i, 16 * (i + 1)) for i in position)
            np.testing.assert_array_equal(parts[index], frame[region].ravel())

    def test_global_frequency_vector_then_local_frequency_vector_then_token_id(self):
        chunks = np.zeros((27, 16 ** 3), np.uint16)
        frequency = np.array([10**9, 12_000, 12_000, 20_000, 25_000], np.int64)
        for index, first_id, first_count, second_id in (
            (0, 1, 8, 4), (1, 2, 1, 3), (3, 1, 8, 3), (4, 2, 8, 3)
        ):
            chunks[index, :first_count] = first_id
            chunks[index, first_count:] = second_id
        chunks[2] = 1
        selected, _ = select_frame(chunks, frequency)
        # 2 has the shorter frequency prefix. 4 beats 3 on ID, 3 beats 1 on
        # local count, and 1 beats 0 on the second global count.
        self.assertEqual(list(selected)[:5], [2, 4, 3, 1, 0])

    def test_tie_count_precedes_id_and_spatial_tiebreaks(self):
        chunks = np.zeros((27, 16 ** 3), np.uint16)
        chunks[0, 0] = 1
        chunks[1, 0] = 2
        selected, stats = select_frame(chunks, np.array([10**8, 1, 1], np.int64))
        self.assertEqual(list(selected), [1, 0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(stats['tied_pairs'], 1 + 24 * 23 // 2)
        self.assertEqual(stats['pairs_tied_after_id'], 24 * 23 // 2)
        self.assertEqual(stats['tied_groups'], 2)
        self.assertEqual(stats['tied_chunks'], 26)
        self.assertEqual(stats['frames_with_tie'], 1)
        for k in (4, 6, 8):
            self.assertEqual(stats[f'cutoff_{k}_tie'], 1)

    def test_adjacent_state_changes_outrank_per_frame_rarity(self):
        frames = np.zeros((5, 27, 16 ** 3), np.uint16)
        frames[:, 0, 0] = 3
        frames[:, 5, 0] = [1, 2, 1, 1, 1]
        frames[:, 6, 0] = [2, 1, 2, 2, 2]
        frames[:, 7, :5] = np.array([0, 4, 4, 0, 4])[:, None]
        frames[:, 13, 0] = [0, 1, 2, 3, 4]
        motion = motion_scores(frames)
        np.testing.assert_array_equal(motion[[0, 5, 6, 7, 13]], [0, 2, 2, 3, 4])
        frequency = np.array([10**8, 90_000, 100_000, 50_000, 250_000], np.int64)
        for t, expected in [(0, [7, 5, 6, 0]), (1, [7, 6, 5, 0])]:
            selected, _ = select_frame(frames[t], frequency, motion=motion)
            self.assertEqual(list(selected)[:4], expected)
            self.assertNotIn(13, selected)
