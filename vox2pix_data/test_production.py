"""Small coordinator fixtures only; these do not start Craftium or create native data."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .__main__ import main, parser
from .production import Progress, _leftover_native, production_config, run_tasks
from .protocol import RejectedScene, V3_FINAL_PROFILE
from .v3_native import search_world


class FixtureWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.slots = {i: dict(scene_id=i, kind="surface", biome="fixture") for i in range(10)}
        self.completed = set()

    def commit(self, meta):
        self.completed.add(meta["scene_id"])


def options(**fields):
    return SimpleNamespace(scene_ids=None, max_scenes=2, worlds_per_task=1,
                           search_round=0, craftium_dir=Path("unused"), sites_per_world=1,
                           survey_columns=16, scene_timeout=1, task_retries=0, **fields)


class ProductionTests(unittest.TestCase):
    def test_cli_defaults_keep_historical_pilot_and_select_full_production(self):
        with patch("vox2pix_data.__main__.collect_pilot", return_value=0) as pilot:
            self.assertEqual(main(["collect", "--dataset", "unused"]), 0)
            self.assertEqual((pilot.call_args.args[0].scenes, pilot.call_args.args[0].workers), (200, 1))
        with patch("vox2pix_data.production.collect_production", return_value=0) as collect:
            self.assertEqual(main(["collect", "--mode", "production", "--dataset", "unused"]), 0)
            args = collect.call_args.args[0]
            self.assertEqual(args.scenes, 10000)
            self.assertGreaterEqual(args.workers, 1)
            self.assertEqual(args.lighting_profile, V3_FINAL_PROFILE)
            self.assertEqual(args.renderer, "window" if os.sys.platform == "darwin" else "software")

    def test_production_config_fixes_rendering_and_excludes_runtime_limits(self):
        args = parser().parse_args(["collect", "--mode", "production", "--dataset", "unused",
                                   "--scenes", "10", "--workers", "2", "--max-scenes", "2"])
        config = production_config(args)
        self.assertEqual(config["collection_plan_version"], "balanced-v3-1")
        self.assertEqual(config["split_percent"], [80, 10, 10])
        self.assertEqual(config["views_per_scene"], 5)
        self.assertNotIn("max_scenes", config)
        self.assertNotIn("workers", config)
        args.lighting_profile = "global-lut"
        with self.assertRaisesRegex(ValueError, "only the final"):
            production_config(args)

    def test_adaptive_cli_only_changes_runtime_scheduling(self):
        with patch("vox2pix_data.production.collect_production", return_value=0) as collect:
            self.assertEqual(main(["collect", "--mode", "production", "--dataset", "unused",
                                   "--scenes", "200", "--workers", "16", "--adaptive-workers",
                                   "--adaptive-start", "4", "--adaptive-window", "60"]), 0)
            args = collect.call_args.args[0]
            self.assertEqual((args.adaptive_workers, args.adaptive_start, args.adaptive_window),
                             (True, 4, 60))
            self.assertEqual(args.workers, 16)
            config = production_config(args)
            self.assertNotIn("adaptive_workers", config)
            self.assertNotIn("adaptive_window", config)

    def test_adaptive_limit_changes_keep_every_committed_scene(self):
        class Controller:
            target, maximum = 1, 2
            best_workers, best_rate, phase, reason = 1, 1., "fixture", "fixture scheduling"

            def __init__(self):
                self.history = []
                self.finished = False

            def update(self, *, now, completed, in_flight, queued):
                self.target = 1 if completed >= 3 else 2 if completed >= 1 else 1

            def snapshot(self):
                return dict(target=self.target, maximum=self.maximum, finished=self.finished)

            def finish(self, **kwargs):
                self.finished = True

        lock = threading.Lock()
        calls, active, peak = [], 0, 0

        def capture(slot, *fields):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append(slot["scene_id"])
            time.sleep(.01)
            with lock:
                active -= 1
            return (slot,)

        with TemporaryDirectory() as temporary, ThreadPoolExecutor(2) as pool:
            writer = FixtureWriter(temporary)
            args = options()
            args.max_scenes = 10
            controller = Controller()
            progress = Progress(writer, controller.target)
            self.assertEqual(run_tasks(writer, {}, args, "unused", pool, progress,
                                       lambda: False, capture, controller), 0)
            self.assertEqual(sorted(calls), list(range(10)))
            self.assertEqual(writer.completed, set(range(10)))
            self.assertEqual(peak, 2)
            self.assertEqual(progress.state["workers"], 1)
            self.assertTrue(progress.state["adaptive"]["finished"])
            self.assertEqual(progress.state["status"], "COMPLETE")

    def test_two_parallel_slots_stop_at_limit_without_replacing_failures(self):
        lock = threading.Lock()
        calls, active, peak = [], 0, 0

        def capture(slot, *fields):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append(slot["scene_id"])
            time.sleep(.02)
            with lock:
                active -= 1
            return (slot,)

        with TemporaryDirectory() as temporary, ThreadPoolExecutor(2) as pool:
            writer = FixtureWriter(temporary)
            progress = Progress(writer, 2)
            self.assertEqual(run_tasks(writer, {}, options(), "unused", pool, progress, lambda: False, capture), 0)
            self.assertEqual(set(calls), {0, 1})
            self.assertEqual(peak, 2)
            self.assertEqual(progress.state["status"], "LIMIT_REACHED")
            self.assertEqual(progress.state["completed"], 2)
            self.assertEqual(progress.next_attempt, {})

    def test_failed_seed_cursor_survives_resume_and_uses_next_seed(self):
        seen = []

        def capture(slot, config, craftium, library, worlds, sites, columns, seconds, first):
            seen.append(first)
            if first == 0:
                raise RejectedScene("fixture bounded search exhausted")
            return (slot,)

        with TemporaryDirectory() as temporary, ThreadPoolExecutor(1) as pool:
            writer = FixtureWriter(temporary)
            args = options()
            args.max_scenes = 1
            first = Progress(writer, 1)
            self.assertEqual(run_tasks(writer, {}, args, "unused", pool, first, lambda: False, capture), 1)
            self.assertEqual(first.state["status"], "INCOMPLETE")
            second = Progress(writer, 1)
            self.assertEqual(run_tasks(writer, {}, args, "unused", pool, second, lambda: False, capture), 0)
            self.assertEqual(seen, [0, 1])
            self.assertEqual(writer.completed, {0})

    def test_owned_cleanup_excludes_pid_one_parent_and_unrelated_tagged_helpers(self):
        result = SimpleNamespace(stdout="91 /tmp/bin/luanti\n92 /usr/bin/python\n")
        with patch("vox2pix_data.production.owned_pids", return_value={1, os.getpid(), 91, 92}), \
                patch("vox2pix_data.production.subprocess.run", return_value=result), \
                patch("vox2pix_data.production.os.kill") as kill, \
                patch("vox2pix_data.production.time.sleep"):
            _leftover_native("fixture-run")
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {91})

    def test_real_biome_mismatch_rejects_before_acquire(self):
        native = Mock(search_deadline=time.monotonic() + 5)
        native.definitions = {}
        native.command.side_effect = [({"status": "found", "site": [0, 0]}, None, None),
                                      ({"valid": True, "surface_biome": "wrong"}, None, None)]
        nodes = np.zeros((63, 63, 63), dtype=np.uint16)
        with patch("vox2pix_data.v3_native.survey", return_value=(nodes, (0, 0, 0), [])), \
                patch("vox2pix_data.v3_native.site_candidates", return_value=[((24, 24, 24), 23, "surface_floor")]), \
                patch("vox2pix_data.v3_native.scene_visibility"), \
                patch("vox2pix_data.v3_native.biome_group", return_value=None):
            result = search_world(native, "surface", {"survey_y": [0, 62]}, np.random.default_rng(42),
                                  "unused", Counter(), 1, 16, target_biome="target")
        self.assertIsNone(result)
        native.acquire.assert_not_called()
        self.assertEqual(native.command.call_args_list[0].args[0], "search")


if __name__ == "__main__":
    unittest.main()
