"""Synthetic scheduler measurements only; no Native, network or dataset capture."""

import unittest

from .adaptive import AdaptiveConcurrency


class AdaptiveTests(unittest.TestCase):
    def controller(self, **fields):
        return AdaptiveConcurrency(max_workers=32, window_seconds=60, settle_seconds=0, **fields)

    def step(self, controller, time, count, **fields):
        options = dict(now=time, completed=count, in_flight=controller.target, queued=100)
        options.update(fields)
        return controller.update(**options)

    def test_ramp_and_confirmed_decline_roll_back_without_cancelling(self):
        c = self.controller()
        self.step(c, 0, 0)
        self.assertEqual(self.step(c, 60, 10), 8)
        self.step(c, 61, 10)
        self.assertEqual(self.step(c, 121, 30), 16)
        self.step(c, 122, 30)
        self.assertEqual(self.step(c, 182, 45), 16)  # 25% below best; confirm once.
        self.assertEqual(self.step(c, 242, 60), 8)
        self.assertEqual(c.outcome, "rollback_after_decline")
        # Lowering a scheduling target does not cancel any of the 16 live tasks.
        self.assertEqual(self.step(c, 243, 60, in_flight=16), 8)
        self.assertEqual(c.best_workers, 8)

    def test_one_bad_window_is_not_a_rollback(self):
        c = self.controller()
        self.step(c, 0, 0)
        self.step(c, 60, 10)
        self.step(c, 61, 10)
        self.step(c, 121, 18)  # First low window at 8.
        self.assertEqual(c.phase, "confirming_decline")
        self.assertEqual(self.step(c, 181, 30), 16)  # Recovery can keep ramping.
        self.assertEqual(c.best_workers, 8)

    def test_warmup_does_not_inflate_throughput(self):
        c = AdaptiveConcurrency(16, window_seconds=60, settle_seconds=15)
        self.step(c, 0, 0)
        self.step(c, 10, 100)
        self.step(c, 15, 100)
        self.assertEqual(self.step(c, 74, 110), 4)
        self.assertEqual(self.step(c, 75, 110), 8)
        self.assertEqual(c.best_rate, 600)

    def test_tail_or_underfilled_pool_does_not_create_decline(self):
        c = self.controller()
        self.step(c, 0, 0)
        self.step(c, 60, 10)
        self.step(c, 61, 10)
        self.assertEqual(self.step(c, 121, 11, in_flight=3, queued=0), 8)
        self.assertEqual(c.phase, "tail")
        self.assertEqual(len(c.history), 1)
        result = c.finish(now=130, completed=12)
        self.assertEqual(result["outcome"], "budget_exhausted")
        self.assertEqual(result["best_workers"], 4)

    def test_not_enough_budget_does_not_launch_next_stage(self):
        c = self.controller()
        self.step(c, 0, 0)
        self.assertEqual(self.step(c, 60, 10, queued=15), 4)
        self.assertEqual(c.outcome, "budget_exhausted")

    def test_ceiling_and_pressure_are_explicit_numeric_inputs(self):
        c = AdaptiveConcurrency(4, start_workers=8, window_seconds=60, settle_seconds=0)
        self.assertEqual(c.target, 4)
        self.step(c, 0, 0)
        self.step(c, 60, 10)
        self.assertEqual(c.outcome, "max_workers_tested")
        self.assertEqual(self.step(c, 61, 10, resource_pressure=True), 2)
        self.assertEqual(c.outcome, "resource_pressure")
        self.assertEqual(self.step(c, 62, 10, resource_pressure=False), 2)
        result = c.finish(now=65, completed=10)
        self.assertEqual(result["history"][-1]["selected_workers"], 2)

    def test_no_commits_cannot_justify_more_workers(self):
        c = self.controller()
        self.step(c, 0, 0)
        self.assertEqual(self.step(c, 60, 0), 4)
        self.assertIsNone(c.best_workers)
        self.assertEqual(c.history[-1]["decision"], "no_commits")


if __name__ == "__main__":
    unittest.main()
