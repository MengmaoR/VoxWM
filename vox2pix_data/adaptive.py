"""Small throughput-based concurrency ramp; it never cancels running scenes.

The coordinator calls ``update`` after filling the current in-flight target.
Only committed scenes count as progress. Warm-up, a partially filled pool and
the final drain are excluded from comparisons. This is a bounded heuristic,
not a claim to have measured a machine's global optimum.
"""


class AdaptiveConcurrency:
    def __init__(self, max_workers, start_workers=4, window_seconds=90,
                 settle_seconds=15, decline_fraction=.10):
        if max_workers < 1 or start_workers < 1 or window_seconds <= 0 or settle_seconds < 0:
            raise ValueError("Adaptive worker counts/window must be positive; settle must be nonnegative")
        if not 0 < decline_fraction < 1:
            raise ValueError("Adaptive decline fraction must be between zero and one")
        self.maximum = int(max_workers)
        self.target = min(int(start_workers), self.maximum)
        self.window_seconds = float(window_seconds)
        self.settle_seconds = float(settle_seconds)
        self.decline_fraction = float(decline_fraction)
        self.best_workers = None
        self.best_rate = None
        self.phase = "starting"
        self.outcome = "searching"
        self.reason = None
        self.history = []
        self._settle_until = None
        self._window_start = None
        self._window_completed = None
        self._declines = 0
        self._locked = False

    def _reset_window(self):
        self._settle_until = None
        self._window_start = None
        self._window_completed = None

    def _lock(self, target, outcome, reason):
        self.target = target
        self._locked = True
        self.phase = "locked"
        self.outcome = outcome
        self.reason = reason
        self._reset_window()

    def update(self, *, now, completed, in_flight, queued, resource_pressure=False):
        """Return the next scheduling target; excess in-flight work drains normally.

        ``now`` is monotonic time; ``completed`` is cumulative committed scenes.
        ``queued`` excludes running scenes. ``resource_pressure`` is an optional
        caller-provided resource signal, allowing tests without reading a host.
        """
        if resource_pressure:
            lower = min(self.best_workers or self.target, max(1, self.target // 2))
            if self.outcome != "resource_pressure" or lower != self.target:
                self.history.append(dict(event="resource_pressure", at=now,
                                         workers=self.target, selected_workers=lower))
            self._lock(lower, "resource_pressure", "resource pressure; drain excess scenes without cancelling")
            return self.target
        if self._locked:
            return self.target
        if in_flight < self.target:
            # This can be the startup or final drain. Neither is performance loss.
            self._reset_window()
            self.phase = "tail" if not queued else "waiting_for_full_pool"
            return self.target
        if self._settle_until is None:
            self._settle_until = now + self.settle_seconds
        if now < self._settle_until:
            self.phase = "settling"
            return self.target
        if self._window_start is None:
            self._window_start, self._window_completed = now, completed
            self.phase = "measuring"
            return self.target
        elapsed = now - self._window_start
        if elapsed < self.window_seconds:
            return self.target
        count = completed - self._window_completed
        if count < 0:
            raise ValueError("Adaptive committed-scene count moved backwards")
        rate = count * 3600 / elapsed
        record = dict(event="window", workers=self.target, start=self._window_start,
                      end=now, seconds=elapsed, committed=count, scenes_per_hour=rate)
        self.history.append(record)
        self._window_start, self._window_completed = now, completed
        if not count:
            record["decision"] = "no_commits"
            # No initial throughput evidence cannot justify raising concurrency.
            if self.best_rate is None:
                return self.target
        old_best = self.best_rate
        if self.best_rate is None or rate > self.best_rate:
            self.best_workers, self.best_rate = self.target, rate
        decline = old_best is not None and rate <= old_best * (1 - self.decline_fraction)
        self._declines = self._declines + 1 if decline else 0
        if self._declines >= 2:
            record["decision"] = "rollback"
            self._lock(self.best_workers, "rollback_after_decline",
                       f"two consecutive full windows at least {100*self.decline_fraction:g}% "
                       "below the best observed throughput")
        elif decline:
            record["decision"] = "confirm_decline"
            self.phase = "confirming_decline"
        elif self.target >= self.maximum:
            record["decision"] = "max_workers_tested"
            self._lock(self.best_workers, "max_workers_tested", "configured concurrency ceiling was tested")
        else:
            next_target = min(self.maximum, self.target * 2)
            if queued < 2 * next_target:
                record["decision"] = "budget_exhausted"
                self._lock(self.best_workers, "budget_exhausted",
                           "not enough queued scenes for two full waves at the next concurrency")
            else:
                record["decision"] = "increase"
                self.target = next_target
                self._declines = 0
                self._reset_window()
                self.phase = "settling"
        return self.target

    def finish(self, *, now, completed):
        """Finish without interpreting an undersized final window as slowdown."""
        if not self._locked:
            self.outcome = "budget_exhausted"
            self.reason = "collection ended before another full comparison window"
        self.phase = "finished"
        self.history.append(dict(event="finished", at=now, completed=completed,
                                 selected_workers=self.target if self._locked else self.best_workers or self.target,
                                 outcome=self.outcome))
        return self.snapshot()

    def snapshot(self):
        return dict(target=self.target, maximum=self.maximum, best_workers=self.best_workers,
                    best_scenes_per_hour=self.best_rate, phase=self.phase,
                    outcome=self.outcome, reason=self.reason,
                    window_seconds=self.window_seconds, settle_seconds=self.settle_seconds,
                    decline_fraction=self.decline_fraction, history=list(self.history))
