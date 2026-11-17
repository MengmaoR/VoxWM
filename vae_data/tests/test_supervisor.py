import json
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from vae_data.supervisor import (MARKER, TerminalDashboard, _linux_tagged_pids,
    build_command, load_config, owned_pids, progress_signature, supervise,
    terminate_owned)


class TTYBuffer(io.StringIO):
    def isatty(self):return True


class SupervisorTests(unittest.TestCase):
    def config(self, root):
        path = Path(root) / "supervisor.json"
        path.write_text(json.dumps({"dataset": str(Path(root) / "data")}))
        return load_config(path)

    def test_default_is_64_shards_with_one_world_each_and_external_ten_minute_watchdog(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            command = build_command(config, python=sys.executable)
            self.assertEqual((config["count"], config["workers"], config["no_progress_seconds"]), (100, 64, 600))
            self.assertEqual(command[command.index("--workers") + 1], "64")
            self.assertEqual(command[command.index("--stall-seconds") + 1], "0")
            self.assertIn("--resume", command)

    def test_remaining_shards_reduce_worlds_instead_of_stacking_worlds_on_one_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            shards = config["dataset"] / "shards"
            for shard in range(3, 100):
                ready = shards / f"shard_{shard:04d}" / "READY"
                ready.parent.mkdir(parents=True)
                ready.write_text("{}")
            command = build_command(config, python=sys.executable)
            self.assertEqual(command[command.index("--workers") + 1], "3")

    def test_completed_attempt_is_progress_even_without_a_sample_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            partial = config["dataset"] / "shards/.shard_0000.partial"
            partial.mkdir(parents=True)
            before = progress_signature(config)
            (partial / "cursor.json").write_text('{"attempts": 1}')
            self.assertNotEqual(before, progress_signature(config))

    def test_linux_process_scan_requires_the_exact_unique_run_tag(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            for pid, value in (("101", b"A=1\0" + MARKER.encode() + b"=wanted\0"),
                               ("102", b"A=1\0" + MARKER.encode() + b"=other\0"),
                               ("103", b"A=1\0")):
                entry = proc / pid
                entry.mkdir()
                (entry / "environ").write_bytes(value)
            self.assertEqual(_linux_tagged_pids("wanted", proc), {101})

    def test_terminal_dashboard_renders_progress_grid_and_stall_countdown(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            stream = TTYBuffer();dashboard = TerminalDashboard(True, stream)
            now = time.monotonic()
            status = {"shard_0000": {"ready": True, "samples": 1000, "attempts": 11},
                      "shard_0001": {"ready": False, "samples": 250, "attempts": 7}}
            dashboard.render("RUNNING", config, status, now - 60, now - 12, now - 3,
                             1, 123, 70, 2, "attempt.log", force=True)
            output = stream.getvalue()
            self.assertIn("Shards    1/2", output)
            self.assertIn("Samples    1250/2000", output)
            self.assertIn("00:R", output)
            self.assertIn("01:250", output)
            self.assertIn("automatic restart in 00:09:47", output)

    def test_stall_stops_and_restarts_the_same_command(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(folder)
            config.update(no_progress_seconds=.04, monitor_interval_seconds=.01,
                          status_interval_seconds=10, restart_delay_seconds=0,
                          shutdown_grace_seconds=.1, shutdown_kill_seconds=.1,
                          max_restarts=1, terminal_dashboard=False)
            sleeping = [sys.executable, "-c", "import time;time.sleep(30)", "--workers", "1"]
            with mock.patch("vae_data.supervisor.verify_process_tracking"), \
                 mock.patch("vae_data.supervisor.build_command", return_value=sleeping):
                self.assertEqual(supervise(config), 2)
            runs = list(config["log_dir"].glob("run_*"))
            events = [json.loads(line) for line in (runs[0] / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "launch" for row in events), 2)
            self.assertEqual(sum(row["event"] == "stalled" for row in events), 2)
            self.assertEqual(sum(row["event"] == "restart" for row in events), 1)

    @unittest.skipUnless(Path("/proc").is_dir(), "exact environment-tag scan is a Linux production feature")
    def test_cleanup_uses_unique_environment_tag_and_leaves_unrelated_process(self):
        run_id = "unit_" + os.urandom(8).hex()
        owned_env = dict(os.environ, **{MARKER: run_id})
        owned = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], env=owned_env)
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        try:
            deadline = time.monotonic() + 2
            while owned.pid not in owned_pids(run_id) and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertIn(owned.pid, owned_pids(run_id))
            self.assertEqual(terminate_owned(owned, run_id, .5, .5), [])
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            if owned.poll() is None:
                owned.kill()
            if unrelated.poll() is None:
                unrelated.terminate()
            owned.wait()
            unrelated.wait()


if __name__ == "__main__":
    unittest.main()
