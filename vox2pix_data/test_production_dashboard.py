"""Small read-only progress checks; no native sessions or dataset generation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from .dashboard import production_rates, show_dashboard, v3_progress_lines
from .dashboard_processes import collector_process


SLOTS = [dict(scene_id=0, kind="surface", biome="forest"),
         dict(scene_id=1, kind="cave", biome="forest"),
         dict(scene_id=2, kind="underwater")]
CONFIG = dict(mode="production", views_per_scene=5)
JOB = dict(pid=9999, workers=8, live_workers=2, worlds=[])


def state(**changes):
    result = dict(status="RUNNING", pid=9999, started_at=100., updated_at=700., workers=8,
                  completed=1, initial_completed=0, counts={"surface": 1},
                  biome_counts={"forest": {"surface": 1, "cave": 0}},
                  attempts=2, failed=1, exhausted_count=0, last_error="scene 1: no legal cave",
                  last_commit_at=650., commit_times=[650.],
                  active=[dict(scene_id=1, kind="cave", biome="forest", started_at=680.)])
    result.update(changes)
    return result


class ProductionDashboardTests(TestCase):
    def test_progress_uses_scene_counts_and_displays_active_biome(self):
        text = "\n".join(v3_progress_lines(Path("data"), SLOTS, CONFIG, set(), [],
                                         state(), JOB, 100., 700.))
        self.assertIn("Scenes 1/3", text)
        self.assertIn("RGB frames 5/15", text)
        self.assertIn("surface 1/1 | cave 0/1 | underwater 0/1", text)
        self.assertIn("in-flight scenes 1", text)
        self.assertIn("scene 0001 | cave | forest", text)
        self.assertIn("no fixed biome quota", text)
        self.assertIn("failed attempts 1", text)
        self.assertIn("Last error: scene 1", text)

    def test_resume_average_does_not_count_old_commits(self):
        average, recent = production_rates(state(completed=1002, initial_completed=1000,
                                                  commit_times=[650., 690.]), 100., 700.)
        self.assertEqual(average, 12.)
        self.assertEqual(recent, 12.)
        self.assertEqual(production_rates(state(completed=1000, initial_completed=1000,
                                                commit_times=[]), 100., 700.), (None, None))

    def test_exhausted_slots_do_not_get_an_eta(self):
        text = "\n".join(v3_progress_lines(Path("data"), SLOTS, CONFIG, set(), [],
                                         state(exhausted_count=1, commit_times=[650., 690.]),
                                         JOB, 100., 700.))
        self.assertIn("exhausted slots 1", text)
        self.assertIn("ETA --", text)

    def test_once_reads_atomic_progress_without_scanning_scene_data(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(mode="smoke", scene_kinds=["surface", "cave", "underwater"],
                          scenes_per_kind=1, views_per_scene=5)
            (root / "manifest.json").write_text(json.dumps(dict(config=config)))
            (root / "collection_progress.json").write_text(json.dumps(
                state(status="COMPLETE", pid=1, completed=3,
                      counts={kind: 1 for kind in config["scene_kinds"]}, active=[])))
            output = io.StringIO()
            with patch("vox2pix_data.dashboard.collector_process", return_value=None), \
                    patch("vox2pix_data.dashboard.committed_paths", side_effect=AssertionError("unexpected scan")), \
                    redirect_stdout(output):
                self.assertEqual(show_dashboard(root, once=True), 0)
            self.assertIn("COMPLETE", output.getvalue())
            self.assertIn("Scenes 3/3", output.getvalue())

    def test_linux_process_lookup_accepts_auto_workers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            entry = root / "proc/123"
            entry.mkdir(parents=True)
            (entry / "cwd").symlink_to(root)
            (entry / "cmdline").write_bytes(b"python3\0-m\0vox2pix_data\0collect\0--dataset\0data\0--workers\0auto\0")
            with patch("vox2pix_data.dashboard_processes.os.sched_getaffinity", return_value={1, 2, 3}, create=True):
                job = collector_process(root / "data", root / "proc")
            self.assertEqual(job["pid"], 123)
            self.assertEqual(job["workers"], 3)


if __name__ == "__main__":
    main()
