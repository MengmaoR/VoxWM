import http.client
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from voxel_vae.early_stopping import (EarlyStoppingState, state_from_metrics,
                                      validate_early_stopping_config)
from voxel_vae.training_curves import read_curve_data, write_training_curves
from voxel_vae.training_monitor import build_status


CONFIG = validate_early_stopping_config({
    "enabled": True, "monitor": "validation_loss_total", "mode": "min",
    "patience_validations": 2, "min_delta": 0.001, "min_steps": 500,
})


def validation(step, total):
    return {
        "phase": "E00-Masked", "global_step": step, "event": "validation",
        "loss_components": {
            "loss_total": total, "loss_reconstruction": total - 0.01,
            "loss_static": total - 0.2, "loss_dynamic_unchanged": total - 0.1,
            "loss_dynamic_changed": total + 0.3,
            "dynamic_changed_by_category": {
                "D1": total, "D2": total + 0.1,
                "D3": total + 0.2, "D4": total + 0.3,
            },
        },
        "static_preservation": {"exact_token_accuracy": 0.8},
        "dynamic_reconstruction": {
            "changed_voxel_exact_accuracy": 0.2,
            "unchanged_voxel_exact_accuracy": 0.75,
        },
    }


class TrainingMonitorTests(unittest.TestCase):
    def test_early_stopping_requires_two_non_improving_validations(self):
        state = EarlyStoppingState()
        self.assertEqual(state.observe(500, 3.0, CONFIG), (True, False))
        self.assertEqual(state.observe(1000, 3.2, CONFIG), (False, False))
        self.assertEqual(state.observe(1500, 3.1, CONFIG), (False, True))
        self.assertEqual(state.best_step, 500)
        self.assertEqual(state.stop_step, 1500)

    def test_min_delta_resets_patience_only_for_real_improvement(self):
        state = EarlyStoppingState()
        state.observe(500, 3.0, CONFIG)
        self.assertEqual(state.observe(1000, 2.9995, CONFIG), (True, False))
        self.assertEqual(state.bad_validations, 1)
        self.assertEqual(state.best_step, 1000)
        self.assertEqual(state, EarlyStoppingState.from_dict(state.to_dict(CONFIG)))
        improved, stopped = state.observe(1500, 2.5, CONFIG)
        self.assertTrue(improved)
        self.assertFalse(stopped)
        self.assertEqual(state.bad_validations, 0)

    def test_history_reconstruction_curves_and_read_only_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{
                "phase": "E00-Masked", "global_step": 25,
                "loss_total": 2.0, "loss_reconstruction": 1.99,
                "loss_static": 1.5, "loss_dynamic_unchanged": 2.1,
                "loss_dynamic_changed": 2.4, "loss_beta_kl": 0.01,
                "dynamic_changed_by_category": {
                    "D1": 2.1, "D2": 2.2, "D3": 2.3, "D4": 2.4},
            }, validation(500, 3.0), validation(1000, 3.2)]
            metrics = root / "metrics.jsonl"
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
            state = state_from_metrics(metrics, CONFIG)
            (root / "early_stopping.json").write_text(json.dumps(
                state.to_dict(CONFIG)))
            (root / "run_metadata.json").write_text(json.dumps({"status": "RUNNING"}))
            svg, data_path = write_training_curves(metrics, root)
            self.assertIn("Training Dynamic-Changed CE by category", svg.read_text())
            data = json.loads(data_path.read_text())
            self.assertEqual(data["train"][0]["changed_D4"], 2.4)
            self.assertEqual(read_curve_data(metrics)["validation"][-1]["loss_total"], 3.2)
            status = build_status(root)
            self.assertEqual(status["status"], "RUNNING")
            self.assertEqual(status["latest_step"], 1000)
            self.assertEqual(status["best_step"], 500)
            self.assertEqual(status["bad_validations"], 1)

    def test_resume_ignores_uncheckpointed_steps_and_torn_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            rows = [validation(500, 3.0), validation(1000, 3.2), validation(1500, 3.3)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"global_step":')
            saved = state_from_metrics(path, CONFIG, maximum_step=1000)
            self.assertEqual(saved.bad_validations, 1)
            self.assertFalse(saved.stopped)
            with path.open("a") as stream:
                stream.write("\n" + json.dumps({"global_step": 1000, "event": "resume"}) + "\n")
                stream.write(json.dumps(validation(1500, 2.0)) + "\n")
            self.assertEqual(state_from_metrics(path, CONFIG).best_step, 1500)
            self.assertEqual([point["loss_total"] for point in read_curve_data(path)["validation"]],
                             [3.0, 3.2, 2.0])

    def test_http_monitor_reads_logs_without_creating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = root / "metrics.jsonl"
            content = json.dumps(dict(validation(1000, 3.0), phase="E01-Masked")) + "\n"
            metrics.write_text(content)
            with subprocess.Popen([
                    sys.executable, "-B", "-m", "voxel_vae.training_monitor",
                    "--port", "0", "--output-dir", str(root),
            ], stdout=subprocess.PIPE, text=True) as process:
                try:
                    address = process.stdout.readline().strip()
                    port = int(address.rsplit(":", 1)[-1])
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                    try:
                        for path, expected in (("/", b"Masked VAE training"),
                                               ("/api/status", b'"phase": "E01-Masked"'),
                                               ("/loss_curves.svg", b"E01-Masked loss curves"),
                                               ("/loss_curves.json", b'"loss_total": 3.0')):
                            connection.request("GET", path)
                            response = connection.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assertIn(expected, response.read())
                    finally:
                        connection.close()
                finally:
                    process.terminate()
                    process.wait(timeout=3)
            self.assertEqual(list(root.iterdir()), [metrics])
            self.assertEqual(metrics.read_text(), content)


if __name__ == "__main__":
    unittest.main()
