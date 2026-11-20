"""Read-only auto-refreshing web monitor for Masked VAE training."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .training_curves import read_curve_data, render_svg


HTML = '''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxWM · Masked VAE training</title>
<style>
body { margin: 0; background: #0d141d; color: #dce8f1; font: 14px system-ui; }
main { padding: 20px; max-width: 1200px; margin: auto; }
h1 { font-size: 22px; margin: 0 0 16px; }
#cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 8px; margin: 0; }
#cards div { background: #15212d; border: 1px solid #304456; border-radius: 6px; padding: 10px 12px; }
dt { color: #a7bfd0; }
dd { font: 16px ui-monospace, monospace; margin: 5px 0 0; overflow-wrap: anywhere; }
img { width: 100%; margin: 15px 0; border: 1px solid #304456; }
pre { background: #111b25; padding: 12px; overflow: auto; }
</style>
<main>
  <h1 id="heading">VoxWM · Masked VAE training</h1>
  <dl id="cards"></dl>
  <p id="notice" role="status"></p>
  <img id="curve" alt="Training and validation loss by optimizer step">
  <details><summary>Latest loss components and run details</summary><pre id="detail"></pre></details>
</main>
<script>
const fields = ["Logged status", "Optimizer step", "Workers", "Train pool", "Training total",
                "Validation total", "Best validation", "Checks without improvement", "Log age (s)"];
const cards = document.getElementById("cards");
const cells = fields.map(label => {
  const group = document.createElement("div");
  const term = document.createElement("dt");
  const value = document.createElement("dd");
  term.textContent = label;
  group.append(term, value);
  cards.append(group);
  return value;
});
const notice = document.getElementById("notice");
let revision;
const loss = value => value == null ? "—" : value.toFixed(5);

async function refresh() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const run = await response.json();
    if (run.phase) {
      document.title = `VoxWM · ${run.phase} training`;
      document.getElementById("heading").textContent = document.title;
    }
    const step = run.latest_step == null ? "—" : `${run.latest_step} / ${run.max_steps ?? "—"}`;
    const best = run.best_step == null ? loss(run.best_validation_loss) :
      `${loss(run.best_validation_loss)} @ ${run.best_step}`;
    const patience = run.bad_validations == null ? "—" :
      `${run.bad_validations} / ${run.patience_validations}`;
    const pool = run.training_pool?.samples?.toLocaleString() ?? "—";
    const values = [run.status, step, run.world_size ?? "—", pool, loss(run.latest_train_loss),
                    loss(run.latest_validation_loss), best, patience, run.log_age_seconds ?? "—"];
    cells.forEach((cell, index) => { cell.textContent = values[index]; });
    document.getElementById("detail").textContent = JSON.stringify(run, null, 2);
    notice.textContent = run.error || run.stop_reason || "";
    if (run.metrics_revision !== revision) {
      revision = run.metrics_revision;
      document.getElementById("curve").src = `/loss_curves.svg?v=${revision ?? 0}`;
    }
  } catch (error) {
    notice.textContent = `Unable to read metrics (${error.message})`;
  }
  setTimeout(refresh, 5000);
}
refresh();
</script>
</html>'''


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_status(output_dir):
    output = Path(output_dir)
    metrics = output / "metrics.jsonl"
    try:
        stat = metrics.stat()
    except FileNotFoundError:
        stat = None
    curves = read_curve_data(metrics)
    metadata = _read_json(output / "run_metadata.json")
    early = _read_json(output / "early_stopping.json")
    train = curves["train"][-1] if curves["train"] else {}
    validation = curves["validation"][-1] if curves["validation"] else {}
    age = None if stat is None else max(0.0, time.time() - stat.st_mtime)
    best = min(curves["validation"], key=lambda row: row["loss_total"], default={})
    return {
        "phase": metadata.get("phase") or curves["phase"],
        "training_pool": metadata.get("training_pool"),
        "validation_states": metadata.get("validation_states"),
        "world_size": metadata.get("world_size"),
        "cuda_visible_devices": metadata.get("cuda_visible_devices"),
        "status": metadata.get("status", "HISTORY" if stat else "WAITING"),
        "latest_step": max((step for step in (
            train.get("step"), validation.get("step"), metadata.get("global_step"))
            if step is not None), default=None),
        "max_steps": metadata.get("max_steps"),
        "latest_train_loss": train.get("loss_total"),
        "latest_train_components": train,
        "latest_validation_loss": validation.get("loss_total"),
        "latest_validation": validation,
        "best_validation_loss": early.get("best_metric", best.get("loss_total")),
        "best_step": early.get("best_step", best.get("step")),
        "bad_validations": early.get("bad_validations"),
        "patience_validations": early.get("config", {}).get("patience_validations"),
        "early_stopped": early.get("stopped", False),
        "stop_reason": early.get("stop_reason"),
        "error": metadata.get("error"),
        "log_age_seconds": None if age is None else round(age, 1),
        "metrics_revision": None if stat is None else str(stat.st_mtime_ns),
        "output_dir": str(output.resolve()),
    }


def serve(output_dir, host, port):
    output = Path(output_dir).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                body, content = HTML.encode(), "text/html; charset=utf-8"
            elif path == "/api/status":
                body = json.dumps(build_status(output), allow_nan=False).encode()
                content = "application/json"
            elif path == "/loss_curves.svg":
                body = render_svg(read_curve_data(output / "metrics.jsonl")).encode()
                content = "image/svg+xml"
            elif path == "/loss_curves.json":
                body = json.dumps(
                    read_curve_data(output / "metrics.jsonl"),
                    allow_nan=False).encode()
                content = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    print(f"Masked training monitor: http://{host}:{server.server_port}", flush=True)
    print(f"Output: {output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/voxel_vae/e01/phase_masked"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args(argv)
    serve(args.output_dir, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
