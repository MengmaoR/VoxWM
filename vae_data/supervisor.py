"""External restart supervisor for one sharded native collection invocation.

The supervisor is deliberately outside the collector.  It tags every process
started for one attempt, watches shard-local durable progress, and never uses a
program-name match when stopping a stalled run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .sharded.protocol import CATEGORIES


ROOT = Path(__file__).resolve().parents[1]
MARKER = "VOXWM_SUPERVISOR_ID"
DEFAULTS = {
    "shard": 0,
    "count": 100,
    "workers": 64,
    "resume": True,
    "wall_seconds": 86400,
    "collector_stall_seconds": 0,
    "no_progress_seconds": 600,
    "monitor_interval_seconds": 5,
    "status_interval_seconds": 30,
    "terminal_dashboard": True,
    "restart_delay_seconds": 10,
    "shutdown_grace_seconds": 10,
    "shutdown_kill_seconds": 10,
    "max_restarts": 0,
    "restart_on_exit": True,
    "renderer": "software",
    "seed": 42,
    "max_attempts": 5000,
    "sites_per_world": 8,
    "tick_batch_size": 24,
    "only_category": None,
    "craftium_dir": "third_party/craftium",
    "environment": {},
}
ALLOWED = set(DEFAULTS) | {"dataset", "log_dir"}


def _positive(name, value, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Supervisor config must be a JSON object")
    unknown = sorted(set(raw) - ALLOWED)
    if unknown:
        raise ValueError("Unknown supervisor config fields: " + ", ".join(unknown))
    if "dataset" not in raw:
        raise ValueError("Supervisor config requires dataset")
    config = dict(DEFAULTS)
    config.update(raw)
    for key in ("shard", "count", "workers", "max_attempts", "sites_per_world"):
        _positive(key, config[key], allow_zero=key == "shard")
        if not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    if config["shard"] + config["count"] > 100:
        raise ValueError("shard + count must not exceed 100")
    for key in ("wall_seconds", "no_progress_seconds", "monitor_interval_seconds",
                "status_interval_seconds", "shutdown_grace_seconds", "shutdown_kill_seconds"):
        _positive(key, config[key])
    for key in ("collector_stall_seconds", "restart_delay_seconds", "max_restarts"):
        _positive(key, config[key], allow_zero=True)
    if not isinstance(config["max_restarts"], int):
        raise ValueError("max_restarts must be an integer")
    if config["tick_batch_size"] not in (1, 24, 48):
        raise ValueError("tick_batch_size must be 1, 24 or 48")
    if config["only_category"] is not None and config["only_category"] not in CATEGORIES:
        raise ValueError("only_category must be null or one of " + ", ".join(CATEGORIES))
    if config["renderer"] not in ("auto", "window", "software"):
        raise ValueError("renderer must be auto, window or software")
    if config["resume"] is not True:
        raise ValueError("resume must remain true so a supervised restart can preserve committed data")
    for key in ("restart_on_exit", "terminal_dashboard"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(config["environment"], dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in config["environment"].items()):
        raise ValueError("environment must contain string keys and values")
    config["dataset"] = _root_path(config["dataset"])
    config["craftium_dir"] = _root_path(config["craftium_dir"])
    if config.get("log_dir"):
        config["log_dir"] = _root_path(config["log_dir"])
    else:
        dataset = config["dataset"]
        config["log_dir"] = dataset.parent / (dataset.name + "_supervisor")
    return config


def _root_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def incomplete_target_count(config):
    if config.get("only_category"):
        category=config["only_category"];quota=CATEGORIES[category]
        return sum(row["categories"].get(category,0)<quota
                   for row in shard_status(config).values())
    root = config["dataset"] / "shards"
    return sum(not (root / f"shard_{shard:04d}" / "READY").is_file()
               for shard in range(config["shard"], config["shard"] + config["count"]))


def build_command(config, python=None):
    # Keep the venv entry point. Resolving a uv/venv symlink would bypass that
    # environment and can lose Craftium and dataset dependencies on restart.
    python = str(python or sys.executable)
    # The production default is at most 64 worlds and exactly one world for each
    # unfinished shard.  The collector's allocation therefore receives no more
    # workers than active shards, including near the end of a resumed run.
    workers = max(1, min(config["workers"], incomplete_target_count(config)))
    command = [python, "-m", "vae_data", "generate",
        "--dataset", str(config["dataset"]),
        "--shard", str(config["shard"]),
        "--count", str(config["count"]),
        "--workers", str(workers),
        "--wall-seconds", str(config["wall_seconds"]),
        "--stall-seconds", str(config["collector_stall_seconds"]),
        "--max-attempts", str(config["max_attempts"]),
        "--sites-per-world", str(config["sites_per_world"]),
        "--tick-batch-size", str(config["tick_batch_size"]),
        "--renderer", config["renderer"],
        "--seed", str(config["seed"]),
        "--craftium-dir", str(config["craftium_dir"]),
        "--resume"]
    if config.get("only_category"):
        command.extend(("--only-category",config["only_category"]))
    return command


def progress_paths(config):
    root = config["dataset"] / "shards"
    paths = []
    for shard in range(config["shard"], config["shard"] + config["count"]):
        name = f"shard_{shard:04d}"
        for folder in (root / name, root / ("." + name + ".partial")):
            paths.extend(folder / item for item in
                         ("cursor.json", "progress", "READY"))
    return paths


def progress_signature(config):
    signature = []
    for path in progress_paths(config):
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            pass
    return tuple(signature)


def targets_ready(config):
    root = config["dataset"] / "shards"
    return all((root / f"shard_{shard:04d}" / "READY").is_file()
               for shard in range(config["shard"], config["shard"] + config["count"]))


def targets_complete(config,status=None):
    if not config.get("only_category"):
        return targets_ready(config)
    status=status or shard_status(config);category=config["only_category"]
    return all(row["categories"].get(category,0)>=CATEGORIES[category]
               for row in status.values())


def shard_status(config):
    root = config["dataset"] / "shards"
    result = {}
    for shard in range(config["shard"], config["shard"] + config["count"]):
        name = f"shard_{shard:04d}"
        final, partial = root / name, root / ("." + name + ".partial")
        folder = final if final.exists() else partial
        row = {"ready": (final / "READY").is_file(), "samples": 0,
               "attempts": 0, "categories": {category:0 for category in CATEGORIES}}
        if row["ready"]:
            row["samples"] = 1000
            row["categories"] = dict(CATEGORIES)
            statistics = final / "statistics.json"
            try:
                row["attempts"] = int(json.loads(statistics.read_text()).get("attempts", 0))
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                pass
            result[name] = row
            continue
        database = folder / "journal.sqlite"
        if database.is_file():
            connection = None
            try:
                connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro",
                                             uri=True, timeout=.05)
                connection.execute("PRAGMA query_only=ON")
                row["samples"] = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
                row["attempts"] = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                row["categories"].update(connection.execute(
                    "SELECT category,COUNT(*) FROM samples GROUP BY category").fetchall())
            except (OSError, sqlite3.Error):
                row["read_busy"] = True
            finally:
                if connection is not None:
                    connection.close()
        result[name] = row
    return result


def status_totals(status):
    return {"ready": sum(row["ready"] for row in status.values()),
            "samples": sum(row["samples"] for row in status.values()),
            "attempts": sum(row["attempts"] for row in status.values()),
            "read_busy": sum(bool(row.get("read_busy")) for row in status.values()),
            "categories": {category:sum(row["categories"].get(category,0)for row in status.values())
                           for category in CATEGORIES}}


def _duration(seconds):
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _bar(value, total, width=30):
    ratio = 0 if total <= 0 else max(0., min(1., value / total))
    filled = round(width * ratio)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _shard_grid(status, columns=10):
    cells = []
    for name, row in status.items():
        shard = int(name.rsplit("_", 1)[1])
        if row["ready"]:
            value = "R"
        elif row.get("read_busy"):
            value = "BUSY"
        elif row["samples"]:
            value = str(row["samples"])
        else:
            value = "---"
        cells.append(f"{shard:02d}:{value}".ljust(8))
    return ["".join(cells[i:i + columns]).rstrip()
            for i in range(0, len(cells), columns)]


class TerminalDashboard:
    """ANSI dashboard for a TTY; concise periodic records for redirected output."""
    def __init__(self, enabled=True, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.drawn = False
        self.baseline_samples = None
        self.baseline_time = None

    def render(self, phase, config, status, supervisor_started, last_progress,
               status_updated, restarts, pid, owned_count, active_worlds,
               attempt_log, force=False, reason=None):
        if not self.enabled:
            return
        now = time.monotonic()
        totals = status_totals(status)
        shard_target = len(status)
        sample_target = shard_target * 1000
        progress_age = max(0., now - last_progress)
        stall_left = max(0., config["no_progress_seconds"] - progress_age)
        focus = config.get("only_category")
        focus_value = totals["categories"].get(focus,0) if focus else 0
        focus_target = CATEGORIES[focus] * shard_target if focus else 0
        measured_value = focus_value if focus else totals["samples"]
        measured_target = focus_target if focus else sample_target
        if self.baseline_samples is None:
            self.baseline_samples = measured_value
            self.baseline_time = now
        measured = max(1e-9, now - self.baseline_time)
        rate = max(0., measured_value - self.baseline_samples) / measured * 3600
        remaining = max(0, measured_target - measured_value)
        eta = _duration(remaining / rate * 3600) if rate else "--"
        focus_summary = (f" focus={focus}:{focus_value}/{focus_target}" if focus else "")
        if not self.tty:
            if force:
                self.stream.write(
                    f"[dashboard] phase={phase} shards={totals['ready']}/{shard_target} "
                    f"samples={totals['samples']}/{sample_target} attempts={totals['attempts']} "
                    f"owned={owned_count} worlds={active_worlds} restarts={restarts} "
                    f"rate={rate:.1f}/h eta={eta} progress_age={int(progress_age)}s "
                    f"stall_in={int(stall_left)}s" + focus_summary
                    + (f" reason={reason}" if reason else "") + "\n")
                self.stream.flush()
            return
        lines = [
            "VoxWM Native Collection Supervisor",
            f"{phase:<10} uptime {_duration(now - supervisor_started)} | restart {restarts} | "
            f"collector pid {pid or '-'} | owned processes {owned_count}",
            f"Shards  {totals['ready']:>3}/{shard_target:<3} {_bar(totals['ready'], shard_target)}",
            f"Samples {totals['samples']:>7}/{sample_target:<7} "
            f"{_bar(totals['samples'], sample_target)}",
        ]
        if focus:
            lines.append(f"Focus {focus:<6} {focus_value:>7}/{focus_target:<7} "
                         f"{_bar(focus_value,focus_target)}")
        lines.extend((
            f"Rate {rate:.1f} {focus or 'all'} samples/hour | ETA {eta}",
            f"Attempts {totals['attempts']} | requested worlds {active_worlds} "
            f"(max {config['workers']}, 1/shard) | busy reads {totals['read_busy']}",
            f"Last durable progress {int(progress_age)}s ago | automatic restart in {_duration(stall_left)} "
            f"| counts age {int(now - status_updated)}s",
            f"Dataset: {config['dataset']}",
            f"Attempt log: {attempt_log or '-'}",
            "Shard grid: R=READY, number=committed samples, ---=not started",
            *_shard_grid(status),
        ))
        if reason:
            lines.append("Reason: " + str(reason))
        prefix = "\x1b[2J\x1b[H" if not self.drawn else "\x1b[H"
        self.stream.write(prefix + "\n".join(lines) + "\x1b[J\n")
        self.stream.flush()
        self.drawn = True


def _linux_tagged_pids(run_id, proc=Path("/proc")):
    proc = Path(proc)
    if not proc.is_dir():
        return None
    wanted = f"{MARKER}={run_id}".encode()
    uid = os.geteuid()
    found = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            values = (entry / "environ").read_bytes().split(b"\0")
            if wanted in values:
                found.add(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            pass
    return found


def _ps_tagged_pids(run_id):
    """Same-user fallback for macOS development; production Linux uses /proc."""
    marker = f"{MARKER}={run_id}"
    try:
        output = subprocess.check_output(
            ["ps", "eww", "-axo", "pid=,uid=,command="], text=True,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return set()
    found = set()
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and fields[1].isdigit() and int(fields[1]) == os.geteuid() and marker in fields[2]:
            found.add(int(fields[0]))
    return found


def owned_pids(run_id):
    found = _linux_tagged_pids(run_id)
    return _ps_tagged_pids(run_id) if found is None else found


def terminate_owned(process, run_id, grace_seconds=10, kill_seconds=10):
    """Stop only processes carrying this launch's unguessable environment tag.

    A new collection is not started if an uninterruptible tagged process remains.
    """
    supervisor_pid = os.getpid()
    for sig, duration in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_seconds)):
        deadline = time.monotonic() + duration
        while True:
            pids = owned_pids(run_id) - {supervisor_pid}
            if process.poll() is None:
                pids.add(process.pid)
            if not pids:
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    pass
                return []
            # Stop descendants before the top process; repeated tag scans also
            # catch a child created while SIGTERM is being delivered.
            ordered = sorted(pids - {process.pid}, reverse=True) + ([process.pid] if process.pid in pids else [])
            for pid in ordered:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(.2)
    return sorted(owned_pids(run_id) - {supervisor_pid})


def verify_process_tracking(timeout=2):
    """Fail closed before collection if exact process ownership is unavailable."""
    run_id = "probe_" + uuid.uuid4().hex
    environment = dict(os.environ)
    environment[MARKER] = run_id
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
        env=environment, start_new_session=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    found = False
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.pid in owned_pids(run_id):
                found = True
                break
            if process.poll() is not None:
                break
            time.sleep(.02)
    finally:
        lingering = terminate_owned(process, run_id, .5, .5)
    if not found:
        raise RuntimeError("Cannot verify exact environment-tag process ownership; collection was not started")
    if lingering:
        raise RuntimeError("Process-ownership probe could not be cleaned up: " + ", ".join(map(str, lingering)))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class Recorder:
    def __init__(self, folder, config, command):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.root = Path(folder) / f"run_{stamp}_{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.events = open(self.root / "events.jsonl", "a", buffering=1)
        serial = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
        _atomic_json(self.root / "config.json", serial)
        _atomic_json(self.root / "command.json", {"argv": command})

    def event(self, kind, **fields):
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **fields}
        self.events.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def state(self, **fields):
        _atomic_json(self.root / "state.json", fields)

    def close(self):
        self.events.close()


def _acquire_dataset_lock(config):
    dataset = config["dataset"]
    dataset.parent.mkdir(parents=True, exist_ok=True)
    path = dataset.parent / ("." + dataset.name + ".collection_supervisor.lock")
    stream = open(path, "a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"Another supervisor already owns dataset {dataset}")
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
    stream.flush()
    return stream


def supervise(config):
    verify_process_tracking()
    command = build_command(config)
    lock = _acquire_dataset_lock(config)
    recorder = Recorder(config["log_dir"], config, command)
    dashboard = TerminalDashboard(config["terminal_dashboard"])
    stop = threading.Event()
    old_handlers = {}

    def request_stop(signum, _frame):
        recorder.event("signal", signal=signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    restarts = 0
    process = None
    run_id = None
    exit_code = 1
    supervisor_started = last_progress = status_updated = time.monotonic()
    status = shard_status(config)
    pids = []
    active_worlds = 0
    log_path = None
    final_reason = None
    try:
        while not stop.is_set():
            if targets_complete(config,status):
                status = shard_status(config);status_updated = time.monotonic()
                recorder.event("complete", restarts=restarts, shards=status)
                exit_code = 0
                if config.get("only_category"):
                    final_reason = config["only_category"] + " quota complete"
                break
            command = build_command(config)
            active_worlds = int(command[command.index("--workers") + 1])
            run_id = uuid.uuid4().hex
            environment = dict(os.environ, **config["environment"])
            environment[MARKER] = run_id
            log_path = recorder.root / f"attempt_{restarts:04d}.log"
            log = open(log_path, "ab", buffering=0)
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            launched = last_progress = last_status = time.monotonic()
            signature = progress_signature(config)
            status = shard_status(config);status_updated = launched
            recorder.event("launch", restart=restarts, pid=process.pid, run_id=run_id,
                           log=log_path.name, argv=command)
            dashboard.render("RUNNING", config, status, supervisor_started, last_progress,
                status_updated, restarts, process.pid, 1, active_worlds, log_path, force=True)
            recorder.state(status="RUNNING", restart=restarts, pid=process.pid,
                           run_id=run_id, owned_pids=[process.pid], progress_age_seconds=0,
                           shards=status, totals=status_totals(status), attempt_log=str(log_path))
            reason = None
            while not stop.is_set():
                try:
                    code = process.wait(timeout=config["monitor_interval_seconds"])
                except subprocess.TimeoutExpired:
                    code = None
                now = time.monotonic()
                current = progress_signature(config)
                if current != signature:
                    signature = current
                    last_progress = now
                    recorder.event("progress", restart=restarts)
                pids = sorted(owned_pids(run_id))
                refreshed = False
                if now - last_status >= config["status_interval_seconds"]:
                    last_status = now
                    status = shard_status(config)
                    status_updated = now
                    refreshed = True
                    recorder.state(status="RUNNING", restart=restarts, pid=process.pid,
                                   run_id=run_id, owned_pids=pids,
                                   progress_age_seconds=round(now - last_progress, 1),
                                   shards=status, totals=status_totals(status),
                                   attempt_log=str(log_path))
                dashboard.render("RUNNING", config, status, supervisor_started, last_progress,
                    status_updated, restarts, process.pid, len(pids), active_worlds,
                    log_path, force=refreshed)
                if code is not None:
                    reason = f"collector exited with code {code} before all requested shards were READY"
                    status = shard_status(config);status_updated = now
                    recorder.event("collector_exit", restart=restarts, code=code,
                                   runtime_seconds=round(now - launched, 3), shards=status)
                    if targets_complete(config,status):
                        reason = None
                        exit_code = 0
                        if config.get("only_category"):
                            final_reason = config["only_category"] + " quota complete"
                    elif not config["restart_on_exit"]:
                        exit_code = code or 1
                    dashboard.render("COMPLETE" if reason is None else "EXITED", config,
                        status, supervisor_started, last_progress, status_updated, restarts,
                        process.pid, len(pids), active_worlds, log_path, force=True, reason=reason)
                    break
                if now - last_progress >= config["no_progress_seconds"]:
                    reason = f"no completed attempt, sample commit or READY marker for {config['no_progress_seconds']}s"
                    status = shard_status(config);status_updated = now
                    recorder.event("stalled", restart=restarts, reason=reason,
                                   owned_pids=pids, shards=status)
                    dashboard.render("STALLED", config, status, supervisor_started,
                        last_progress, status_updated, restarts, process.pid, len(pids),
                        active_worlds, log_path, force=True, reason=reason)
                    break
            lingering = terminate_owned(process, run_id,
                config["shutdown_grace_seconds"], config["shutdown_kill_seconds"])
            log.close()
            if lingering:
                final_reason = "Tagged collection processes survived SIGKILL"
                recorder.event("cleanup_failed", run_id=run_id, lingering_pids=lingering)
                recorder.state(status="FAIL", reason="tagged processes survived SIGKILL",
                               lingering_pids=lingering)
                print("Supervisor refused to restart because owned processes remain: " +
                      ", ".join(map(str, lingering)), file=sys.stderr)
                exit_code = 2
                break
            if stop.is_set():
                exit_code = 130
                break
            if reason is None:
                status = shard_status(config);status_updated = time.monotonic()
                recorder.event("complete", restarts=restarts, shards=status)
                if config.get("only_category"):
                    final_reason = config["only_category"] + " quota complete"
                break
            if process.returncode is not None and not config["restart_on_exit"]:
                final_reason = reason
                recorder.state(status="FAIL", reason=reason, shards=shard_status(config))
                print(reason, file=sys.stderr)
                break
            if config["max_restarts"] and restarts >= config["max_restarts"]:
                final_reason = f"Restart limit reached after: {reason}"
                recorder.event("restart_limit", restarts=restarts, reason=reason)
                exit_code = 2
                break
            restarts += 1
            recorder.event("restart", restart=restarts, reason=reason,
                           delay_seconds=config["restart_delay_seconds"])
            dashboard.render("RESTARTING", config, status, supervisor_started,
                last_progress, status_updated, restarts, None, 0, 0, log_path,
                force=True, reason=reason)
            stop.wait(config["restart_delay_seconds"])
        final = "COMPLETE" if exit_code == 0 else "INTERRUPTED" if exit_code == 130 else "FAIL"
        status = shard_status(config);status_updated = time.monotonic()
        if stop.is_set():final_reason = "Supervisor interrupted by user"
        dashboard.render(final, config, status, supervisor_started, last_progress,
            status_updated, restarts, None, 0, 0, log_path, force=True,
            reason=final_reason)
        recorder.state(status=final, restarts=restarts, shards=status,
                       totals=status_totals(status), reason=final_reason)
        return exit_code
    finally:
        if process is not None and run_id is not None and (process.poll() is None or owned_pids(run_id)):
            lingering = terminate_owned(process, run_id,
                config["shutdown_grace_seconds"], config["shutdown_kill_seconds"])
            if lingering:
                recorder.event("final_cleanup_failed", run_id=run_id, lingering_pids=lingering)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        recorder.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--check-config", action="store_true",
                   help="Validate and print the resolved command without starting collection")
    return p.parse_args(argv)


def main(argv=None):
    args = parser(argv)
    try:
        config = load_config(args.config)
        command = build_command(config)
        if args.check_config:
            print(json.dumps({"command": command,
                "dataset": str(config["dataset"]), "log_dir": str(config["log_dir"]),
                "defaults": {"count": config["count"], "workers": config["workers"],
                             "no_progress_seconds": config["no_progress_seconds"],
                             "only_category": config["only_category"],
                             "terminal_dashboard": config["terminal_dashboard"],
                             "status_interval_seconds": config["status_interval_seconds"]}}, indent=2))
            return 0
        return supervise(config)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Supervisor configuration/runtime error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
