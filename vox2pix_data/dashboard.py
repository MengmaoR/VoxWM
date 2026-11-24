"""Read-only terminal progress for committed scenes and active native workers."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import shutil
import sys
import time

from vae_data.supervisor import _bar, _duration

from .dashboard_processes import collector_process
from .native import scene_id_for_world_seed
from .protocol import HEIGHTS, V3_KINDS, plan, v3_plan
from .storage import committed_paths, read_json

ENDED = re.compile(rb"^\d+/\d+ scenes committed; (\d+) slots remain unfilled\.$")
FAILED = re.compile(rb"^Scene (\d+) \S+ \S+: (.*)$")
RECENT_SECONDS = 10 * 60


class RunLog:
    def __init__(self):
        self.path = None
        self.inode = None
        self.offset = 0
        self.ended = False
        self.error = None
        self.failures = {}
        self.updated = None

    def read(self, path):
        if path is None:
            self.__init__()
            return
        stat = path.stat()
        if path != self.path or stat.st_ino != self.inode or stat.st_size < self.offset:
            self.__init__()
            self.path, self.inode = path, stat.st_ino
        if stat.st_size == self.offset:
            return
        with path.open("rb") as stream:
            stream.seek(self.offset)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                self.offset += len(line)
                message = line.rstrip()
                failure = FAILED.match(message)
                if failure:
                    scene_id = int(failure[1])
                    self.failures[scene_id] = failure[2].decode("utf-8", errors="replace")
                self.ended |= bool(ENDED.match(message))
                if message.startswith((b"Vox2Pix collect:", b"Traceback (most recent call last):")):
                    self.error = message.decode("utf-8", errors="replace")
        self.updated = stat.st_mtime


def select_log(dataset, job, explicit, initialized):
    if explicit is not None:
        return explicit
    if job and job["log"]:
        return job["log"]
    min_time = job["started"] if job and job["started"] is not None else initialized
    logs = Path("runs") / dataset.name
    recent = [(p.stat().st_mtime, p) for p in logs.glob("collect-*.log")]
    return max((row for row in recent if row[0] >= min_time - 1), default=(0, None))[1]


def rates(timestamps, started, ended):
    elapsed = max(0, ended - started)
    interval = min(elapsed, RECENT_SECONDS)
    sampled = [t for t in timestamps if started <= t <= ended]
    recent = sum(t > ended - interval for t in sampled)
    average = len(sampled) * 3600 / elapsed if elapsed and sampled else None
    rate = recent * 3600 / interval if interval >= 30 and recent >= 2 else None
    return average, rate


def progress_lines(dataset, slots, config, committed, timestamps, job, log, initialized, now, width=80):
    done = len(committed)
    total = len(slots)
    per_scene = config["views_per_scene"]
    missing = total - done
    failed = set(log.failures).intersection(slot["scene_id"] for slot in slots) - committed
    phase = "IDLE"
    if missing == 0:
        phase = "COMPLETE"
    elif job:
        phase = "RUNNING"
    elif log.ended:
        phase = "INCOMPLETE"
    elif log.error:
        phase = "ERROR"
    started = job["started"] if job and job["started"] is not None else initialized
    last = max(timestamps, default=None)
    ended = now if job else max(last or started, log.updated or started)
    average, recent = rates(timestamps, started, ended)
    eta = "00:00:00" if not missing else "--"
    if missing and job and recent and not failed:
        eta = _duration(3600 * missing / recent)
    counts = Counter(slot["height"] for slot in slots)
    accepted = Counter(slot["height"] for slot in slots if slot["scene_id"] in committed)
    elapsed = _duration(ended - started)
    lines = [
        f"Vox2Pix 1K dashboard · {phase} · elapsed {elapsed}",
        f"Scenes {done}/{total} {_bar(done, total)} | RGB frames {done * per_scene}/{total * per_scene}",
        " | ".join(f"{height} {accepted[height]}/{counts[height]}" for height in HEIGHTS),
        f"Missing {missing} | exhausted in log {len(failed)} | last commit "
        + (_duration(now - last) + " ago" if last is not None else "--"),
    ]
    average_label = f"{average:.1f}/h" if average is not None else "--"
    recent_label = f"{recent:.1f}/h" if recent is not None else "--"
    rate_line = f"Scene rate {average_label} average · {recent_label} recent (up to 10 min) | ETA {eta}"
    if missing and failed:
        rate_line += " (exhausted slots need a retry)" if job else " (collector exited with gaps)"
    lines.append(rate_line)
    lines.append(f"Dataset {dataset} | log {log.path or '--'}")

    active = set()
    if job:
        worlds = job["worlds"]
        lines.append(f"Collector PID {job['pid']} | workers {job['live_workers']}/{job['workers']} "
                     f"| active native worlds {len(worlds)}")
        for world in worlds:
            seed = world["world_seed"]
            scene_id = scene_id_for_world_seed(seed, config["seed"]) if seed is not None else None
            label = f"{scene_id:03d}" if scene_id is not None else "--"
            lines.append(f"  world {world['pid']} | scene {label} | {world['kind'] or 'starting'}")
            if scene_id is not None:
                active.add(scene_id)
    else:
        lines.append("Collector inactive | live workers 0 | active native worlds 0")
        if log.error and phase == "ERROR":
            lines.append(log.error)

    biomes = defaultdict(list)
    for slot in slots:
        biomes[slot["biome"]].append(slot)
    lines.extend(["", "Biome slots · S / Sh / D · # committed  * active  x exhausted  - pending"])
    cells = []
    for biome, rows in biomes.items():
        first = rows[0]["scene_id"]
        mask = "".join("#" if (i := slot["scene_id"]) in committed else "*"
                       if i in active else "x" if i in failed else "-" for slot in rows)
        planned = Counter(slot["height"] for slot in rows)
        selected = Counter(slot["height"] for slot in rows if slot["scene_id"] in committed)
        ratio = " ".join(f"{selected[h]}/{planned[h]}" for h in HEIGHTS)
        cells.append(f"{biome:<15} {first:03d} {mask} {ratio}")
    cell_width = max(map(len, cells)) + 2
    columns = max(1, width // cell_width)
    for start in range(0, len(cells), columns):
        lines.append("".join(cell.ljust(cell_width) for cell in cells[start:start + columns]).rstrip())

    if failed:
        lines.append("")
        lines.append("Latest exhausted slots")
        for scene_id in list(log.failures)[-3:]:
            if scene_id in failed:
                lines.append(f"  {scene_id:03d} {log.failures[scene_id][:180]}")
    return lines


def _pid_running(pid):
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def production_rates(state, started, ended):
    """Use this invocation's count for average and its retained commits for recent rate."""
    elapsed = max(0, ended - started)
    count = max(0, state["completed"] - state["initial_completed"])
    average = count * 3600 / elapsed if count and elapsed else None
    commits = [stamp for stamp in state.get("commit_times", []) if started <= stamp <= ended]
    interval = min(elapsed, RECENT_SECONDS)
    if len(commits) >= 1000:
        interval = min(interval, ended - commits[0])
    recent_count = sum(stamp > ended - interval for stamp in commits)
    recent = recent_count * 3600 / interval if interval >= 30 and recent_count >= 2 else None
    return average, recent


def v3_progress_lines(dataset, slots, config, committed, timestamps, state, job,
                      initialized, now, width=80):
    """Scene-level progress; the monitor never starts or owns collection processes."""
    state = state or {}
    total = len(slots)
    done = state.get("completed", len(committed))
    missing = total - done
    running = bool(job) or _pid_running(state.get("pid"))
    status = state.get("status", "RUNNING" if running else "IDLE")
    if not missing:
        status = "COMPLETE"
    elif status == "RUNNING" and not running:
        status = "INACTIVE (last saved state RUNNING)"
    started = state.get("started_at") or (job or {}).get("started") or initialized
    last_commit = state.get("last_commit_at") or max(timestamps, default=None)
    updated = state.get("updated_at", initialized)
    ended = now if running else max(last_commit or started, updated, started)
    average, recent = (production_rates(state, started, ended) if "initial_completed" in state
                       else rates(timestamps, started, ended))
    eta = "00:00:00" if not missing else "--"
    if running and missing and recent and not state.get("exhausted_count", 0):
        eta = _duration(3600 * missing / recent)
    targets = Counter(slot["kind"] for slot in slots)
    accepted = (Counter(state["counts"]) if "counts" in state else
                Counter(slot["kind"] for slot in slots if slot["scene_id"] in committed))
    frames = config["views_per_scene"]
    failed = state.get("failed", 0)
    active = state.get("active", [])
    lines = [
        f"Vox2Pix v3 {config['mode']} · {status} · elapsed {_duration(ended - started)}",
        f"Scenes {done}/{total} {_bar(done, total)} | RGB frames {done * frames}/{total * frames}",
        " | ".join(f"{kind} {accepted[kind]}/{targets[kind]}" for kind in V3_KINDS),
        f"Remaining {missing} | failed attempts {failed} | exhausted slots {state.get('exhausted_count', 0)} "
        f"| attempts {state.get('attempts', '--')} | last commit "
        + (_duration(max(0, now - last_commit)) + " ago" if last_commit is not None else "--"),
        "Scene rate " + (f"{average:.1f}/h" if average is not None else "--") + " average · "
        + (f"{recent:.1f}/h" if recent is not None else "--") + f" recent (up to 10 min) | ETA {eta}",
        f"Dataset {dataset} | last progress update {_duration(max(0, now - updated))} ago",
    ]
    workers = state.get("workers", (job or {}).get("workers", "--"))
    if running:
        lines.append(f"Collector PID {state.get('pid') or (job or {}).get('pid')} | worker limit {workers} "
                     f"| in-flight scenes {len(active)}")
        if job:
            lines.append(f"Live workers {job['live_workers']} | native worlds {len(job['worlds'])}")
    else:
        lines.append("Collector inactive | dashboard remains read-only")
    adaptive = state.get("adaptive")
    if adaptive:
        best = adaptive.get("best_workers")
        best_rate = adaptive.get("best_scenes_per_hour")
        best_label = f"{best} ({best_rate:.1f} scenes/h)" if best_rate is not None else "--"
        lines.append(f"Adaptive workers {adaptive['target']}/{adaptive['maximum']} ceiling "
                     f"| best observed {best_label}")
        lines.append(f"Adaptive {adaptive['phase']} / {adaptive['outcome']} "
                     f"| {adaptive.get('reason') or 'comparing committed-scene throughput'}")
    for task in active:
        scene_id = task["scene_id"]
        if scene_id in committed:
            continue
        elapsed = max(0, now - task.get("started_at", now))
        lines.append(f"  scene {scene_id:04d} | {task['kind']} | {task.get('biome') or 'natural water'} "
                     f"| {_duration(elapsed)} | {task.get('stage', 'search/capture')}")
    biomes = sorted({slot["biome"] for slot in slots if slot.get("biome") is not None})
    if biomes:
        planned = Counter((slot["biome"], slot["kind"]) for slot in slots if slot.get("biome"))
        selected = (Counter({(biome, kind): count for biome, kinds in state["biome_counts"].items()
                             for kind, count in kinds.items()}) if "biome_counts" in state else
                    Counter((slot["biome"], slot["kind"]) for slot in slots
                            if slot.get("biome") and slot["scene_id"] in committed))
        lines.extend(["", "Biome coverage · Surface accepted/target | Cave accepted/target"])
        for biome in biomes:
            cells = " | ".join(f"{kind} {selected[biome, kind]}/{planned[biome, kind]}"
                               for kind in ("surface", "cave"))
            lines.append(f"  {biome:<20} {cells}")
        lines.append("  Underwater follows natural water; no fixed biome quota.")
    if state.get("last_error"):
        lines.extend(["", f"Last error: {state['last_error']}"])
    return lines


def show_dashboard(dataset, log_path=None, interval=5, once=False):
    dataset = dataset.resolve()
    manifest = dataset / "manifest.json"
    config = read_json(manifest)["config"]
    v3 = config.get("mode") in ("smoke", "production")
    slots = v3_plan(config) if v3 else plan(config["seed"])
    initialized = manifest.stat().st_mtime
    log = RunLog()
    tty = sys.stdout.isatty()
    try:
        while True:
            job = collector_process(dataset)
            if not v3:
                log.read(select_log(dataset, job, log_path, initialized))
            width, height = shutil.get_terminal_size()
            if v3:
                progress = dataset / "collection_progress.json"
                state = read_json(progress) if progress.is_file() else None
                if state and "initial_completed" in state and "biome_counts" in state:
                    committed, timestamps = {}, []
                else:
                    committed = committed_paths(dataset)
                    timestamps = [(scene / "scene.json").stat().st_mtime for scene in committed.values()]
                lines = v3_progress_lines(dataset, slots, config, set(committed), timestamps, state,
                                          job, initialized, time.time(), width)
            else:
                committed = committed_paths(dataset)
                timestamps = [(scene / "scene.json").stat().st_mtime for scene in committed.values()]
                lines = progress_lines(dataset, slots, config, set(committed), timestamps, job, log,
                                       initialized, time.time(), width)
            if tty and not once:
                if len(lines) >= height:
                    lines = lines[:max(1, height - 2)] + ["Use --once for all biome rows and failure details."]
                sys.stdout.write("\x1b[H" + "\n".join(lines) + "\x1b[J\n")
                sys.stdout.flush()
            else:
                print("\n".join(lines), flush=True)
            if once:
                return 0
            time.sleep(interval)
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
