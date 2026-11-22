"""Read-only terminal progress for committed scenes and active native workers."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import shutil
import sys
import time

from vae_data.supervisor import _bar, _duration

from .dashboard_processes import collector_process
from .native import scene_id_for_world_seed
from .protocol import HEIGHTS, plan
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


def show_dashboard(dataset, log_path=None, interval=5, once=False):
    dataset = dataset.resolve()
    manifest = dataset / "manifest.json"
    config = read_json(manifest)["config"]
    slots = plan(config["seed"])
    initialized = manifest.stat().st_mtime
    log = RunLog()
    tty = sys.stdout.isatty()
    try:
        while True:
            job = collector_process(dataset)
            log.read(select_log(dataset, job, log_path, initialized))
            committed = committed_paths(dataset)
            timestamps = [(scene / "scene.json").stat().st_mtime for scene in committed.values()]
            width, height = shutil.get_terminal_size()
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
