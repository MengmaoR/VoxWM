"""Bounded parallel v3 collection with one parent writer and resumable seed cursors."""

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from vae_data.supervisor import owned_pids
from voxwm_gt.native import renderer_config
from voxwm_gt.runtime import stop_tree

from .geometry import build_library
from .protocol import RejectedScene, V3_FINAL_PROFILE, V3_KINDS
from .storage import DatasetWriter, read_json, write_json
from .v3_native import capture_v3
from .v3_sampling import SURVEY_Y


MARKER = "VOX2PIX_COLLECTION_RUN"


def worker_count(value):
    if value == "auto":
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            return os.cpu_count() or 1
    value = int(value)
    if value < 1:
        raise ValueError("workers must be auto or a positive integer")
    return value


def production_config(args):
    if args.lighting_profile != V3_FINAL_PROFILE:
        raise ValueError("Production uses only the final RGB32 / absolute Fog20–30 profile")
    if args.scenes < 10 or args.scenes % 10:
        raise ValueError("Production scene count must be a positive multiple of ten")
    if tuple(args.scene_kinds) != V3_KINDS:
        raise ValueError("Production preserves the surface/cave/underwater plan")
    bounds = list(args.survey_y or SURVEY_Y)
    if not -30900 <= bounds[0] < bounds[1] <= 30900 or not 48 <= bounds[1] - bounds[0] + 1 <= 512:
        raise ValueError("Survey must span 48–512 cells within the playable Lua map")
    return dict(mode="production", scenes=args.scenes, scene_kinds=list(V3_KINDS),
                views_per_scene=5, width=640, height=360, seed=args.seed,
                renderer=renderer_config(args.renderer)[0], survey_y=bounds,
                lighting_profile=V3_FINAL_PROFILE, collection_plan_version="balanced-v3-1",
                split_percent=[80, 10, 10])


class Progress:
    """Small current-state file; seed reservations survive interrupted tasks."""

    def __init__(self, writer, workers):
        self.path = writer.root / "collection_progress.json"
        prior = read_json(self.path) if self.path.exists() else {}
        self.next_attempt = {key: value for key, value in prior.get("next_attempt", {}).items()
                             if int(key) not in writer.completed}
        self.writer = writer
        now = time.time()
        source_root = Path(__file__).resolve().parent.parent
        source = dict(git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
            working_tree_modified=bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_root, text=True).strip()))
        self.state = dict(status="RUNNING", started_at=now, updated_at=now,
                          collector_source=source,
                          pid=os.getpid(), workers=workers, scene_count=len(writer.slots),
                          completed=len(writer.completed), initial_completed=len(writer.completed),
                          frames=len(writer.completed) * 5, active=[], attempts=0, failed=0,
                          exhausted_count=0, last_error=None, commit_times=[], last_commit_at=None,
                          counts=dict.fromkeys(V3_KINDS, 0), biome_counts={})
        for scene_id in writer.completed:
            self._count(writer.slots[scene_id])
        self.save()

    def _count(self, slot):
        self.state["counts"][slot["kind"]] += 1
        if slot.get("biome"):
            counts = self.state["biome_counts"].setdefault(slot["biome"], dict(surface=0, cave=0))
            counts[slot["kind"]] += 1

    def reserve(self, slot, worlds, minimum=0):
        key = str(slot["scene_id"])
        first = max(minimum, self.next_attempt.get(key, 0))
        self.next_attempt[key] = first + worlds
        self.state["attempts"] += 1
        # Save before launching: a crash skips the reserved seeds, never replays them.
        self.save()
        return first

    def committed(self, slot):
        now = time.time()
        self.next_attempt.pop(str(slot["scene_id"]), None)
        self._count(slot)
        self.state["completed"] = len(self.writer.completed)
        self.state["frames"] = self.state["completed"] * 5
        self.state["last_commit_at"] = now
        self.state["commit_times"] = (self.state["commit_times"] + [now])[-1000:]

    def save(self, **fields):
        self.state.update(fields, updated_at=time.time(), next_attempt=self.next_attempt)
        write_json(self.path, self.state)


def _leftover_native(run_id):
    """Clean only this run's native or display children, including adopted ones."""
    pids = owned_pids(run_id, marker=MARKER) - {1, os.getpid()}
    if not pids:
        return
    rows = subprocess.run(["ps", "-p", ",".join(map(str, sorted(pids))), "-o", "pid=,comm="],
                          capture_output=True, text=True, check=False).stdout.splitlines()
    native = {int(row.split(None, 1)[0]) for row in rows
              if len(row.split(None, 1)) == 2
              and Path(row.split(None, 1)[1]).name in ("luanti", "minetest", "Xvfb")}
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid in native:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        if native and signum == signal.SIGTERM:
            time.sleep(.2)


@contextmanager
def collection_pool(workers):
    run_id, previous = uuid4().hex, os.environ.get(MARKER)
    os.environ[MARKER] = run_id
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        yield pool
    except BaseException:
        # Capture worker descendants before killing the workers; native clients
        # use separate Unix sessions and cannot be cleaned by killpg(parent).
        for process in list((pool._processes or {}).values()):
            if process.is_alive():
                stop_tree(process.pid, grace=.2)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        _leftover_native(run_id)
        if previous is None:
            os.environ.pop(MARKER, None)
        else:
            os.environ[MARKER] = previous


def run_tasks(writer, config, args, library, pool, progress, stop_requested, capture_fn=capture_v3):
    wanted = set(args.scene_ids) if args.scene_ids is not None else set(writer.slots)
    if not wanted <= writer.slots.keys():
        raise ValueError("Requested scene IDs are not in this dataset plan")
    selected = [slot for scene_id, slot in writer.slots.items()
                if scene_id in wanted and scene_id not in writer.completed]
    if args.max_scenes is not None:
        selected = selected[:args.max_scenes]
    queue = deque(selected)
    retries = Counter()
    pending = {}

    def publish():
        progress.save(active=[record for _, record in pending.values()])

    while queue or pending:
        if stop_requested():
            raise KeyboardInterrupt
        for _, record in pending.values():
            if time.time() - record["started_at"] > args.scene_timeout + 60:
                raise TimeoutError(f"Scene {record['scene_id']} exceeded its search/startup deadline")
        while queue and len(pending) < progress.state["workers"]:
            slot = queue.popleft()
            first = progress.reserve(slot, args.worlds_per_task, args.search_round * 8)
            future = pool.submit(capture_fn, slot, config, args.craftium_dir.resolve(), library,
                                 args.worlds_per_task, args.sites_per_world, args.survey_columns,
                                 args.scene_timeout, first)
            record = dict(slot, started_at=time.time(), first_attempt=first, world_budget=args.worlds_per_task)
            pending[future] = slot, record
            publish()
        ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
        for future in ready:
            slot, _ = pending.pop(future)
            try:
                result = future.result()
                result[0]["collector_source"] = progress.state["collector_source"]
                writer.commit(*result)
            except RejectedScene as exc:
                progress.state["failed"] += 1
                progress.state["last_error"] = str(exc)
                retries[slot["scene_id"]] += 1
                if retries[slot["scene_id"]] <= args.task_retries:
                    queue.append(slot)
                else:
                    progress.state["exhausted_count"] += 1
                print(f"V3 production scene {slot['scene_id']} {slot['kind']} rejected: {exc}", flush=True)
            else:
                progress.committed(slot)
                print(f"V3 production committed {slot['scene_id']} {slot['kind']} "
                      f"({len(writer.completed)}/{len(writer.slots)})", flush=True)
            publish()
        if not ready and time.time() - progress.state["updated_at"] >= 5:
            publish()
    incomplete = any(slot["scene_id"] not in writer.completed for slot in selected)
    status = "INCOMPLETE" if incomplete else "COMPLETE" if len(writer.completed) == len(writer.slots) else "LIMIT_REACHED"
    progress.save(status=status, active=[])
    return int(incomplete)


def collect_production(args):
    config = production_config(args)
    requested_stop = False

    def interrupt(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    if hasattr(signal, "SIGHUP"):
        handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, interrupt)
    try:
        with DatasetWriter(args.dataset, config) as writer:
            workers = min(worker_count(args.workers), max(1, len(writer.slots) - len(writer.completed)))
            if args.max_scenes is not None:
                workers = min(workers, args.max_scenes)
            progress = Progress(writer, workers)
            if len(writer.completed) == len(writer.slots):
                progress.save(status="COMPLETE")
                return 0
            try:
                with TemporaryDirectory(prefix="vox2pix-production-projector-") as temporary:
                    library = build_library(Path(temporary))
                    with collection_pool(workers) as pool:
                        return run_tasks(writer, config, args, library, pool, progress, lambda: requested_stop)
            except KeyboardInterrupt:
                progress.save(status="INTERRUPTED", active=[])
                return 130
            except Exception as exc:
                progress.save(status="ERROR", active=[], last_error=f"{type(exc).__name__}: {exc}")
                raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
