"""Read Linux process state for an existing Vox2Pix collector."""
import json
import os
from pathlib import Path
import time


def _args(entry):
    try:
        return os.fsdecode((entry / "cmdline").read_bytes()).rstrip("\0").split("\0")
    except OSError:
        return []


def _option(args, name, default=None):
    for index in range(len(args) - 1, -1, -1):
        if args[index].startswith(name + "="):
            return args[index][len(name) + 1:]
        if args[index] == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _started(entry, proc):
    try:
        # The first stat field after the closing parenthesis is field 3.
        fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        uptime = float((proc / "uptime").read_text().split()[0])
        return time.time() - uptime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def _descendants(entry, proc):
    pending = [entry]
    seen = {entry}
    while pending:
        parent = pending.pop()
        for thread in (parent / "task").glob("*/children"):
            try:
                children = thread.read_text().split()
            except OSError:
                continue
            for child in children:
                path = proc / child
                if child.isdecimal() and path not in seen:
                    seen.add(path)
                    pending.append(path)
                    yield path


def _world(entry):
    world = {"pid": int(entry.name), "kind": None, "world_seed": None}
    try:
        cwd = (entry / "cwd").resolve(strict=True)
    except (OSError, RuntimeError):
        return world
    command = cwd / "worlds/world/vox2pix_command.json"
    try:
        kind = json.loads(command.read_text()).get("kind")
        if kind in ("search", "search_v3", "survey", "anchor", "pose", "snapshot"):
            world["kind"] = kind
    except (OSError, ValueError, AttributeError):
        pass
    try:
        for line in (cwd / "minetest.conf").read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "fixed_map_seed":
                world["world_seed"] = int(value)
    except (OSError, ValueError):
        pass
    return world


def collector_process(dataset, proc=Path("/proc")):
    """Return a lightweight snapshot, or None without a matching live process."""
    dataset, proc = Path(dataset).resolve(), Path(proc)
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        argv = _args(entry)
        if "-m" not in argv or not Path(argv[0]).name.startswith(("python", "pypy")):
            continue
        module = argv.index("-m")
        if argv[module:module + 3] != ["-m", "vox2pix_data", "collect"]:
            continue
        options = argv[module + 3:]
        path = _option(options, "--dataset")
        if path is None:
            continue
        try:
            cwd = (entry / "cwd").resolve(strict=True)
            if (cwd / path).resolve() != dataset:
                continue
            worker_option = _option(options, "--workers", "1")
            if worker_option == "auto":
                workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            else:
                workers = int(worker_option)
        except (OSError, RuntimeError, ValueError):
            continue

        live_workers, worlds = 0, []
        for child in _descendants(entry, proc):
            args = _args(child)
            if not args:
                continue
            if Path(args[0]).name in ("luanti", "minetest"):
                worlds.append(_world(child))
            elif any(arg.startswith("from multiprocessing.spawn import spawn_main") for arg in args):
                live_workers += 1

        log = None
        try:
            output = (entry / "fd/1").resolve(strict=True)
            if output.is_file():
                log = output
        except (OSError, RuntimeError):
            pass
        return dict(pid=int(entry.name), workers=workers, live_workers=live_workers,
                    worlds=sorted(worlds, key=lambda world: world["pid"]),
                    started=_started(entry, proc), log=log)
    return None
