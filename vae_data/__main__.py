"""Explicit sharded entry, with a process-tree wall watchdog for blocking native IO."""
import json,os,signal,subprocess,sys,time
from pathlib import Path


def main():
    args=sys.argv[1:]
    # Parse before spawning, including --wall-seconds=N spelling.
    from .sharded.cli import parser,main as command
    parsed=parser(args)
    if not hasattr(parsed,'wall_seconds'):return command(args)
    budget=parsed.wall_seconds
    worker=subprocess.Popen([sys.executable,'-m','vae_data.sharded.cli',*args],start_new_session=True)
    try:return worker.wait(timeout=budget)
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as interruption:
        pairs=[tuple(map(int,row.split()))for row in subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines()if row.strip()]
        owned={worker.pid}
        while True:
            more={pid for pid,parent in pairs if parent in owned}-owned
            if not more:break
            owned|=more
        for sig in(signal.SIGTERM,signal.SIGKILL):
            for pid in sorted(owned,reverse=True):
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
            if sig==signal.SIGTERM:time.sleep(.5)
        worker.wait();root=Path(parsed.dataset)
        timed_out=isinstance(interruption,subprocess.TimeoutExpired)
        if root.exists():
            name='wall_timeout.json'if timed_out else'interrupted.json'
            (root/name).write_text(json.dumps(dict(status='INCONCLUSIVE',wall_seconds=budget,reason='Explicit native wall limit reached'if timed_out else'Interrupted; committed samples retained'))+'\n')
        return 124 if timed_out else 130

if __name__=='__main__':raise SystemExit(main())
