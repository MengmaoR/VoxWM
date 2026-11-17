"""Owned-process cleanup and explicit collection infrastructure failures."""
import os,signal,subprocess,time


class InfrastructureError(RuntimeError):pass


def stop_process(proc,grace=2):
    if proc is None:return
    # Engine/Xvfb are started with start_new_session=True. Their descendants
    # belong to this group even when a launcher exits before the actual engine.
    for sig in(signal.SIGTERM,signal.SIGKILL):
        try:os.killpg(proc.pid,sig)
        except ProcessLookupError:pass
        if proc.poll()is None:
            try:proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:continue
        if sig==signal.SIGTERM:
            # Also remove a surviving descendant after an already-exited wrapper.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        break


def stop_tree(pid,grace=.5):
    """Signal only this command's process descendants, including new sessions."""
    pairs=[tuple(map(int,row.split()))for row in subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines()if row.strip()]
    owned={pid}
    while True:
        more={child for child,parent in pairs if parent in owned}-owned
        if not more:break
        owned|=more
    for sig in(signal.SIGTERM,signal.SIGKILL):
        for child in sorted(owned,reverse=True):
            try:os.kill(child,sig)
            except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(grace)
