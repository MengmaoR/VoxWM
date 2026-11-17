"""Explicit entry with wall, infrastructure and committed-progress watchdogs."""
import json,os,subprocess,sys,time
from pathlib import Path
from voxwm_gt.runtime import stop_tree


def progress_signature(root,first,count):
    root=Path(root);paths=[root/'vocabulary.json']
    for i in range(first,first+count):
        for name in(f'shard_{i:04d}',f'.shard_{i:04d}.partial'):
            paths.append(root/'shards'/name/'progress')
    signature=[]
    for path in paths:
        try:
            stat=path.stat();signature.append((str(path),stat.st_mtime_ns,stat.st_size))
        except FileNotFoundError:pass
    return tuple(signature)


def main():
    args=sys.argv[1:]
    from .sharded.cli import parser,main as command
    parsed=parser(args)
    if not hasattr(parsed,'wall_seconds'):return command(args)
    budget=parsed.wall_seconds;root=Path(parsed.dataset);run_id=f'{os.getpid()}_{time.time_ns()}'
    worker=subprocess.Popen([sys.executable,'-m','vae_data.sharded.cli',*args],start_new_session=True,
        env=dict(os.environ,VOXWM_RUN_ID=run_id))
    started=last_progress=time.monotonic();last_check=started;signature=None
    stall=getattr(parsed,'stall_seconds',0);reason=None;name=None;code=1;status='FAIL'
    fatal=root/'.runtime'/f'infrastructure_{run_id}.json'
    try:
        while True:
            remaining=budget-(time.monotonic()-started)
            if remaining<=0:
                name,code,status,reason='wall_timeout.json',124,'INCONCLUSIVE','Explicit native wall limit reached';break
            try:
                completed=worker.wait(timeout=min(1.,remaining))
                if completed==0 or not fatal.exists():return completed
            except subprocess.TimeoutExpired:pass
            if fatal.exists():
                name='infrastructure_failure.json';reason=fatal.read_text();break
            now=time.monotonic()
            if stall and now-last_check>=min(5.,stall):
                last_check=now;current=progress_signature(root,parsed.shard,parsed.count)
                if signature is None or current!=signature:signature=current;last_progress=now
                elif now-last_progress>=stall:
                    name,code,reason='stalled.json',125,f'No committed samples or new vocabulary tokens for {stall}s';break
    except KeyboardInterrupt:
        name,code,status,reason='interrupted.json',130,'INCONCLUSIVE','Interrupted; committed samples retained'
    # Stop the whole owned tree before writing diagnostics, including independent
    # engine and Xvfb sessions. No attempt is made to delete another run's lock.
    stop_tree(worker.pid);worker.wait()
    print(reason,file=sys.stderr,flush=True)
    if root.exists():(root/name).write_text(json.dumps(dict(status=status,wall_seconds=budget,reason=reason))+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
