"""External Unix wall-clock watchdog also covers blocking native startup / socket reads."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args=sys.argv[1:]
def option(name,default=None):
    return args[args.index(name)+1] if name in args else default
budget=option('--wall-seconds')
if not budget or args and args[0]=='status' or '--help' in args or '-h' in args:
    from .cli import main
    main()
else:
    worker=subprocess.Popen([sys.executable,'-m','vae_data.cli',*args],start_new_session=True)
    try:raise SystemExit(worker.wait(timeout=float(budget)))
    except subprocess.TimeoutExpired:
        rows=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True)
        pairs=[tuple(map(int,row.split())) for row in rows.splitlines() if row.strip()]
        owned={worker.pid}
        while True:
            more={pid for pid,parent in pairs if parent in owned}-owned
            if not more:break
            owned|=more
        for pid in reversed(sorted(owned)):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(1)
        for pid in owned:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        worker.wait()
        root=Path(option('--dataset','.'))
        if root.exists():(root/'wall_timeout.json').write_text(json.dumps(dict(status='INCONCLUSIVE',hard_wall_seconds=float(budget),reason='Native worker exceeded wall budget'))+'\n')
        print('INCONCLUSIVE: hard native wall-time limit reached',file=sys.stderr);raise SystemExit(124)
