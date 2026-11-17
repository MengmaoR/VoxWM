"""Native 100K generator: explicit independent 1K shards; never starts by default."""
import argparse,json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .protocol import CONFIG,dump
from .storage import initialize
from .generate import generate_shard

ROOT=Path(__file__).resolve().parents[2]

def parser(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in('generate','check','status','native-smoke','native-six','clock-verify','static-verify','static-ten'):
        q=sub.add_parser(name);q.add_argument('--dataset',type=Path,required=True)
        if name in('generate','native-smoke','native-six','clock-verify','static-verify','static-ten'):
            q.add_argument('--wall-seconds',type=float,required=True);q.add_argument('--craftium-dir',type=Path,default=ROOT/'third_party/craftium')
            q.add_argument('--renderer',choices=('auto','window','software'),default='auto');q.add_argument('--seed',type=int,default=42)
        if name in('generate','check'):q.add_argument('--shard',type=int,default=0)
        if name=='generate':
            q.add_argument('--count',type=int,default=1);q.add_argument('--workers',type=int,default=1);q.add_argument('--resume',action='store_true')
            q.add_argument('--max-attempts',type=int,default=5000);q.add_argument('--sites-per-world',type=int,default=8)
        if name=='static-ten':q.add_argument('--preview-dataset',type=Path)
        if name=='static-verify':
            q.set_defaults(seed=191664964)
            q.add_argument('--site',type=int,nargs=2,default=[-272,1456],metavar=('E','N'))
        if name=='clock-verify':
            q.add_argument('--ticks',type=int,choices=(24,48,120,240),default=240)
            q.add_argument('--consumer-work-ms',type=float,default=0.)
            q.add_argument('--compare-to',type=Path)
        if name=='check':q.add_argument('--allow-incomplete',action='store_true')
        if name=='native-six':
            q.add_argument('--attempts-per-type',type=int,choices=(1,2,3),default=3)
            q.add_argument('--skip-d2',action='store_true',help='Do not rerun the already accepted D2 example')
        if name=='native-smoke':
            q.add_argument('--samples-per-category',type=int,choices=(1,2),default=2)
            q.add_argument('--attempts-per-category',type=int,default=2)
    q=sub.add_parser('export-traces');q.add_argument('--dataset',type=Path,required=True);q.add_argument('--report-dir',type=Path)
    q=sub.add_parser('view');q.add_argument('--dataset',type=Path,required=True);q.add_argument('--host',default='127.0.0.1');q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('profile-planner');q.add_argument('--real',type=Path,required=True);q.add_argument('--fire',type=Path);q.add_argument('--pair',type=Path);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if hasattr(a,'wall_seconds')and(not 0<a.wall_seconds<float('inf')):p.error('Wall budget must be finite and positive')
    if hasattr(a,'shard')and not 0<=a.shard<100:p.error('--shard must be 0..99')
    if a.command=='generate'and(a.count<1 or a.shard+a.count>100 or a.workers<1 or a.max_attempts<1 or a.sites_per_world<1):p.error('Positive counts within 100 shards required')
    if a.command=='native-smoke'and not 1<=a.attempts_per_category<=4:p.error('Native smoke has a finite 1..4 attempts/category cap')
    if a.command=='native-six'and a.wall_seconds>900:p.error('Six-type native validation has a 900 second total cap')
    if a.command=='static-ten'and a.wall_seconds>600:p.error('Ten-Static collection wall cap <=600s')
    if a.command=='static-verify'and a.wall_seconds>180:p.error('Static verification wall cap <=180s')
    if a.command=='clock-verify'and(not 0<=a.consumer_work_ms<=20 or a.wall_seconds>120):p.error('Clock test: CPU workload 0..20ms, wall cap <=120s')
    return a


def status(root):
    root=Path(root);ready=[];partial={}
    for i in range(100):
        name=f'shard_{i:04d}';final=root/'shards'/name;temp=root/'shards'/('.'+name+'.partial')
        if(final/'READY').exists():ready.append(name)
        else:
            path=final if final.exists()else temp
            if(path/'statistics.json').exists():partial[name]=json.loads((path/'statistics.json').read_text())
    return dict(completed_shards=len(ready),completed_samples=1000*len(ready),target_shards=100,ready=ready,incomplete=partial)


def main(argv=None):
    a=parser(argv)
    if a.command=='view':
        from .viewer.server import serve
        serve(a.dataset,a.host,a.port);return 0
    if a.command=='export-traces':
        from .export_trace import export
        print(export(a.dataset,a.report_dir));return 0
    if a.command=='profile-planner':
        from .planning.profile import run
        result=run(a.output,a.real,a.fire,a.pair)
    elif a.command=='status':result=status(a.dataset)
    elif a.command=='check':
        from .validate import check_shard
        final=a.dataset/'shards'/f'shard_{a.shard:04d}';path=final if final.exists()else final.parent/('.'+final.name+'.partial')
        result=check_shard(a.dataset,path,not a.allow_incomplete)
        print(json.dumps(result,indent=2));return 0 if result['valid']else 1
    elif a.command=='static-ten':
        from .static_ten import run
        result=run(a)
    elif a.command=='static-verify':
        from .static_verify import run
        result=run(a)
    elif a.command=='clock-verify':
        from .clock_verify import run
        result=run(a)
    elif a.command=='native-six':
        from .native_six import run
        result=run(a)
    elif a.command=='native-smoke':
        from .smoke import run
        result=run(a)
    else:
        initialize(a.dataset);deadline=time.monotonic()+max(1,a.wall_seconds-10)
        tasks=[(a.dataset,i,a.resume,deadline,a.craftium_dir,a.renderer,a.seed,a.max_attempts,a.sites_per_world)for i in range(a.shard,a.shard+a.count)]
        if a.workers==1:results=[generate_shard(*task)for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(a.workers,a.count),mp_context=multiprocessing.get_context('spawn'))as executor:
                futures=[executor.submit(generate_shard,*task)for task in tasks];results=[future.result()for future in futures]
        result=dict(shards=results,progress=status(a.dataset));dump(a.dataset/'last_generation.json',result)
    print(json.dumps(result,indent=2));return 1 if result.get('status')=='FAIL'else 0

if __name__=='__main__':raise SystemExit(main())
