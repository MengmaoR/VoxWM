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
    for name in('generate','check','status','native-smoke'):
        q=sub.add_parser(name);q.add_argument('--dataset',type=Path,required=True)
        if name in('generate','native-smoke'):
            q.add_argument('--wall-seconds',type=float,required=True);q.add_argument('--craftium-dir',type=Path,default=ROOT/'third_party/craftium')
            q.add_argument('--renderer',choices=('auto','window','software'),default='auto');q.add_argument('--seed',type=int,default=42)
        if name in('generate','check'):q.add_argument('--shard',type=int,default=0)
        if name=='generate':
            q.add_argument('--count',type=int,default=1);q.add_argument('--workers',type=int,default=1);q.add_argument('--resume',action='store_true')
            q.add_argument('--max-attempts',type=int,default=5000);q.add_argument('--sites-per-world',type=int,default=8)
        if name=='check':q.add_argument('--allow-incomplete',action='store_true')
        if name=='native-smoke':
            q.add_argument('--samples-per-category',type=int,choices=(1,2),default=2)
            q.add_argument('--attempts-per-category',type=int,default=2)
    a=p.parse_args(argv)
    if hasattr(a,'wall_seconds')and(not 0<a.wall_seconds<float('inf')):p.error('Wall budget must be finite and positive')
    if hasattr(a,'shard')and not 0<=a.shard<100:p.error('--shard must be 0..99')
    if a.command=='generate'and(a.count<1 or a.shard+a.count>100 or a.workers<1 or a.max_attempts<1 or a.sites_per_world<1):p.error('Positive counts within 100 shards required')
    if a.command=='native-smoke'and not 1<=a.attempts_per_category<=4:p.error('Native smoke has a finite 1..4 attempts/category cap')
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
    if a.command=='status':result=status(a.dataset)
    elif a.command=='check':
        from .validate import check_shard
        final=a.dataset/'shards'/f'shard_{a.shard:04d}';path=final if final.exists()else final.parent/('.'+final.name+'.partial')
        result=check_shard(a.dataset,path,not a.allow_incomplete)
        print(json.dumps(result,indent=2));return 0 if result['valid']else 1
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
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
