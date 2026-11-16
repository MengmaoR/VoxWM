"""Exactly six unique native types, <=3 attempts/type, <=1 success/type, <=900s total."""
from pathlib import Path
import time,traceback
import numpy as np
from .protocol import Vocabulary,VERSION,SIMULATION_HZ,NO_CHANGE_TICKS
from .native import Native
from .generate import collect
from .centers import prepare_dynamic,height_statistics
from .trace import Trace
from .smoke import save_example
from .storage import atomic_json

TYPES=('D1','D2','D3','D4-A','D4-B','D4-C')

def limits(args):
    if not 0<args.wall_seconds<=900:raise ValueError('Six-type total wall limit is 900 seconds')
    if not 1<=args.attempts_per_type<=3:raise ValueError('At most three native attempts/type')

def run(args):
    limits(args);root=Path(args.dataset)
    if root.exists():raise FileExistsError('Use a new output directory; this native budget cannot silently restart')
    root.mkdir(parents=True);Vocabulary().save(root/'vocabulary.json')
    started=time.monotonic();deadline=started+args.wall_seconds-12;rng=np.random.default_rng(args.seed)
    result=dict(version=VERSION,simulation_hz=SIMULATION_HZ,no_change_ticks=NO_CHANGE_TICKS,source_domain='center local[1..14]^3',status='INCONCLUSIVE',source='real_native',unique_types=list(TYPES),successes={t:0 for t in TYPES},attempts=[],
        wall_budget=args.wall_seconds,attempts_per_type=args.attempts_per_type,max_successes_per_type=1,
        formal_shard_generated=False,models_or_training=False)
    sites={'D1':[(80,1984),(96,1936),(112,1952)],'D2':[(800,-1232),(-288,1456),(96,1936)],
        'D3':[(800,-1232),(-288,1456),(816,-1216)],'D4-A':[(80,1984),(96,1936),(112,1952)],
        'D4-B':[(800,-1232),(816,-1216),(-288,1456)],'D4-C':[(800,-1232),(80,1984),(-288,1456)]}
    atomic_json(root/'native_results.json',result)
    for label in TYPES:
        if time.monotonic()+20>=deadline:break
        world=None;first_height=int(rng.integers(2))
        for attempt in range(args.attempts_per_type):
            if result['successes'][label] or time.monotonic()+15>=deadline:break
            category=label.split('-')[0];subtype=label.split('-')[1]if'-'in label else''
            folder=root/'traces'/label/f'attempt_{attempt+1}';trace=Trace(folder,Vocabulary(),label)
            height='surface'if category in('D1','D3')else('surface','cave')[(first_height+attempt)%2]
            row=dict(height_type=height,unique_type=label,category=category,subtype=subtype,attempt=attempt+1,trace=str(folder.relative_to(root)),source='real_native')
            attempt_start=time.monotonic()
            try:
                if world is None:
                    v=Vocabulary();world=Native(191664964,v,deadline,args.craftium_dir,args.renderer);world.__enter__()
                trace.v=world.v
                frame,_=prepare_dynamic(world,sites[label][attempt],category,rng,subtype,height,diagnostic=True)
                chain,details=collect(world,frame,category,rng,subtype,trace=trace);row.update(details)
                if chain:
                    row['output']=save_example(root,label,chain,world.v);result['successes'][label]=1
            except Exception as exc:
                row.update(status='failed',reason=f'{type(exc).__name__}: {exc}',error_class=type(exc).__name__)
                (folder/'exception.txt').write_text(traceback.format_exc())
                if world:trace.recover(world)
                trace.finish(status=row)
                if world:
                    world.__exit__(None,None,None);world=None
            finally:
                if not trace.stream.closed:trace.finish(status=row)
                row['wall_seconds']=time.monotonic()-attempt_start
                result['attempts'].append(row);result['height_statistics']=height_statistics(result['attempts']);result['wall_seconds']=time.monotonic()-started
                atomic_json(root/'native_results.json',result)
                print('NATIVE6',label,attempt+1,row.get('status'),row.get('reason'),f"{row['wall_seconds']:.1f}s",flush=True)
            if time.monotonic()+15>=deadline:break
        if world:world.__exit__(None,None,None)
    result['wall_seconds']=time.monotonic()-started
    result['not_attempted']=[t for t in TYPES if not any(r['unique_type']==t for r in result['attempts'])]
    result['status']='GO'if all(result['successes'].values())else'INCONCLUSIVE'
    result['interpretation']='Native attempts only. Failure is not replaced by synthetic data; formal 1K/100K throughput is unverified.'
    atomic_json(root/'native_results.json',result)
    return result
