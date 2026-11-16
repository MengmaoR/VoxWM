"""Strictly bounded native examples, never an official shard or a large map scan."""
from pathlib import Path
import json,time
import numpy as np
import zarr
from numcodecs import Blosc
from .protocol import Vocabulary,Chain,dump
from .native import Native
from .generate import collect
from .centers import prepare_dynamic
from .storage import globalize,atomic_json
from .validate import validate_chain


def save_example(root,label,chain,vocabulary):
    frames=chain.frames+([chain.anchor]if chain.anchor else[]);mapped,_=globalize(root,frames,vocabulary)
    saved=Chain(chain.category,mapped[:5],mapped[-1]if chain.anchor else None,chain.events,chain.intervention,chain.subtype)
    validate_chain(saved,Vocabulary(Path(root)/'vocabulary.json'))
    folder=Path(root)/'examples'/label;folder.mkdir(parents=True)
    for pool,states in(('snapshots',saved.frames),('anchors',[saved.anchor]if saved.anchor else[])):
        g=zarr.open_group(str(folder/(pool+'.zarr')),mode='w')
        for name,dtype in(('tokens','<u2'),('light','u1')):
            array=np.stack([getattr(f,name)for f in states])if states else np.empty((0,48,48,48),dtype)
            g.create_dataset(name,data=array,chunks=(1,48,48,48),compressor=Blosc(cname='zstd',clevel=3))
    dump(folder/'metadata.json',dict(category=chain.category,subtype=chain.subtype,frames=[f.meta for f in saved.frames],
        anchor=saved.anchor.meta if saved.anchor else None,events=chain.events,intervention=chain.intervention,
        source='real_native',formal_shard=False))
    return str(folder.relative_to(root))


def run(args):
    root=Path(args.dataset)
    if root.exists():raise FileExistsError('Native example output must be new; success budgets cannot silently restart')
    root.mkdir(parents=True);Vocabulary().save(root/'vocabulary.json')
    deadline=time.monotonic()+max(1,args.wall_seconds-12);rng=np.random.default_rng(args.seed);start=time.monotonic()
    result=dict(status='INCONCLUSIVE',source='real_native',attempts=[],successes={k:0 for k in('D1','D2','D3','D4')},
        wall_budget=args.wall_seconds,max_successes_per_category=args.samples_per_category,max_attempts_per_category=args.attempts_per_category,
        full_shard_generated=False,full_100k_generated=False,gpu_training=False)
    sites={'D1':[(80,1984),(96,1936)],'D2':[(96,1936),(80,1984)],'D3':[(-288,1456),(800,-1232)],'D4':[(80,1984),(800,-1232)]}
    atomic_json(root/'native_results.json',result)
    for category in result['successes']:
        if time.monotonic()+25>=deadline:break
        world=None
        try:
            v=Vocabulary();world=Native(191664964,v,deadline,args.craftium_dir,args.renderer);world.__enter__()
            for attempt in range(args.attempts_per_category):
                if result['successes'][category]>=args.samples_per_category or time.monotonic()+15>=deadline:break
                row=dict(category=category,attempt=attempt,subtype=('A','B','C')[attempt%3]if category=='D4'else'',source='real_native')
                try:
                    frame,_=prepare_dynamic(world,sites[category][attempt%2],category,rng,row['subtype'])
                    chain,details=collect(world,frame,category,rng,row['subtype']);row.update(details)
                    if chain:
                        row['output']=save_example(root,category+'_'+str(result['successes'][category]),chain,v)
                        result['successes'][category]+=1
                except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                result['attempts'].append(row);atomic_json(root/'native_results.json',result)
                print('NATIVE',category,attempt,row.get('status'),row.get('reason',''),flush=True)
        except Exception as exc:result['attempts'].append(dict(category=category,status='startup_failed',reason=f'{type(exc).__name__}: {exc}'))
        finally:
            if world:world.__exit__(None,None,None)
            atomic_json(root/'native_results.json',result)
    result['wall_seconds']=time.monotonic()-start
    result['status']='GO'if all(n>0 for n in result['successes'].values())else'INCONCLUSIVE'
    result['interpretation']='Bounded native wiring/examples only; full shard/100K throughput and biome quotas have not been run.'
    atomic_json(root/'native_results.json',result)
    return result
