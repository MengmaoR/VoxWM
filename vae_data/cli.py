"""Unified native discovery/replay/collection. No command starts 200K by default."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .core import CONFIG,dump
from .storage import Dataset
from .pipeline import discover,reconstruct,collection,schedule

ROOT=Path(__file__).resolve().parents[1]

def parser(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('discover','replay','collect','validate','status'))
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--wall-seconds',type=float);p.add_argument('--craftium-dir',type=Path,default=ROOT/'third_party/craftium')
    p.add_argument('--renderer',choices=('auto','window','software'),default='auto');p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--world-count',type=int);p.add_argument('--sites-per-world',type=int,default=8);p.add_argument('--scene-id');p.add_argument('--category',choices=list(CONFIG['quotas']),default='Ordinary')
    p.add_argument('--subtype',choices=('bamboo','reeds','vine'),default='bamboo');p.add_argument('--seconds',type=float)
    p.add_argument('--intervention',choices=('auto','always','never'),default='auto');p.add_argument('--source-mode',choices=('attached','suspended'))
    p.add_argument('--rollout-tag',default='',help='Explicitly request a new independent rollout, not native time restoration');p.add_argument('--repeat',type=int,default=3);p.add_argument('--full-200k',action='store_true');p.add_argument('--max-scenes',type=int,default=1)
    a=p.parse_args(argv)
    if a.command!='status' and(not a.wall_seconds or a.wall_seconds<=0):p.error('Native commands require --wall-seconds')
    if a.repeat<1 or a.max_scenes<1 or a.sites_per_world<1 or a.world_count is not None and a.world_count<1:p.error('Counts must be positive')
    if a.seconds is not None and a.seconds<0:p.error('--seconds must be non-negative')
    return a

def main(argv=None):
    a=parser(argv);deadline=time.monotonic()+max(1,(a.wall_seconds or 0)-15)
    dataset=Dataset(a.dataset,a.resume or a.command in ('replay','collect','validate','status'))
    try:
        if a.command=='status':print(json.dumps(dict(counts=dataset.counts(),indexed_scenes=len(dataset.scenes())),indent=2));return
        if a.command=='discover':
            rng=np.random.default_rng(42);sites=[dict(x=int(x)//16*16,z=int(z)//16*16,depth=(0,16,64,128)[i%4]) for i,(x,z) in enumerate(rng.integers(-2048,2049,(a.sites_per_world,2)))]
            seeds=a.seeds if a.world_count is None else [int(s) for s in rng.choice(2147483646,size=a.world_count,replace=False)+1]
            discover(dataset,seeds,sites,deadline,a.craftium_dir,a.renderer)
        elif a.command=='validate':
            from .validation import validate
            validate(dataset,deadline,a.craftium_dir,a.renderer)
        elif a.command=='replay':
            scene=next((s for s in dataset.scenes() if s['scene_id']==a.scene_id),None)
            if not scene:raise ValueError('Choose --scene-id from scene_index.parquet')
            result=reconstruct(dataset,scene,a.repeat,a.seconds or 4,deadline,a.craftium_dir,a.renderer);print(json.dumps({k:v for k,v in result.items() if k!='rollouts'},indent=2))
        else:
            categories=list(CONFIG['quotas']) if a.full_200k else[a.category];n=0
            while time.monotonic()<deadline:
                progress=False
                for category in categories:
                    if dataset.counts().get(category,0)>=CONFIG['quotas'][category]:continue
                    for scene in schedule(dataset,category,n):
                        if a.scene_id and scene['scene_id']!=a.scene_id:continue
                        subtype=a.subtype
                        if a.full_200k and category=='D4':
                            choices=('bamboo','reeds','vine');ordered=choices[n%3:]+choices[:n%3]
                            subtype=next((kind for kind in ordered if scene['signals'].get(kind,0) and not dataset.completed(scene.get('region_id',scene['scene_id'])+'_D4_'+kind)),subtype)
                        if category=='D4' and not scene['signals'].get(subtype,0):continue
                        key=scene.get('region_id',scene['scene_id'])+'_'+category+'_'+(subtype if category=='D4'else'')
                        if a.rollout_tag:key+='_'+a.rollout_tag
                        if dataset.completed(key):continue
                        r=collection(dataset,scene,category,subtype if category=='D4'else'',a.seconds if a.seconds is not None else CONFIG['observation_seconds'].get(category,0),
                            deadline,a.craftium_dir,a.renderer,a.intervention,a.source_mode,key)
                        print(category,scene['scene_id'],r['status'],r.get('standard_snapshots',0),flush=True);n+=1;progress=True;break
                    if not a.full_200k and n>=a.max_scenes:return
                if not progress:break
            missing={k:max(0,q-dataset.counts().get(k,0)) for k,q in CONFIG['quotas'].items()}
            dump(dataset.root/'quota_shortfall.json',missing);print('Remaining:',missing)
    finally:
        dump(dataset.root/'collection_statistics.json',dataset.summary())
        dataset.close()

if __name__=='__main__':main()
