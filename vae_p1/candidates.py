"""Reproducible scene plans and stratified selection; no game imports."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

CONFIG=json.loads((Path(__file__).parent/'config.json').read_text())
PLANTS=('bamboo','reeds','vine')

def scene_rng(seed,category,index,subtype=''):
    return np.random.default_rng(np.random.SeedSequence([seed,int(category[1]),index,PLANTS.index(subtype)+1 if subtype else 0]))

def d2_schedule(n,rng):
    names=list(CONFIG['controlled_d2_scene_ratios']);weights=np.array(list(CONFIG['controlled_d2_scene_ratios'].values()))
    counts=np.floor(weights*n).astype(int)
    order=np.argsort(-(weights*n-counts),kind='stable')
    for i in order[:n-int(counts.sum())]:counts[i]+=1
    result=[name for name,count in zip(names,counts) for _ in range(int(count))]
    rng.shuffle(result);return result

def plan(seeds,categories,settings,sites,delay):
    scenes=[]
    for seed in seeds:
        d2=d2_schedule(sites,scene_rng(seed,'D2',0))
        for category in categories:
            for subtype in (PLANTS if category=='D4' else ('',)):
                for index in range(sites):
                    rng=scene_rng(seed,category,index,subtype)
                    x,z=(int(v)//16*16 for v in rng.integers(-2048,2049,2))
                    pair=f'{seed}_{category}_{subtype or "all"}_{index}'
                    for setting in settings:
                        scenes.append(dict(scene_id=pair+'_'+setting,pair_id=pair,world_seed=seed,index=index,
                            category=category,subtype=subtype,setting=setting,site=dict(x=x,z=z,
                            depth=32 if category=='D2' and index%2==0 else 64 if category=='D2' else 0),
                            interaction_context=d2[index] if category=='D2' and setting=='C' else 'natural',
                            delay_seconds=delay if setting=='B' else 0))
    return scenes

def choose(candidates,rng,limit,spacing=0):
    """Random round-robin over context strata, then enforce spatial separation."""
    groups=defaultdict(list)
    for c in sorted(candidates,key=lambda c:tuple(c['position'])):groups[c['stratum_key']].append(c)
    keys=list(sorted(groups));rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    result=[]
    while keys and len(result)<limit:
        remaining=[]
        for key in keys:
            group=groups[key]
            while group:
                c=group.pop()
                if all(np.linalg.norm(np.array(c['position'])-np.array(p['position']))>=spacing for p in result):
                    result.append(c);break
            if group:remaining.append(key)
            if len(result)==limit:break
        keys=remaining
    return result

def fire_draw(candidates,rng):
    if len(candidates)!=20:raise ValueError('Controlled fire needs exactly 20 physically eligible dispersed candidates')
    draws=rng.random(20).tolist();selected=[i for i,p in enumerate(draws) if p<0.1]
    fallback=None
    if not selected:fallback=int(rng.integers(20));selected=[fallback]
    return dict(probability=0.1,draws=draws,selected_indices=selected,empty_draw_fallback=fallback)

def next_plant_window(current,can_grow):
    return next((t for t in CONFIG['plant_windows'] if t>current),None) if can_grow else None

def budget(scenes):
    initial=sum(CONFIG['observation_seconds'][s['category']]+s['delay_seconds'] for s in scenes)
    maximum=sum((1800 if s['category']=='D4' else CONFIG['observation_seconds'][s['category']])+s['delay_seconds'] for s in scenes)
    return dict(scene_instances=len(scenes),nominal_simulation_seconds=initial,maximum_simulation_seconds=maximum,
        wall_seconds_estimate=None,note='Simulation time only; early stopping, setup/discovery and measured simulation/wall throughput change actual cost.')
