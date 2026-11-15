"""Shared protocol, immutable token representation, traceable hashes and sampling decisions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from vae_p1.tracking import Vocabulary,Event,Reactions,FireMerges
from vae_p0.sampling import State
from vae_pilot.sampling import biome_group,split_for_seed

HERE=Path(__file__).parent
CONFIG=json.loads((HERE/'config.json').read_text())
BIOMES=json.loads((HERE.parent/'vae_pilot/config.json').read_text())['biome_groups']

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def state_hash(state,vocabulary):
    names,param=vocabulary.decode(state.tokens)
    data=''.join(f'{n}\0{int(p)}\n' for n,p in zip(names.ravel(),param.ravel())).encode()
    return hashlib.sha256(data).hexdigest()
def region_of(prepared):
    return dict(low=prepared['bounds_enu'][0],high=prepared['bounds_enu'][1],surface_y=prepared['surface_y'])
def sparse(candidates,rng,spacing,maximum=None):
    order=rng.permutation(len(candidates));chosen=[]
    for i in order:
        c=candidates[int(i)];p=np.asarray(c['position'])
        if all(np.linalg.norm(p-np.asarray(x['position']))>=spacing for x in chosen):chosen.append(c)
        if maximum and len(chosen)>=maximum:break
    return chosen

def source_actions(inspect,category,rng,config=CONFIG,mode=None,near_water=False):
    n=int(rng.integers(config['source_count_range'][0],config['source_count_range'][1]+1));actions=[];draws=[]
    for _ in range(n):
        attached=bool(rng.random()<config['attached_weight']) if mode is None else mode=='attached'
        pool=inspect['attached' if attached else 'suspended']
        options=[c for c in pool if all(np.linalg.norm(np.asarray(c['position'])-a['position'])>=config['source_spacing'] for a in actions)]
        if near_water:options=[c for c in options if c.get('water_distance') is not None]
        draws.append(dict(attached=attached,eligible=len(options)))
        if not options:continue
        c=options[int(rng.integers(len(options)))];actions.append(dict(type='water' if category=='D1' else 'lava',position=c['position'],attached=attached,
            water_distance=c.get('water_distance'),water_position=c.get('water_position')))
    return actions,dict(requested_sources=n,position_draws=draws,actual_sources=len(actions),near_water_requested=near_water)

def fire_actions(inspect,rng,config=CONFIG):
    pool=sparse(inspect['fire'],rng,config['fire_spacing'],config['fire_candidate_budget'])
    draws=rng.random(len(pool)).tolist();indices=[i for i,x in enumerate(draws) if x<config['fire_probability']]
    fallback=None
    if not indices and pool:fallback=int(rng.integers(len(pool)));indices=[fallback]
    actions=[dict(type='fire',position=pool[i]['position'],under=pool[i]['under']) for i in indices]
    return actions,dict(raw_legal_count=inspect['fire_legal_count'],sparse_candidates=pool,probability=config['fire_probability'],
        draws=draws,selected_indices=indices,empty_fallback=fallback)

def trim_actions(inspect,subtype,rng):
    actions=[];seen=set()
    for plant in inspect['plants'][subtype]:
        if plant['id'] in seen or not plant['complete']:continue
        seen.add(plant['id'])
        if subtype!='vine':
            h=plant['height'];retain=int(rng.integers(1,h+1))
            cuts=[] if retain==h else [plant['members'][retain]]
            actions.append(dict(type='trim',plant=plant,retain_height=retain,cut_positions=cuts))
        else:
            anchors={tuple(p) for p in plant['anchors']}
            options=[p for p in plant['members'] if tuple(p) not in anchors]
            cuts=[options[int(rng.integers(len(options)))]] if options and rng.random()<0.8 else []
            actions.append(dict(type='trim',plant=plant,cut_positions=cuts))
    return actions,dict(natural_plants=len(inspect['plants'][subtype]),unique_complete_plants=len(actions),subtype=subtype)
