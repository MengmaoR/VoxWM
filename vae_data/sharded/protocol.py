"""Frozen 1K-shard recipe and lossless world-aligned region representation."""
from dataclasses import dataclass,field
from pathlib import Path
import json
import numpy as np
from vae_p0.codec import Vocabulary
from vae_pilot.sampling import biome_group,split_for_seed

VERSION='vae-region48-fixed24-v2'
SIMULATION_HZ=24
FIXED_DT=1/SIMULATION_HZ
NO_CHANGE_TICKS={'D1':48,'D2':48,'D3':120,'D4':48}
BIOMES=json.loads((Path(__file__).resolve().parents[2]/'vae_pilot/config.json').read_text())['biome_groups']
CATEGORIES={'Static':500,'D1':125,'D2':125,'D3':125,'D4':125}
STRATA={'surface':8,'elevated_complex':4,'shallow_underground':4,'deep_cave':4}
CENTER=(slice(16,32),)*3
INTERIOR=(slice(17,31),)*3
PRODUCTS={'mcl_core:stone','mcl_core:cobble','mcl_core:obsidian'}
CONFIG=dict(version=VERSION,shards=100,shard_size=1000,region_size=48,chunk_size=16,axes='ENU',
    categories=CATEGORIES,static_per_biome=STRATA,biome_transition_soft_target=4,chain_states=5,
    candidate_limit=10,simulation_hz=SIMULATION_HZ,fixed_dt=FIXED_DT,no_change_ticks=NO_CHANGE_TICKS,liquid_update=1.0,liquid_ranges={'water':7,'lava':3},
    capture_schedule="every completed native tick",intervention_local_bounds=[1,14],max_observation_seconds={'D1':30.,'D2':40.,'D3':40.,'D4':60.},
    split_percent=[80,10,10],max_samples_per_seed_per_shard=100,biome_groups=BIOMES,
    initialization=dict(source_count_max=4,sampling='all qualified points: independent p=1/N, uniform zero-hit fallback'))

def check_sources(actions,center):
    """Initialization only: propagation is allowed throughout the full region."""
    origin=np.asarray(center)
    for action in actions:
        position=np.asarray(action.get('position',[]))
        if position.shape!=(3,)or position.dtype.kind not in'iuf' or not np.all(np.isfinite(position)&(position==np.floor(position))&(position-origin>=1)&(position-origin<=14)):
            raise ValueError('intervention_outside_center_14: '+str(action.get('position')))
    return True


def d4_targets(shard):
    result={'A':8,'B':8,'C':8};result[('C','A','B')[shard%3]]+=1
    return result

def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

@dataclass
class Frame:
    tokens:np.ndarray
    light:np.ndarray
    meta:dict
    def __post_init__(self):
        if self.tokens.shape!=(48,48,48) or self.tokens.dtype!=np.uint16:raise ValueError('Expected uint16[48,48,48] tokens')
        if self.light.shape!=self.tokens.shape or self.light.dtype!=np.uint8:raise ValueError('Expected uint8[48,48,48] light')
        raw_center=np.asarray(self.meta['center_origin'])
        if raw_center.dtype.kind not in 'iu':raise ValueError('Chunk origin must contain integer node coordinates')
        center=raw_center.astype(int)
        if center.shape!=(3,) or np.any(center%16):raise ValueError('Center must be world-aligned in ENU')
        if self.meta.get('region_origin')!=list((center-16).tolist()):raise ValueError('Region is not the 3³ neighborhood of its center')
    @property
    def center(self):return self.tokens[CENTER]
    @property
    def key(self):return f"{self.meta['rollout_id']}:{self.meta['simulation_step']}"
    @property
    def time(self):return float(self.meta['sim_time'])
    @property
    def step(self):return int(self.meta['simulation_step'])

@dataclass
class Chain:
    category:str
    frames:list
    anchor:Frame|None=None
    events:list=field(default_factory=list)
    intervention:dict=field(default_factory=dict)
    subtype:str=''


def assemble(chunks,center_origin,meta):
    low=np.asarray(center_origin,dtype=int)-16;tokens=np.empty((48,)*3,np.uint16);light=np.empty((48,)*3,np.uint8)
    expected={tuple((low+np.array([x,y,z])*16).tolist()) for x in range(3)for y in range(3)for z in range(3)}
    if set(chunks)!=expected:raise ValueError('Incomplete or moved 27-chunk region')
    for origin,state in chunks.items():
        offset=np.asarray(origin)-low;sl=tuple(slice(int(i),int(i+16))for i in offset)
        tokens[sl]=state.tokens;light[sl]=state.light
    return Frame(tokens,light,dict(meta,center_origin=list(map(int,center_origin)),region_origin=low.tolist()))


def kind_mask(tokens,vocabulary,kind):
    names=np.asarray([name for name,_ in vocabulary.entries],object)
    if kind=='water':table=np.array(['water' in n and ('source'in n or 'flowing'in n)for n in names])
    elif kind=='lava':table=np.array(['lava' in n and ('source'in n or 'flowing'in n)for n in names])
    elif kind=='fire':table=np.array([n.startswith('mcl_fire:') and 'fire'in n for n in names])
    elif kind=='product':table=np.isin(names,list(PRODUCTS))
    else:raise ValueError(kind)
    return table[tokens]


def adjacent(mask):
    out=np.zeros(mask.shape,bool)
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        out[tuple(a)]|=mask[tuple(b)];out[tuple(b)]|=mask[tuple(a)]
    return out


def target_change_mask(previous,current,vocabulary,category):
    if previous.meta['center_origin']!=current.meta['center_origin']:raise ValueError('Dynamic center moved')
    delta=previous.tokens!=current.tokens
    if category in('D1','D2'):
        kind='water'if category=='D1'else'lava'
        relevant=kind_mask(previous.tokens,vocabulary,kind)|kind_mask(current.tokens,vocabulary,kind)
        if category=='D2':relevant&=~kind_mask(current.tokens,vocabulary,'product')
    elif category=='D3':
        old=kind_mask(previous.tokens,vocabulary,'fire');new=kind_mask(current.tokens,vocabulary,'fire')
        fuel_names=current.meta.get('flammable_names',[])
        table=np.array([name in fuel_names for name,_ in vocabulary.entries],bool)
        relevant=old|new|((table[previous.tokens]|table[current.tokens])&adjacent(old|new))
        names=np.asarray([name for name,_ in vocabulary.entries],object)
        delta&=~(old&new&(names[previous.tokens]==names[current.tokens])) # Fire age alone is not structural propagation.
    elif category=='D4':
        relevant=np.zeros(delta.shape,bool)
        for kind in('water','lava'):relevant|=kind_mask(previous.tokens,vocabulary,kind)|kind_mask(current.tokens,vocabulary,kind)
        relevant|=kind_mask(current.tokens,vocabulary,'product')&~kind_mask(previous.tokens,vocabulary,'product')
    else:raise ValueError(category)
    return delta&relevant


def target_changes(previous,current,vocabulary,category):
    return int(target_change_mask(previous,current,vocabulary,category)[CENTER].sum())


def fire_change_counts(previous,current,vocabulary):
    """Center-only observed transitions; near-fire fuel loss is not a write journal."""
    names=np.array([name for name,_ in vocabulary.entries],object)
    before=names[previous.center];after=names[current.center];delta=previous.center!=current.center
    old=kind_mask(previous.tokens,vocabulary,'fire');new=kind_mask(current.tokens,vocabulary,'fire')
    a=old[CENTER];b=new[CENTER];fuel=np.isin(before,current.meta.get('flammable_names',[]))
    near=adjacent(old|new)[CENTER]
    masks=dict(new_fire=delta&(before=='air')&b,fuel_to_fire=delta&fuel&b,
        fuel_removed_near_fire=delta&fuel&(after=='air')&near,fire_extinguished=delta&a&(after=='air'),
        fire_age_update=delta&a&b&(before==after))
    structural=target_change_mask(previous,current,vocabulary,'D3')[CENTER]
    covered=np.zeros((16,)*3,bool)
    for key,mask in masks.items():
        if key!='fire_age_update':covered|=mask
    masks['other_structural_fire_transition']=structural&~covered
    return {key:int(mask.sum())for key,mask in masks.items()}
