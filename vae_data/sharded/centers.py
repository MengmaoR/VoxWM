"""Cheap center-only geometry screening, separate from expensive all-source planning."""
from collections import defaultdict
import time
import numpy as np
from .protocol import CENTER,kind_mask,adjacent

AIR_MIN=65
POOL_MIN=19

def components(mask,minimum=1):
    mask=np.asarray(mask,bool);seen=np.zeros(mask.shape,bool);accepted=np.zeros(mask.shape,bool);largest=0
    for start in np.argwhere(mask):
        start=tuple(start)
        if seen[start]:continue
        stack=[start];seen[start]=True;cells=[]
        while stack:
            p=stack.pop();cells.append(p)
            for axis in range(3):
                for step in(-1,1):
                    q=list(p);q[axis]+=step
                    if 0<=q[axis]<mask.shape[axis]:
                        q=tuple(q)
                        if mask[q]and not seen[q]:seen[q]=True;stack.append(q)
        largest=max(largest,len(cells))
        if len(cells)>=minimum:
            xyz=np.array(cells);accepted[tuple(xyz.T)]=True
    return largest,accepted

def choose_height(category,rng):
    return 'surface'if category in('D1','D3')else('surface','cave')[int(rng.integers(2))]

def pool_kind(category,subtype):return ('water'if subtype=='A'else'lava')if category=='D4'and subtype in('A','C')else''

def prepare_dynamic(world,site,category,rng,subtype='',height_type=None,diagnostic=False):
    height_type=height_type or choose_height(category,rng)
    if height_type not in('surface','cave')or category in('D1','D3')and height_type!='surface':raise ValueError('Unsupported dynamic height type')
    result=world.find_centers(site,height_type,pool_kind(category,subtype));candidates=result.get('candidates')or[]
    if not candidates and not(diagnostic and result.get('diagnostic')):raise ValueError('No qualified '+height_type+' center at the bounded site: '+str(result.get('rejected')))
    # Validation only: retain one real rejected region for diagnosis, never plan/place in it.
    info=dict(candidates[int(rng.integers(len(candidates)))]if candidates else result['diagnostic'])
    info.update(candidate_count=len(candidates),probe_domain=result['probe_domain'],probe_rejections=result['rejected'],
                selection='uniform aligned chunk; no generic natural-fluid/flammable voxel recentering')
    frame,response=world.prepare_at(info['center'],info['surface_y'])
    world.scene_selection=info;frame.meta.update(scene_selection=info,height_type=height_type,stratum=height_type)
    return frame,response

def spatial_gate(frame,vocabulary):
    start=time.perf_counter();names=np.array([n for n,_ in vocabulary.entries],object)[frame.tokens]
    air=names[CENTER]=='air';largest,_=components(air)
    info=frame.meta.get('scene_selection')or{};height=frame.meta.get('height_type',info.get('height_type'))
    result=dict(height_type=height,largest_connected_air_component=largest,air_required=AIR_MIN)
    if height not in('surface','cave'):return False,dict(result,reason='missing_dynamic_height_evidence',filter_seconds=time.perf_counter()-start)
    if largest<AIR_MIN:return False,dict(result,reason='center_connected_air_below_65',filter_seconds=time.perf_counter()-start)
    terrain=np.isin(names,frame.meta.get('terrain_names',[]));surfaces=np.asarray(info.get('solid_surface_heights',[]))
    if surfaces.shape!=(16,16):return False,dict(result,reason='missing_solid_surface_grid',filter_seconds=time.perf_counter()-start)
    # Surface evidence must be a genuine outdoor solid–AIR interface in this center,
    # not a water top or a cave floor below an overhead terrain roof.
    boundary=0
    for x in range(16):
        for y in range(16):
            z=int(surfaces[x,y])-frame.meta['region_origin'][2]
            if 16<=z<32 and terrain[16+x,16+y,z]and names[16+x,16+y,z+1]=='air':boundary+=1
    z=np.arange(frame.meta['center_origin'][2],frame.meta['center_origin'][2]+16)
    underground=air&(z[None,None,:]<surfaces[:,:,None]);largest_cave,_=components(underground)
    result.update(surface_boundary_columns=boundary,largest_underground_air_component=largest_cave)
    valid=boundary>0 if height=='surface'else boundary==0 and largest_cave>=AIR_MIN
    result.update(reason=None if valid else'not_real_'+height+'_geometry',filter_seconds=time.perf_counter()-start)
    return valid,result

def stable_pool(before,after,vocabulary,kind):
    mask=(before.center==after.center)&kind_mask(before.center,vocabulary,kind)&kind_mask(after.center,vocabulary,kind)
    largest,qualified=components(mask,POOL_MIN);region=np.zeros((48,)*3,bool);region[CENTER]=qualified
    return largest,region

def natural_interaction(frame,vocabulary):return bool((kind_mask(frame.tokens,vocabulary,'water')&adjacent(kind_mask(frame.tokens,vocabulary,'lava'))).any())

def height_statistics(rows):
    result={}
    for row in rows:
        category=row.get('category');height=row.get('height_type')
        if category not in('D1','D2','D3','D4')or height not in('surface','cave'):continue
        label=category+('-'+row['subtype']if category=='D4'and row.get('subtype')else'')
        counts=result.setdefault(label,{h:dict(attempts=0,spatial_qualified=0,qualified=0,accepted=0)for h in('surface','cave')})[height]
        counts['attempts']+=1;counts['spatial_qualified']+=bool(row.get('scene_spatial_qualified'));counts['qualified']+=bool(row.get('planning_qualified'))
        counts['accepted']+=row.get('status')in('success','committed')
    return result
