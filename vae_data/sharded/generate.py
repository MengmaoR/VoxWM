"""Bounded native attempts, strict center-only collection, and per-shard scheduling."""
from pathlib import Path
from collections import Counter
import json,time,subprocess
import numpy as np
from .protocol import CONFIG,BIOMES,STRATA,Frame,Chain,kind_mask,biome_group,dump
from .native import Native
from .sampling import Collector
from .reachability import plan,pool_components
from .storage import Writer,atomic_json
from vae_data.core import fire_actions,source_actions


def prepare(world,site,stratum,rng):
    location=world.locate(int(site[0])//16*16,int(site[1])//16*16)
    if location['status']!='located':raise ValueError(location['status'])
    surface=location['surface_y'];depth={'surface':0,'elevated_complex':-16,'shallow_underground':16,'deep_cave':int(rng.integers(4,9))*16}[stratum]
    center=[int(site[0])//16*16,int(site[1])//16*16,int(np.floor((surface-depth)/16)*16)]
    frame,result=world.prepare_at(center,surface);frame.meta['stratum']=stratum
    frame.meta['terrain_heights']=location['heights'];frame.meta['terrain_native']=result['chunks'][13]['stratum']
    if stratum=='elevated_complex':
        air=np.mean(np.array([name=='air'for name,_ in world.v.entries],bool)[frame.center])
        if not .02<air<.98:raise ValueError('Elevated candidate is empty/simple solid; not used to fill quota')
    return frame,result


def center_hint(world,frame,category,rng):
    if category=='D3':
        table=np.array([name in world.flammable for name,_ in world.v.entries],bool);mask=table[frame.tokens]
    else:mask=kind_mask(frame.tokens,world.v,'water'if category in('D1','D4')else'lava')
    points=np.argwhere(mask)
    if not len(points):return frame
    p=points[int(rng.integers(len(points)))]+np.asarray(frame.meta['region_origin'])
    center=(p//16*16).tolist()
    if center==world.center:return frame
    shifted,_=world.prepare_at(center,world.surface_y);return shifted


def collect(world,frame,category,rng,subtype='',max_seconds=None):
    intervention=dict(mode='native_initialization',rng_state=rng.bit_generator.state);stable_pool=None
    if category=='D4':
        if subtype in('A','C'):
            # Two native observations, not a manufactured pool. Only source cells stable
            # in both are eligible; flowing nodes alone do not establish a static pool.
            before=frame;world.wait(frame.time+1.1);frame,_=world.capture48()
            kind='water'if subtype=='A'else'lava';old=kind_mask(before.tokens,world.v,kind);now=kind_mask(frame.tokens,world.v,kind)
            source_lookup=np.array([kind in name and 'source'in name for name,_ in world.v.entries],bool)
            stable=pool_components((before.tokens==frame.tokens)&old&now&source_lookup[frame.tokens]);stable_pool=stable
            if stable.sum()<4:return None,dict(status='rejected',reason='No observed stable natural pool')
            intervention['pool_observation_seconds']=frame.time-before.time
        proposal=plan(frame,world.v,subtype,rng,stable_pool=stable_pool)
        if proposal is None:return None,dict(status='rejected',reason='No qualified center-contact geometry')
        actions=proposal['actions'];intervention['reachability']=proposal
    else:
        info=world.inspect(int(rng.integers(2147483647)))
        if category=='D3':actions,selection=fire_actions(info,rng,config=CONFIG['initialization'])
        else:actions,selection=source_actions(info,category,rng,config=CONFIG['initialization'])
        intervention['selection']=selection
    if not actions:return None,dict(status='rejected',reason='No legal source/ignition positions')
    anchor,result=world.place(actions);intervention['requested_actions']=actions;intervention['actual_actions']=result.get('actions')or[]
    if category=='D3'and not any(a['after'].startswith('mcl_fire:')for a in intervention['actual_actions']):
        unused=[point for point in selection['sparse_candidates']if point['position']not in[a['position']for a in actions]]
        if unused:
            point=unused[int(rng.integers(len(unused)))];extra=dict(type='fire',position=point['position'],under=point['under'])
            anchor,result=world.place([extra]);intervention['fallback_actions']=result.get('actions')or[]
            intervention['actual_actions']+=intervention['fallback_actions']
    if not any(a['before']=='air'and(a['after'].startswith('mcl_fire:')if category=='D3'else a['after'].endswith('_source'))for a in intervention['actual_actions']):
        return None,dict(status='rejected',reason='Native initialization did not succeed')
    anchor.meta['phase']='initial_anchor'if category=='D4'else'post_initialization'
    monitor=Collector(anchor,category,world.v,max_seconds)
    while True:
        world.wait(monitor.previous.time+CONFIG['capture_seconds'])
        current,result=world.capture48();current.meta['phase']='native_evolution'
        events=result.get('reaction_events')or[]
        if monitor.observe(current,events):break
    world.entity_probe() # cumulative entity/falling audit before committing a chain
    chain=monitor.select(rng,intervention,subtype)
    if chain:
        from .validate import validate_chain
        validate_chain(chain,world.v)
    return chain,dict(status='success'if chain else'rejected',**monitor.summary(),
        subtype=subtype,world_seed=world.seed,center_origin=world.center,
        audit=world.audit,intervention=intervention)


def choose_category(writer,rng):
    missing=writer.missing();available=[k for k,n in missing['categories'].items()if n>0]
    if not available:return None,missing
    weights=np.array([missing['categories'][k]for k in available],float);weights/=weights.sum()
    return str(rng.choice(available,p=weights)),missing


def generate_shard(root,shard,resume,deadline,craftium,renderer,seed,max_attempts,sites_per_world):
    final=Path(root)/'shards'/f'shard_{shard:04d}'
    if (final/'READY').exists():return dict(shard=shard,status='skipped_ready')
    started=time.monotonic();world=None;rng=np.random.default_rng(seed+shard);attempts=0
    try:
        with Writer(root,shard,resume)as writer:
            cursor=writer.path/'cursor.json'
            if resume and cursor.exists():
                old=json.loads(cursor.read_text());rng.bit_generator.state=old['rng'];attempts=old['attempts']
            for _ in range(max_attempts):
                if time.monotonic()+15>=deadline:break
                category,missing=choose_category(writer,rng)
                if category is None:break
                attempts+=1;entry=dict(attempt=attempts,category=category,source='real_native')
                try:
                    if world is None or attempts%sites_per_world==1:
                        if world:world.__exit__(None,None,None)
                        vocabulary=__import__('vae_p0.codec',fromlist=['Vocabulary']).Vocabulary()
                        world=Native(int(rng.integers(1,2147483647)),vocabulary,deadline,craftium,renderer);world.__enter__()
                    site=rng.integers(-8192,8193,2)//16*16
                    stratum='surface';subtype=''
                    if category=='D4':
                        choices=[k for k,n in missing['d4'].items()if n>0];subtype=choices[int(rng.integers(len(choices)))]
                        # Natural lava pools are also searched underground, never manufactured.
                        if subtype=='C':stratum='deep_cave'if rng.random()<.75 else'surface'
                    if category=='Static':
                        location=world.locate(int(site[0]),int(site[1]));group=biome_group(location.get('biome',''),BIOMES)
                        choices=[m for m in missing['static']if m['biome']==group]
                        if not choices:raise ValueError('Observed biome quota already full or outside the 25 groups')
                        stratum=choices[int(rng.integers(len(choices)))]['stratum']
                    frame,_=prepare(world,site,stratum,rng)
                    if category=='Static':
                        frame.meta['stratum']=stratum;frame.meta['phase']='static_snapshot'
                        saved=writer.submit(Chain('Static',[frame]),world.v);entry.update(status='committed'if saved else'rejected')
                    else:
                        frame=center_hint(world,frame,'D2'if category=='D4'and subtype=='C'else category,rng)
                        chain,result=collect(world,frame,category,rng,subtype);entry.update(result)
                        if chain:entry['status']='committed'if writer.submit(chain,world.v)else'rejected_duplicate_or_quota'
                except Exception as exc:
                    entry.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                    if world:world.__exit__(None,None,None);world=None
                writer.attempt(entry);atomic_json(cursor,dict(attempts=attempts,rng=rng.bit_generator.state))
                if attempts%10==0:writer.export()
            if not any(writer.missing()['categories'].values()):
                result=writer.publish();return dict(shard=shard,status='ready',quality=result)
            stats=writer.export();return dict(shard=shard,status='incomplete',statistics=stats,wall_seconds=time.monotonic()-started)
    except Exception as exc:return dict(shard=shard,status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        if world:world.__exit__(None,None,None)
