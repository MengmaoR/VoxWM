"""Bounded native attempts, strict center-only collection, and per-shard scheduling."""
from pathlib import Path
from collections import Counter
import json,time,subprocess
import numpy as np
from .protocol import CONFIG,BIOMES,STRATA,Frame,Chain,kind_mask,biome_group,dump
from .native import Native
from .sampling import Collector
from .centers import prepare_dynamic,choose_height,spatial_gate,stable_pool,natural_interaction
from .planning import Planner
from .storage import Writer,atomic_json


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


def collect(world,frame,category,rng,subtype='',max_seconds=None,trace=None):
    """One scene, one all-point plan, one initialization; no native retry inside."""
    intervention=dict(mode='native_initialization',rng_state=rng.bit_generator.state);verified_pool=None
    scene_info=frame.meta.get('scene_selection')or{};height=frame.meta.get('height_type',scene_info.get('height_type'))
    def reject(reason,spatial=False,planned=False,**extra):
        status=dict(status='rejected',reason=reason,subtype=subtype,world_seed=world.seed,height_type=height,
            center_origin=world.center,scene_spatial_qualified=spatial,planning_qualified=planned,intervention=intervention,**extra)
        if trace:trace.finish(status=status)
        return None,status
    if trace:trace.observe(frame,'original_scene')
    if scene_info.get('probe_qualified')is False:return reject('center_probe_'+scene_info['diagnostic_rejection_reason'])
    valid,screen=spatial_gate(frame,world.v);intervention['spatial_screen']=screen
    if not valid:return reject(screen['reason'])
    if category=='D4'and subtype in('A','C'):
        before=frame;world.wait(frame.time+1.1);frame,result=world.capture48()
        if trace:trace.observe(frame,'natural_pool_observation',result.get('reaction_events')or[])
        kind='water'if subtype=='A'else'lava'
        count,verified_pool=stable_pool(before,frame,world.v,kind)
        intervention.update(pool_kind=kind,largest_connected_stable_pool=count,pool_observation_seconds=frame.time-before.time)
        valid,screen=spatial_gate(frame,world.v);intervention['spatial_screen_after_pool']=screen
        if not valid:return reject(screen['reason'])
        if count<19:return reject('center_stable_'+kind+'_component_below_19')
    if category=='D4'and subtype=='B'and natural_interaction(frame,world.v):return reject('preexisting_natural_water_lava_contact')
    if category=='D2':intervention['initial_water_voxels']=int(kind_mask(frame.tokens,world.v,'water').sum())
    planner=Planner(frame,world.v)
    actions,selection=planner.select(category,rng,subtype,verified_pool)
    if trace:trace.planning(planner,selection)
    # Only chosen source details are retained in formal metadata; all-point dense
    # tensors are neither constructed nor stored. The debug trace has streamed grids.
    compact={k:v for k,v in selection.items()if k!='details'}
    compact['selected_sources']=[]
    for detail in selection.get('details',[]):
        grid=detail['estimated_arrival_seconds'];indices=np.flatnonzero(grid.ravel()!=255)
        compact['selected_sources'].append(dict({k:v for k,v in detail.items()if k!='estimated_arrival_seconds'},
            reachable_region_indices=indices.tolist(),estimated_arrival_seconds=grid.ravel()[indices].tolist()))
    intervention['planning']=compact
    if not actions:
        metrics=selection.get('metrics',{})
        reason=selection.get('reason','no_qualified_temporal_center_potential')
        if metrics and all(m['legal_source_count']==0 for m in metrics.values()):reason='no_legal_source'
        return reject(reason,spatial=True)
    if trace:trace.observe(frame,'before_intervention')
    anchor,result=world.place(actions)
    intervention.update(requested_actions=actions,actual_actions=result.get('actions')or[])
    anchor.meta['phase']='initial_anchor'if category=='D4'else'post_initialization'
    if trace:trace.observe(anchor,'after_intervention',result.get('reaction_events')or[])
    if not any(a['before']!=a['after']and(a['after'].startswith('mcl_fire:')if category=='D3'else a['after'].endswith('_source'))for a in intervention['actual_actions']):
        return reject('native_initialization_failed',spatial=True,planned=True)
    if category=='D4'and subtype=='B'and not all(any(a['type']==kind and a['after'].endswith('_source')for a in intervention['actual_actions'])for kind in('water','lava')):
        return reject('both_new_fluids_not_initialized',spatial=True,planned=True)
    monitor=Collector(anchor,category,world.v,max_seconds)
    all_reactions=[];contact_observed=False;contact_center=False;structural=0;age_only=0
    while True:
        world.wait(monitor.previous.time+CONFIG['capture_seconds'])
        current,result=world.capture48();current.meta['phase']='native_evolution'
        events=result.get('reaction_events')or[];all_reactions.extend(events)
        if category=='D4':
            from .protocol import adjacent,CENTER
            contacts=kind_mask(current.tokens,world.v,'water')&adjacent(kind_mask(current.tokens,world.v,'lava'))
            contact_observed|=bool(contacts.any());contact_center|=bool(contacts[CENTER].any())
        if category=='D3':
            names=np.asarray([name for name,_ in world.v.entries],object)
            delta=current.center!=monitor.previous.center
            age=delta&(names[current.center]==names[monitor.previous.center])&kind_mask(current.center,world.v,'fire')
            age_only+=int(age.sum());structural+=int((delta&~age).sum())
        if trace:trace.observe(current,'native_observation',events)
        if monitor.observe(current,events):break
    world.entity_probe()
    chain=monitor.select(rng,intervention,subtype)
    reason=None
    if not chain:
        if category=='D3'and monitor.changes==0:reason='fire_age_only_no_structural_propagation'if age_only else'no_structural_fire_propagation'
        elif category=='D4':
            if not all_reactions:reason='contact_without_solidification'if contact_observed else'water_lava_no_observed_contact'
            elif not monitor.events:reason='solidification_only_in_neighbor_region'
            elif not any(e.get('pre')and e.get('post')for e in monitor.events):reason='solidification_missing_complete_pre_post'
            else:reason='complete_event_but_no_valid_five_state_window'
        elif not monitor.changes:reason='native_flow_did_not_change_center'
        else:reason='fewer_than_four_distinct_center_transitions'
    status=dict(status='success'if chain else'rejected',reason=reason,**monitor.summary(),
        subtype=subtype,world_seed=world.seed,height_type=height,scene_spatial_qualified=True,planning_qualified=True,center_origin=world.center,audit=world.audit,
        observed_reaction_count=len(all_reactions),contact_observed=contact_observed,center_contact_observed=contact_center,
        structural_fire_changes=structural,fire_age_only_changes=age_only,intervention=intervention)
    if trace:trace.finish(monitor,chain,status)
    return chain,status


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
                    if category=='Static':
                        location=world.locate(int(site[0]),int(site[1]));group=biome_group(location.get('biome',''),BIOMES)
                        choices=[m for m in missing['static']if m['biome']==group]
                        if not choices:raise ValueError('Observed biome quota already full or outside the 25 groups')
                        stratum=choices[int(rng.integers(len(choices)))]['stratum']
                    if category=='Static':
                        frame,_=prepare(world,site,stratum,rng)
                        frame.meta['stratum']=stratum;frame.meta['phase']='static_snapshot'
                        saved=writer.submit(Chain('Static',[frame]),world.v);entry.update(status='committed'if saved else'rejected')
                    else:
                        height=choose_height(category,rng);entry.update(height_type=height,subtype=subtype)
                        frame,_=prepare_dynamic(world,site,category,rng,subtype,height)
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
