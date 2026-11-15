"""Broad discovery, repeatable scene reconstruction and uninterrupted native collection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .storage import RawTrajectory
from vae_p1.visualize import save_event
from .core import CONFIG,BIOMES,Event,Reactions,FireMerges,dump,state_hash,region_of,source_actions,fire_actions,trim_actions
from .engine import World,SETTINGS

def stamp(frames,scene,rollout,phase):
    for state in frames.values():state.meta.update(phase=phase,rollout_id=rollout,scene_id=scene['scene_id'],data_source='native')
    return frames

def inspect_counts(state,v):
    names=v.names(state.tokens)
    return dict(water=int(v.mask(state.tokens,'D1').sum()),lava=int(v.mask(state.tokens,'D2').sum()),
        fire=int(v.mask(state.tokens,'D3').sum()),bamboo=int(v.plant_mask(state.tokens,'bamboo').sum()),
        reeds=int(v.plant_mask(state.tokens,'reeds').sum()),vine=int(v.plant_mask(state.tokens,'vine').sum()),
        products=int(np.isin(names,['mcl_core:stone','mcl_core:cobble','mcl_core:obsidian']).sum()),
        flammable_signal=int(sum('leaves' in n or 'tree' in n or 'wood' in n for n in names.ravel())))

def discover(dataset,seeds,sites,deadline,craftium,renderer):
    report=[]
    for seed in seeds:
        with World(seed,dataset.v,deadline,craftium,renderer) as world:
            for site in sites:
                if time.monotonic()>=deadline:raise TimeoutError('Discovery wall budget reached')
                p=world.prepare(site)
                if p['status']!='ready':continue
                frames,r=world.capture();info=world.inspect(seed);region=region_of(p)
                for origin,state in frames.items():
                    signal=inspect_counts(state,dataset.v);types=['Ordinary']
                    if signal['water']:types.append('D1')
                    if signal['lava'] or signal['water']:types.append('D2')
                    if signal['flammable_signal'] and info['fire_legal_count']:types.append('D3')
                    if signal['bamboo']+signal['reeds']+signal['vine']:types.append('D4')
                    ident=hashlib.sha256(json.dumps([seed,origin,world.build,SETTINGS],sort_keys=True).encode()).hexdigest()[:20]
                    region_id=hashlib.sha256(json.dumps([seed,region,world.build],sort_keys=True).encode()).hexdigest()[:20]
                    scene=dict(scene_id=ident,region_id=region_id,world_seed=seed,origin=list(origin),region=region,scene_types=types,
                        biome=state.meta['biome'],biome_group=state.meta.get('biome_group'),stratum=state.meta['stratum'],
                        height=state.meta['height'],signals=signal,discovery_score=signal['water']+signal['lava']+5*(signal['bamboo']+signal['reeds']+signal['vine']),
                        build=world.build,mapgen='v7',config=SETTINGS,discovery_source=dict(site=site,phase='activation_capture',sim_time=r['sim_time'],step=r['simulation_step']),
                        discovery_state_hash=state_hash(state,dataset.v),initial_state_hash=None,
                        geometry=dict(legal_fire_points=info['fire_legal_count'],attached_air=info['attached_count'],suspended_air=info['suspended_count']),
                        audit=r['audit'],signals_are_not_verified_events=True)
                    dataset.cache_scene(scene,state);report.append(ident)
                dataset.flush_if_due()
                print('discovered',seed,site,'scenes',len(frames),'plants',{k:len(x) for k,x in info['plants'].items()},flush=True)
    return report

def initial_layout(world,scene,dataset,rollout,raw):
    world.verify_scene(scene)
    world.prepare();bootstrap,result,cached=world.initial()
    reference={};phase={};early_series=[]
    lifecycle_frames=sorted([*cached,(bootstrap,result,'activation_capture')],key=lambda item:item[1]['sim_time'])
    for frames,clock,label in lifecycle_frames:
        stamp(frames,scene,rollout,label)
        for origin,state in frames.items():
            raw.append(state,dict(scene,instance_id=rollout,setting='replay',category='Ordinary',subtype=''),label)
            if origin not in reference:
                reference[origin]=state_hash(state,dataset.v);phase[origin]=label
        if label=='early_native':early_series.append((frames,clock))
    stamp(bootstrap,scene,rollout,'activation_capture')
    for origin,state in bootstrap.items():
        if origin not in reference:reference[origin]=state_hash(state,dataset.v);phase[origin]='activation_capture'
    first=early_series[0][0] if early_series else bootstrap
    current,r=world.capture();stamp(current,scene,rollout,'first_python_capture')
    for state in current.values():raw.append(state,dict(scene,instance_id=rollout,setting='replay',category='Ordinary',subtype=''),'first_python_capture')
    return first,current,r,early_series,dict(initial_hashes={','.join(map(str,k)):v for k,v in reference.items()},
        hash_phases={','.join(map(str,k)):v for k,v in phase.items()},lifecycle=result.get('lifecycle'),generation=result.get('generation'),
        boot=result.get('boot'),first_python_capture=dict(sim_time=r['sim_time'],step=r['simulation_step']))

def reconstruct(dataset,scene,repeats,seconds,deadline,craftium,renderer):
    results=[]
    for i in range(repeats):
        rollout='replay_'+uuid.uuid4().hex;folder=dataset.root/'trajectories'/rollout;folder.mkdir()
        raw=RawTrajectory(folder,dataset)
        try:
            with World(scene['world_seed'],dataset.v,deadline,craftium,renderer,scene['region']) as world:
                first,current,r,cached,meta=initial_layout(world,scene,dataset,rollout,raw)
                series=[];start=r['sim_time']
                while r['sim_time']-start<seconds:
                    world.wait(r['sim_time']+0.25);current,r=world.capture();stamp(current,scene,rollout,'replay_observation')
                    for state in current.values():raw.append(state,dict(scene,instance_id=rollout,setting='replay',category='Ordinary',subtype=''),'replay_observation')
                    series.append(dict(time=r['sim_time'],hashes={str(o):state_hash(s,dataset.v) for o,s in current.items()}))
                result=dict(rollout_id=rollout,scene_id=scene['scene_id'],**meta,temporal_series=series,audit=r['audit'],build=world.build)
                results.append(result);dataset.job(rollout,'complete',result)
        finally:raw.close();dataset.v.save(dataset.root/'vocabulary.json')
    same=all(x['initial_hashes']==results[0]['initial_hashes'] for x in results)
    temporal=all(x['temporal_series']==results[0]['temporal_series'] for x in results)
    result=dict(scene_id=scene['scene_id'],repeats=repeats,spatial_initial_state_reproduced=same,
        exact_temporal_trajectory_reproduced=temporal,rollouts=results,
        matching_initial_chunks=[key for key in results[0]['initial_hashes'] if all(r['initial_hashes'].get(key)==results[0]['initial_hashes'][key] for r in results)],
        varying_initial_chunks=[key for key in results[0]['initial_hashes'] if any(r['initial_hashes'].get(key)!=results[0]['initial_hashes'][key] for r in results)],
        interpretation='Independent native rollouts; initial equality does not imply deterministic ABM timing.')
    dump(dataset.root/'replay_consistency.json',result)
    scene['initial_state_hash']=results[0]['initial_hashes'];scene['initial_hash_phases']=results[0]['hash_phases'];dataset.add_scene(scene)
    return result

def append_raw(raw,frames,scene,rollout,category,subtype,phase):
    stamp(frames,scene,rollout,phase)
    for state in frames.values():raw.append(state,dict(scene,instance_id=rollout,setting='unified',category=category,subtype=subtype),phase)

def collection(dataset,scene,category,subtype,seconds,deadline,craftium,renderer,intervention='auto',source_mode=None,job_id=None):
    rollout=uuid.uuid4().hex;job_id=job_id or scene['scene_id']+'_'+category+'_'+subtype
    folder=dataset.root/'trajectories'/rollout;folder.mkdir();raw=RawTrajectory(folder,dataset)
    start_wall=time.monotonic();record=dict(scene_id=scene['scene_id'],world_seed=scene['world_seed'],rollout_id=rollout,
        category=category,subtype=subtype,requested_seconds=seconds,status='running',source='native',intervention_rng_seed=scene['world_seed'],events=[])
    try:
        if category=='Ordinary' and 'cache_index' in scene:
            related=[s for s in dataset.scenes() if s.get('region_id')==scene.get('region_id') and 'cache_index' in s]
            saved=0
            for source in related:
                state=dataset.cached_state(source)
                if dataset.append(state,'Ordinary',source,rollout,extra=dict(semantic_overlap=inspect_counts(state,dataset.v),source_phase='cached_native_discovery')):saved+=1
            record.update(status='complete',standard_snapshots=saved,observation_seconds=0,audit=scene['audit'],source_phase='cached_native_discovery');return record
        with World(scene['world_seed'],dataset.v,deadline,craftium,renderer,scene['region']) as world:
            first,current,r,cached,layout=initial_layout(world,scene,dataset,rollout,raw);record.update(layout)
            record['entity_probe']=world.entity_probe()['audit'];record['source_build']=world.build
            if category=='Ordinary':
                saved=sum(dataset.append(state,'Ordinary',scene,rollout,extra=dict(semantic_overlap=inspect_counts(state,dataset.v))) for state in current.values())
                record.update(status='complete',standard_snapshots=saved,observation_seconds=0,audit=r['audit']);return record
            initial=first;events={o:Event(s,category,dataset.v,subtype) for o,s in first.items()};previous=first
            reactions=Reactions(dataset.v) if category=='D2'else None;fire=None
            for frames,clock in [*cached,(current,r)]:
                masks=reactions.observe(previous,frames) if reactions else{}
                for origin,state in frames.items():events[origin].append(state,masks.get(origin))
                previous=frames
            early_native=sum(e.updates for e in events.values())
            need=intervention=='always' or intervention=='auto' and not any(len(e.distinct())>=5 and e.has_native_evolution() for e in events.values())
            if intervention=='never':need=False
            record['early_native_changes']=early_native
            if need:
                info=world.inspect(scene['world_seed'])
                selection_seed=int.from_bytes(hashlib.sha256(json.dumps([scene['world_seed'],scene['scene_id'],category,subtype,job_id]).encode()).digest()[:8],'little') & ((1<<63)-1)
                rng=np.random.default_rng(selection_seed);record['intervention_rng_seed']=selection_seed
                if category in ('D1','D2'):actions,selection=source_actions(info,category,rng,mode=source_mode,near_water=category=='D2'and scene['signals']['water']>0)
                elif category=='D3':actions,selection=fire_actions(info,rng)
                else:actions,selection=trim_actions(info,subtype,rng)
                record['selection']=selection
                append_raw(raw,current,scene,rollout,category,subtype,'intervention_before')
                current,r=world.intervene(actions);append_raw(raw,current,scene,rollout,category,subtype,'intervention_after')
                record['interventions']=r['actions']
                if category=='D3' and actions and not any(a.get('after')=='mcl_fire:fire' for a in r['actions']):
                    rest=[c for c in selection['sparse_candidates'] if c['position'] not in [a['position'] for a in actions]]
                    if rest:
                        c=rest[int(rng.integers(len(rest)))];current,r=world.intervene([dict(type='fire',position=c['position'],under=c['under'])])
                        record['fallback_interventions']=r['actions'];append_raw(raw,current,scene,rollout,category,subtype,'intervention_fallback_after')
                if category=='D4':
                    settle_start=r['sim_time'];last=current
                    while r['sim_time']-settle_start<CONFIG['plant_settle_seconds']:
                        world.wait(r['sim_time']+0.5);current,r=world.capture();append_raw(raw,current,scene,rollout,category,subtype,'intervention_settling')
                        last=current
                initial=current;events={o:Event(s,category,dataset.v,subtype) for o,s in current.items()};previous=current
                reactions=Reactions(dataset.v) if category=='D2'else None
                if category=='D3':fire=FireMerges(dataset.v,[a['position'] for a in record['interventions'] if a.get('after')=='mcl_fire:fire'])
                with open(dataset.root/'interventions.jsonl','a')as f:f.write(json.dumps(record)+'\n')
            baseline=next(iter(initial.values())).meta['timestamp'];observation_start=r['sim_time'];last_change=r['sim_time']
            while r['sim_time']-observation_start<seconds:
                world.wait(r['sim_time']+CONFIG['capture_seconds'][category]);frames,r=world.capture()
                append_raw(raw,frames,scene,rollout,category,subtype,'native_observation')
                masks=reactions.observe(previous,frames) if reactions else{}
                changed=0
                for origin,state in frames.items():
                    changed+=int(np.count_nonzero(state.tokens!=previous[origin].tokens));events[origin].append(state,masks.get(origin))
                if fire:fire.observe(frames)
                if changed:last_change=r['sim_time']
                previous=frames
                # Plants never stop merely because one chunk reached 8 states.
                if category!='D4' and r['sim_time']-last_change>=CONFIG['stable_seconds']:break
            # Audit is checked before committing ANY training state from this rollout.
            audit=world.entity_probe()['audit'];record['audit']=audit
            summaries=[];saved=0;aux=0
            for origin,event in events.items():
                from .sampling import select_states
                unique=event.distinct();chosen=select_states(event) if event.has_native_evolution() else[]
                if len(chosen)>=5:
                    from vae_pilot.sampling import clip_size
                    n=clip_size(len(chosen),CONFIG['quotas'][category]-dataset.counts().get(category,0))
                    if n and n<len(chosen):chosen=select_states(event,n)
                    elif not n:chosen=[]
                if 2<=len(unique)<=4:chosen=unique;pool='aux_short_event'
                else:pool='standard'
                event_id=rollout+'/'+','.join(map(str,origin))
                for state in chosen:
                    if dataset.append(state,category,scene,rollout,pool,dict(event_id=event_id,change_progress=state.progress,
                        initial_state_hash=state_hash(initial[origin],dataset.v),intervention_rng_seed=record['intervention_rng_seed'],
                        event_type='growth' if category=='D4' and event.has_native_evolution() else 'decay' if category=='D4' else 'native_dynamics',
                        native_event_confirmed=event.has_native_evolution(),
                        event_attribution='confirmed_contact' if event.reactions else 'unconfirmed' if category=='D2'else 'native_voxel_changes',
                        semantic_overlap=inspect_counts(state,dataset.v))):
                        if pool=='standard':saved+=1
                        else:aux+=1
                summary=event.result();summary.update(event_id=event_id,pool=pool,selected_state_count=len(chosen),
                    selected_progress=[s.progress for s in chosen],selected_timestamps=[s.meta['timestamp'] for s in chosen],
                    selected_steps=[s.meta['simulation_step'] for s in chosen])
                summaries.append(dict(origin=list(origin),**summary))
            best=max(events.values(),key=lambda e:(e.has_native_evolution(),len(e.distinct()),e.updates,int(e.mask(e.records[0]).sum())))
            view_scene=dict(scene_id=rollout,category=category,subtype=subtype,synthetic=False)
            record['visualization']=save_event(dataset.root,view_scene,best,best.select() or best.distinct()[:8])
            if category=='D4':
                from .visualize import phases
                raw.flush();record['phase_visualization']=phases(dataset.root,rollout,category,subtype)
            record.update(status='complete',events=summaries,standard_snapshots=saved,aux_snapshots=aux,
                observation_seconds=r['sim_time']-baseline,native_changes=sum(e.updates for e in events.values()),
                native_growth_positions=sum(int(e.added.sum()) for e in events.values()),confirmed_reactions=reactions.products if reactions else[],
                water_lava_contacts=reactions.contacts if reactions else[],fire_merges=fire.merges if fire else[])
            return record
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');return record
    finally:
        raw.close();record['wall_seconds']=time.monotonic()-start_wall
        dataset.job(job_id,record['status'],record);dataset.flush_if_due()

def schedule(dataset,category,seed_offset=0):
    counts=Counter()
    if category=='Ordinary':
        for biome,n in dataset.db.execute("SELECT biome,COUNT(*) FROM samples WHERE category='Ordinary' AND pool='standard' GROUP BY biome"):counts[biome]=n
    candidates=[s for s in dataset.scenes() if category in s['scene_types']]
    # First balance biome and source seed, then prefer diverse useful signals.
    candidates.sort(key=lambda s:(counts[s.get('biome_group')],s['world_seed'], -s['discovery_score']))
    return candidates[seed_offset:]+candidates[:seed_offset]
