"""P1: dry run by default; real native collection is a separate manual --native action."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .candidates import CONFIG,plan,budget,scene_rng,choose,fire_draw,next_plant_window
from .tracking import Vocabulary,Event,Reactions,FireMerges
from .storage import Store,dump
from .report import summarize
from .visualize import save_event

HERE=Path(__file__).parent

def annotate(frames,scene,selected):
    for state in frames.values():
        state.meta.update(scene_id=scene['scene_id'],instance_id=scene['instance_id'],setting=scene['setting'],
            category=scene['category'],subtype=scene['subtype'],source_positions=[c['position'] for c in selected],
            data_source='synthetic_dry_run' if scene.get('synthetic') else 'native')
    return frames

def record_frames(store,frames,scene,phase):
    for state in frames.values():store.raw.append(state,scene,phase)

def collect(world,scene,store,pairs):
    start=time.monotonic();selected=[];events={};first=None
    scene.update(status='running',interventions=[],intervention_attempts=0,intervention_successes=0)
    prepared=world.prepare();scene['lifecycle']=prepared.get('lifecycle',{});scene['boot']=prepared.get('boot')
    scene['generated_region_callbacks']=prepared.get('generated_region_callbacks',[])
    if prepared['status']!='ready':scene['status']='Invalid Physical Environment';return
    first,clock=world.capture();annotate(first,scene,[]);record_frames(store,first,scene,'activation_first_capture')
    scene['lifecycle']['first_capture']=clock
    rng=scene_rng(scene['world_seed'],scene['category'],scene['index'],scene['subtype'])
    discovery=world.discover(CONFIG['candidate_budget_per_subtype'],int(rng.integers(1,2147483646)))
    scene['discovery']={k:v for k,v in discovery.items() if k!='candidates'}
    scene['lifecycle']['selection_completed']={k:discovery[k] for k in ('sim_time','simulation_step') if k in discovery}
    candidates=discovery['candidates']
    if scene['setting']=='C' and scene['category']=='D1':
        context=('pool_edge','vertical_drop','narrow_channel','obstacle','chunk_boundary')[scene['index']%5]
        candidates=[c for c in candidates if context in c['features'].get('contexts',[c['features']['context']])]
        scene['requested_flow_context']=context;scene['context_eligible']=len(candidates)
    if scene['setting']=='B':
        if scene['pair_id'] not in pairs:raise ValueError('B requires A frozen candidates from the same plan or --paired-candidates')
        selected=pairs[scene['pair_id']];scene['paired_selection_recheck']=world.recheck(selected)
    else:
        count=20 if scene['category']=='D3' and scene['setting']=='C' else CONFIG['parallel_candidates'] if scene['category']=='D4' else 1
        spacing=CONFIG['fire_spacing_voxels'] if scene['category']=='D3' else 2 if scene['category']=='D4' else 0
        selected=choose(candidates,rng,count,spacing)
        if scene['setting']=='A':pairs[scene['pair_id']]=selected
    scene['selected_candidates']=selected
    if not selected or scene['category']=='D3' and scene['setting']=='C' and len(selected)<20:
        scene['status']='No Candidate' if discovery['discovered']==0 else 'Invalid Physical Environment'
        scene['failure_detail']='fewer_than_20_dispersed_fire_points' if scene['category']=='D3' and scene['setting']=='C' else 'no_eligible_candidate_in_requested_context'
        if first:
            e=Event(next(iter(first.values())),scene['category'],store.v,scene['subtype'])
            scene['visualization']=save_event(store.root,scene,e,e.records)
        scene['wall_seconds']=time.monotonic()-start;return
    annotate(first,scene,selected)
    previous=first;delay_changed=0
    if scene['setting']=='B':
        target=clock['sim_time']+scene['delay_seconds']
        while clock['sim_time']<target:
            world.wait_until(min(target,clock['sim_time']+CONFIG['capture_seconds'][scene['category']]))
            current,clock=world.capture();annotate(current,scene,selected);record_frames(store,current,scene,'delayed_natural_wait')
            delay_changed+=sum(int(np.count_nonzero((current[o].tokens!=previous[o].tokens)&
                ((store.v.plant_mask(current[o].tokens,scene['subtype'])|store.v.plant_mask(previous[o].tokens,scene['subtype'])) if scene['category']=='D4' else
                 (store.v.mask(current[o].tokens,scene['category'])|store.v.mask(previous[o].tokens,scene['category']))))) for o in current)
            previous=current
        scene['delay_changed_voxels']=delay_changed;initial=previous
    elif scene['setting']=='C':
        pre,clock=world.capture();annotate(pre,scene,selected);record_frames(store,pre,scene,'intervention_before')
        if scene['category']=='D3':
            draw=fire_draw(selected,rng);scene['fire_randomization']=draw;chosen=[selected[i] for i in draw['selected_indices']]
        else:chosen=selected
        initial,clock=world.intervene(chosen);annotate(initial,scene,selected);record_frames(store,initial,scene,'intervention_after')
        scene['interventions'].extend(clock['interventions'])
        if scene['category']=='D3' and not any(r['success'] for r in clock['interventions']):
            rest=[i for i in range(20) if i not in scene['fire_randomization']['selected_indices']]
            if rest:
                fallback=int(rng.choice(rest));scene['fire_failure_fallback_index']=fallback
                initial,clock=world.intervene([selected[fallback]],fallback=True)
                annotate(initial,scene,selected);record_frames(store,initial,scene,'intervention_fallback_after')
                scene['interventions'].extend(clock['interventions'])
        scene['intervention_attempts']=len(scene['interventions']);scene['intervention_successes']=sum(bool(r['success']) for r in scene['interventions'])
        scene['lifecycle']['intervention_completed']={k:clock[k] for k in ('sim_time','simulation_step')}
        if not scene['intervention_successes']:
            scene['status']='Initialization Failed';scene['wall_seconds']=time.monotonic()-start
            e=Event(next(iter(initial.values())),scene['category'],store.v,scene['subtype']);scene['visualization']=save_event(store.root,scene,e,e.records);return
    else:
        initial=first # Preserve changes during eligibility search, not just after it.
    baseline_time=next(iter(initial.values())).meta['timestamp'];latest_time=baseline_time
    scene['lifecycle']['observation_start']=dict(sim_time=baseline_time,simulation_step=next(iter(initial.values())).meta['simulation_step'])
    events={o:Event(s,scene['category'],store.v,scene['subtype']) for o,s in initial.items()}
    reaction=Reactions(store.v) if scene['category']=='D2' else None
    ignition_positions=[r['position'] for r in scene['interventions'] if r['success']]
    fire=FireMerges(store.v,ignition_positions) if scene['category']=='D3' else None
    previous=initial;window=CONFIG['observation_seconds'][scene['category']];scene['extensions']=[]
    try:
        while True:
            world.wait_until(latest_time+CONFIG['capture_seconds'][scene['category']])
            frames,clock=world.capture();annotate(frames,scene,selected);record_frames(store,frames,scene,'native_observation')
            latest_time=clock['sim_time'];masks=reaction.observe(previous,frames) if reaction else {}
            for origin,state in frames.items():events[origin].append(state,masks.get(origin))
            if fire:fire.observe(frames)
            previous=frames
            if any(len(e.distinct())>=8 and e.has_native_evolution() for e in events.values()):
                scene['early_stop']='sufficient_8_state_native_event';break
            if latest_time-baseline_time>=window:
                if scene['category']!='D4':break
                recheck=world.recheck(selected);new_window=next_plant_window(window,recheck['can_continue_growth'])
                scene['extensions'].append(dict(at_seconds=latest_time-baseline_time,recheck=recheck,next_window=new_window))
                if new_window is None:break
                window=new_window
    finally:
        rows=[];diagnostics=[]
        for e in events.values():
            row,states=store.save_event(e,scene);rows.append(row);diagnostics.append((e,states))
        best=max(diagnostics,key=lambda item:(item[0].has_native_evolution(),len(item[0].distinct()),item[0].updates))
        scene['visualization']=save_event(store.root,scene,*best)
        scene['observation_seconds']=max(0,latest_time-baseline_time)
        scene['standard_snapshots']=sum(r['standard_states'] for r in rows)
        scene['native_evolution']=any(e.has_native_evolution() for e in events.values())
        times=[e.first_native_time for e in events.values() if e.first_native_time is not None]
        scene['first_native_change_latency']=min(times)-baseline_time if times else None
        growth_times=[e.first_growth_time for e in events.values() if e.first_growth_time is not None]
        scene['first_growth_latency']=min(growth_times)-baseline_time if growth_times else None
        scene['confirmed_products']=len(reaction.products) if reaction else 0
        if reaction:
            scene['water_lava_contacts']=reaction.contacts;scene['water_lava_products']=reaction.products
            scene['fluid_state_counts']=reaction.series
        if fire:scene['fire_merges']=fire.merges
        scene['wall_seconds']=time.monotonic()-start
    if scene['standard_snapshots']:scene['status']='accepted'
    elif scene['category']=='D4' and sum(e.losses for e in events.values()) and not any(e.added.any() for e in events.values()):scene['status']='Native Decay Instead of Growth'
    elif not scene['native_evolution']:scene['status']='Already Stabilized Before Observation' if delay_changed else 'No Native Evolution'
    else:scene['status']='Insufficient Distinct States'
    latency=scene['first_growth_latency'] if scene['category']=='D4' else scene['first_native_change_latency']
    scene['delayed_evolution']=bool(latency is not None and latency>=CONFIG['observation_seconds'][scene['category']])
    if scene['status']!='accepted' and scene['delayed_evolution']:scene['status']='Delayed Evolution'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--native',action='store_true');mode.add_argument('--dry-run',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--settings',choices=list('ABC'),nargs='+',default=list('ABC'))
    p.add_argument('--categories',choices=('D1','D2','D3','D4'),nargs='+',default=['D1','D2','D3','D4'])
    p.add_argument('--sites-per-category',type=int,default=1);p.add_argument('--delay-seconds',type=float,default=CONFIG['delayed_natural_seconds'])
    p.add_argument('--wall-seconds',type=float);p.add_argument('--paired-candidates',type=Path)
    p.add_argument('--renderer',choices=('auto','window','software'),default='auto')
    p.add_argument('--craftium-dir',type=Path,default=HERE.parent/'third_party/craftium')
    a=p.parse_args(argv)
    if a.native and (not a.wall_seconds or a.wall_seconds<=0):p.error('--native requires an explicit positive --wall-seconds budget')
    if a.sites_per_category<1 or a.delay_seconds<0 or any(seed<0 or seed>2147483647 for seed in a.seeds):p.error('Invalid scene count, delay or world seed')
    a.settings=[s for s in 'ABC' if s in a.settings]
    if 'B' in a.settings and 'A' not in a.settings and not a.paired_candidates:p.error('B-only needs --paired-candidates from its independent A run')
    return a


def run(args):
    scenes=plan(args.seeds,args.categories,args.settings,args.sites_per_category,args.delay_seconds);planned=budget(scenes)
    v=Vocabulary();store=Store(args.output,v);pairs={};completed=[];failed=False
    if args.paired_candidates:pairs=json.loads(args.paired_candidates.read_text())['selections']
    dump(store.root/'plan.json',dict(scenes=scenes,budget=planned,config=CONFIG,native=args.native))
    deadline=time.monotonic()+(args.wall_seconds or 30)
    if args.native:
        from .backend import NativeWorld
        factory=lambda scene:NativeWorld(scene,v,deadline,args.renderer,args.craftium_dir)
    else:
        from .dry_run import DryWorld
        factory=lambda scene:DryWorld(scene,v)
    try:
        for specification in scenes:
            scene=dict(specification,instance_id=uuid.uuid4().hex,synthetic=not args.native);start=time.monotonic()
            try:
                with factory(scene) as world:collect(world,scene,store,pairs)
            except Exception as exc:
                scene.update(status='Collection / Encoding Failure',error=f'{type(exc).__name__}: {exc}',censored=isinstance(exc,TimeoutError),wall_seconds=time.monotonic()-start)
                failed=True
            finally:
                scene['collection_wall_seconds']=scene.get('wall_seconds')
                scene['wall_seconds']=time.monotonic()-start
            completed.append(scene);store.flush()
            dump(store.root/'pairs.json',dict(source_root=str(store.root),selections=pairs))
            summarize(store.root,completed,store.events,args.native,planned)
            print(scene['scene_id'],scene['status'],scene.get('standard_snapshots',0),flush=True)
            if failed:break
    finally:store.close()
    print('Report:',store.root/'VAE_P1_RESULTS.md',flush=True)
    return 2 if failed else 0

def main(argv=None):raise SystemExit(run(parse_args(argv)))
if __name__=='__main__':main()
