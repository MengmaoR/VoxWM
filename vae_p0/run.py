"""Protocol P0 only: one initial trial and at most one bounded repair per D1..D4."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from vae_pilot.collect import Probe
from vae_pilot.sampling import biome_group,split_for_seed
from voxwm_gt.session import GTSession,DEFAULT_CRAFTIUM
from .codec import Vocabulary,BUILD
from .sampling import State,Event
from .storage import Output

HERE=Path(__file__).parent
TITLES={'D1':'Water Flow','D2':'Lava Flow','D3':'Fire Propagation','D4':'Vegetation Growth'}
# Actual native candidates from the earlier random-world survey; never prescribed voxel states.
HINTS={'D1':(96,1936,16),'D2':(-256,1472,16),'D3':(-288,1456,16),'D4':(800,-1232,16)}
DURATIONS={'D1':24,'D2':32,'D3':45,'D4':65}
REPAIR_SECONDS={'D1':40,'D2':48,'D3':65,'D4':140}
REASONS={'D1':'Inspect an existing open fluid edge and observe longer, without placing water.',
 'D2':'Inspect deeper native cave layers (64/128 below surface) and require a flowable edge.',
 'D3':'Select a naturally dense, non-wet combustible cluster; one new random ignition only.',
 'D4':'Reject mature/blocked bamboo; prefer supported vines with free growth space; longer unchanged native ABMs.'}


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class P0Probe(Probe):
    def __init__(self,session,deadline,vocabulary,seed):
        super().__init__(session,deadline);self.v=vocabulary;self.seed=seed
        boot=json.loads((session.world/'pilot_boot.json').read_text())
        self.names={int(r['id']):r['name'] for r in boot['nodes']}
        self.groups=json.loads((HERE.parent/'vae_pilot/config.json').read_text())['biome_groups']
        self.verified_nodes=0;self.samples_read=0;self.missing_chunks=0

    def capture(self,origins,surface_y,verify=False):
        result=self.command('p0_capture',origins=origins,surface_y=surface_y,verify=verify)
        raw=(self.s.world/'p0_capture.bin').read_bytes()
        if len(raw)!=len(origins)*4096*4:raise ValueError('Incomplete native Node/light/param2 payload')
        dtype=np.dtype([('node','<u2'),('light','u1'),('param2','u1')])
        array=np.frombuffer(raw,dtype=dtype).reshape(len(origins),16,16,16)
        states={}
        for i,info in enumerate(result['chunks']):
            origin=[int(x) for x in info['origin']]
            if any(x%16 for x in origin) or info['voxel_count']!=4096:raise ValueError('Incorrect fixed world chunk')
            if info['mismatches']:raise ValueError('VoxelManip vs native get_node mismatch')
            self.verified_nodes+=int(info['verified'])
            if info['missing']:self.missing_chunks+=1;continue
            cells=array[i];tokens=self.v.encode(cells['node'],cells['param2'],self.names)
            major=[biome_group(n,self.groups) for n in info['biome_points']]
            meta=dict(world_seed=self.seed,chunk_origin=origin,chunk_index=[x//16 for x in origin],axes='ENU',
                biome=info['biome'],surface_biome=info['surface_biome'],biome_group=major[0],
                biome_transition=len(set(x for x in major if x))>1,
                stratum=info['stratum'],surface_height=info.get('surface_height'),height=info['height'],
                air_fraction=info['air_fraction'],solid_fraction=info['solid_fraction'],
                timestamp=result['sim_time'],simulation_step=int(result['simulation_step']),
                engine_us=int(result['engine_us']),wall_time_unix=int(result['wall_time_unix']),
                split=split_for_seed(self.seed,[80,10,10]),craftium_commit=BUILD['craftium'],voxelibre_commit=BUILD['voxelibre'])
            states[tuple(origin)]=State(tokens,cells['light'].copy(),meta);self.samples_read+=1
        return states,result


def ordinary(probe,out,rng):
    requested=['surface','elevated','surface','shallow_underground','deep'];actual=[]
    for index,stratum in enumerate(requested):
        x,z=(int(n)//16*16 for n in rng.integers(-2048,2049,2))
        layer={'surface':'surface','elevated':'canopy','shallow_underground':'subsurface','deep':'underground'}[stratum]
        visit=probe.command('visit',x=x,z=z,target='ordinary',layer=layer,depth=64 if stratum=='deep' else 16)
        if visit['status']!='ready':continue
        active=probe.command('p0_activate',position=visit['center'])
        if not active['all_blocks_active']:raise RuntimeError('Native region activation failed')
        states,_=probe.capture(active['origins'],visit['surface_y'],verify=True)
        candidates=list(states.values())
        candidates.sort(key=lambda s:(s.meta['stratum']==stratum,s.meta['biome_transition'],len(np.unique(s.tokens))),reverse=True)
        selected=candidates[:2]
        if index==4:
            homogeneous=[s for s in candidates if len(np.unique(s.tokens))==1]
            if homogeneous and selected:selected[-1]=homogeneous[0]
        for state in selected:
            state.meta['requested_stratum']=stratum
            if out.append(state,'Ordinary',None):actual.append(state)
    return actual


def dynamic(probe,out,category,attempt,rng):
    start=time.monotonic();candidates_examined=0;chosen=None;visit=None
    x,z,depth=HINTS[category]
    locations=[(x,z,64 if category=='D2' and attempt else depth)]
    if attempt:
        locations += [(x+64,z+64,128 if category=='D2' else 16),
                      (int(rng.integers(-2048,2049))//16*16,int(rng.integers(-2048,2049))//16*16,64)]
    for x,z,depth in locations:
        visit=probe.command('visit',x=x,z=z,depth=depth,
            target={'D1':'water','D2':'lava','D3':'fire','D4':'vegetation'}[category],
            layer='underground' if category=='D2' else 'surface')
        if visit['status']!='ready':continue
        found=probe.command('p0_discover',category=category,repair=bool(attempt))
        candidates_examined+=int(found['examined'])
        if found['candidates']:chosen=found['candidates'][0];break
    trial=dict(category=category,attempt=attempt,repair_reason=REASONS[category] if attempt else None,
               candidate_nodes_examined=candidates_examined,selected_candidate=chosen,
               planned_observation_seconds=(REPAIR_SECONDS if attempt else DURATIONS)[category],
               event_trajectory_id=f'{probe.seed}/{category}/attempt{attempt}',valid_clip=False)
    if not chosen:
        trial.update(status='no_natural_candidate',wall_seconds=time.monotonic()-start,observed_seconds=0)
        return trial,[],[]
    active=probe.command('p0_activate',position=chosen['pos'])
    if not active['all_blocks_active']:raise RuntimeError('Natural candidate region is not active')
    first,clock=probe.capture(active['origins'],visit['surface_y'],verify=True)
    events={origin:Event(s,category,probe.v) for origin,s in first.items()}
    ignition_mode=chosen['ignition_mode'];case_id=trial['event_trajectory_id']
    trial['ignition_mode']=ignition_mode;trial['ignitions']=0
    if category=='D3' and ignition_mode=='controlled_random':
        ignition=probe.command('p0_ignite',case_id=case_id,under=chosen['pos'],above=chosen['above'])
        trial['ignitions']=1;trial['ignition_result']=ignition
        if ignition['status']!='ignited':
            trial.update(status='ignition_failed',wall_seconds=time.monotonic()-start,observed_seconds=0)
            return trial,[],list(first.values())[:1]
    beginning=clock['sim_time'];length=trial['planned_observation_seconds']
    def native_path(event):
        return event.has_native_evolution(ignition_mode)
    while clock['sim_time']-beginning<length:
        frames,clock=probe.capture(active['origins'],visit['surface_y'])
        for origin,state in frames.items():
            if origin in events:events[origin].append(state)
        if any(len(e.distinct())>=8 and native_path(e) for e in events.values()):break
    summaries=[dict(chunk_origin=list(origin),**event.summary()) for origin,event in events.items()]
    eligible=[e for e in events.values() if native_path(e) and len(e.distinct())>=5]
    best=max(eligible or list(events.values()),key=lambda e:(len(e.distinct()),e.updates,int(probe.v.mask(e.records[0].tokens,category).sum())),default=None)
    selected=best.select() if best and native_path(best) else []
    event_id=case_id+'/'+(','.join(map(str,best.records[0].meta['chunk_origin'])) if best else 'none')
    for state in selected:
        if not out.append(state,category,event_id,ignition_mode):raise RuntimeError('Repeated token state within a standard clip')
    failure='no_native_growth' if category=='D4' and best and not native_path(best) else 'no_native_propagation' if category=='D3' and best and not native_path(best) else 'insufficient_distinct_states'
    trial.update(status='valid_clip' if selected else failure,valid_clip=bool(selected),
                 sampled_states=len(selected),event_id=event_id,selected_chunk_origin=best.records[0].meta['chunk_origin'] if selected else None,
                 observed_seconds=clock['sim_time']-beginning,
                 wall_seconds=time.monotonic()-start,chunks=summaries)
    diagnostics=best.distinct()[:8] if best else list(first.values())[:1]
    return trial,selected,diagnostics


def run(args):
    v=Vocabulary();out=Output(args.output,v);rng=np.random.default_rng(42)
    seed=int(rng.integers(1,2147483647)) if args.seed is None else args.seed
    results=dict(world_seed=seed,build=BUILD,status='running',trials=[],ordinary_count=0,
                 protocol='VoxWM_VAE_PILOT_DATA_PROTOCOL_2026-11-13.md',full_200k_started=False,gpu_training=False)
    start=time.monotonic();deadline=start+args.wall_seconds;visual={}
    try:
        with GTSession('procedural',seed,args.renderer,args.craftium_dir,scene_script=HERE.parent/'vae_pilot/scene.lua',
             image_size=(128,128),scene_assets=(HERE/'native.lua',),native_settings={'active_block_range':3,'vae_p0_script':'native.lua'}) as session:
            probe=P0Probe(session,deadline,v,seed)
            ordinary_states=ordinary(probe,out,rng);results['ordinary_count']=len(ordinary_states)
            visual['Ordinary']=(ordinary_states,True);failed=[]
            for category in TITLES:
                trial,selected,diagnostic=dynamic(probe,out,category,0,rng)
                results['trials'].append(trial);visual[category]=(selected or diagnostic,bool(selected))
                dump(out.root/'progress.json',results)
                print(category,trial['status'],trial.get('sampled_states',0),flush=True)
                if not selected:failed.append(category)
            for category in failed:
                trial,selected,diagnostic=dynamic(probe,out,category,1,rng)
                results['trials'].append(trial)
                if selected or diagnostic:visual[category]=(selected or diagnostic,bool(selected))
                dump(out.root/'progress.json',results)
                print(category,'repair',trial['status'],trial.get('sampled_states',0),flush=True)
            results.update(status='completed',verified_native_nodes=probe.verified_nodes,
                           native_chunks_read=probe.samples_read,missing_chunks=probe.missing_chunks,
                           client_reported_elapsed_seconds=session.elapsed)
    except BaseException as exc:
        results.update(status='inconclusive',error=f'{type(exc).__name__}: {exc}')
        if 'probe' in locals():results.update(verified_native_nodes=probe.verified_nodes,native_chunks_read=probe.samples_read,missing_chunks=probe.missing_chunks)
    finally:
        results['wall_seconds']=time.monotonic()-start;results['roundtrip_voxels_checked']=v.checked_voxels
        results['vocabulary_K']=len(v.entries);out.flush()
        from .visualize import save_visuals
        results['visualizations']=save_visuals(out.root,visual,v)
        from .results import summarize
        stats=summarize(out,results);dump(out.root/'statistics.json',stats)
        print('P0:',stats['verdict'],out.root/'VAE_P0_RESULTS.md',flush=True)
    return results


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--craftium-dir',type=Path,default=DEFAULT_CRAFTIUM)
    p.add_argument('--renderer',choices=('auto','window','software'),default='auto')
    p.add_argument('--seed',type=int)
    p.add_argument('--wall-seconds',type=float,default=720)
    a=p.parse_args(argv)
    if not 0<a.wall_seconds<=1200:p.error('P0 wall budget must be 1..1200 seconds')
    return a


def main(argv=None):
    result=run(parse_args(argv))
    if result['status']!='completed':raise SystemExit(2)
if __name__=='__main__':main()
