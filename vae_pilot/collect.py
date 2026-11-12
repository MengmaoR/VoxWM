"""Bounded native-world pilot; --full is a separate, manual action."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from voxwm_gt.session import GTSession, DEFAULT_CRAFTIUM
from .sampling import KINDS,State,aligned_chunks,biome_group,class_masks,change_kinds
from .storage import Dataset,js
from .report import report

HERE=Path(__file__).parent

class Probe:
    def __init__(self,session,deadline):self.s=session;self.seq=0;self.deadline=deadline
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Pilot wall-time budget reached')
    def step(self):
        self.check();self.s.step(np.zeros(4,np.int64))
        if self.s.terminated or self.s.truncated:raise RuntimeError('Native engine terminated')
    def command(self,kind,**fields):
        self.check();self.seq+=1
        data=dict(request_id=self.seq,kind=kind,**fields)
        # Atomic command handover, not a checksum/provenance layer.
        path=self.s.world/'pilot_command.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(js(data));tmp.replace(path)
        for _ in range(2400):
            self.step();response=self.s.world/'pilot_response.json'
            if response.is_file():
                result=json.loads(response.read_text())
                if result['request_id']==self.seq:return result
        raise RuntimeError(f'Native {kind} request did not complete')
    def sample(self,seed,site,lookup,groups):
        chunks=list(aligned_chunks(self.s.info['voxel_obs'],self.s.info['voxel_obs_center']))
        response=self.command('sample',origins=[list(o) for o,_ in chunks])
        current=dict(aligned_chunks(self.s.info['voxel_obs'],self.s.info['voxel_obs_center']))
        requested={o for o,_ in chunks}
        # The first post-teleport observation can still use the previous C++ crop.
        # Keep only the SAME world chunks present in both request metadata and observation.
        current={o:raw for o,raw in current.items() if o in requested}
        states={}
        for origin,raw in current.items():
            # Ignore/unloaded cells are missing observations, never Air.
            nodes=lookup[raw[...,0].astype(int)]
            if np.any(nodes==self.ignore_id):continue
            meta=response['chunks'][','.join(map(str,origin))]
            meta.update(seed=seed,site=site,origin_enu=list(origin),chunk_index_enu=[x//16 for x in origin],
                        sim_time=response['sim_time'],game_time=response['game_time'],
                        engine_us=response['engine_us'],wall_time_unix=response['wall_time_unix'],sample_seq=self.seq)
            meta['biome_group']=biome_group(meta['surface_biome'],groups)
            states[origin]=State(nodes.copy(),raw[...,1].astype(np.uint8),meta)
        return states,response


def evolve_flags(kind,a,b,masks,names,groups):
    old,new=masks[kind][a.nodes],masks[kind][b.nodes]
    if kind=='fire':return bool(np.any(new&~old))
    if kind=='vegetation':
        stage=(a.nodes!=b.nodes)&old&new
        sapling=np.array([g.get('sapling',0)>0 for g in groups],bool)
        grown_tree=np.array([g.get('tree',0)>0 or g.get('leaves',0)>0 for g in groups],bool)
        return bool(np.any(new&~old) or np.any(stage) or np.any(sapling[a.nodes]&grown_tree[b.nodes]))
    flowing=np.array(['flowing' in n for n in names],bool)
    changed=(a.nodes!=b.nodes)|(a.param2!=b.param2)
    return bool(np.any(changed&((new&~old)|(old&new)|flowing[a.nodes]|flowing[b.nodes])))


def gather_site(probe,db,seed,site,target,config,rng,snapshot_cap):
    started=time.monotonic();bounds=config['horizontal_range']
    x,z=(int(v)//16*16 for v in rng.integers(-bounds,bounds+1,size=2))
    layer=('surface','surface','subsurface','underground','canopy')[site%5]
    if target=='lava':layer='underground'
    elif target in ('water','fire','vegetation'):layer='surface'
    reply=probe.command('visit',x=x,z=z,layer=layer,target=target,depth=int(rng.choice([32,64,128,512,1024])))
    if reply['status']!='ready':
        info=dict(status=reply['status'],target=target,layer=layer,seconds=0,chunk_trajectories=0)
        db.site(seed,site,info);return info
    lookup=probe.lookup;masks=class_masks(db.names,db.groups)
    previous,response=probe.sample(seed,site,lookup,config['biome_groups'])
    begin=response['sim_time'];buffers={};serial={};changed_seen=set();presence=set();ignited=False;ignition=None;ignition_attempted=False
    latest=previous.copy();observed=set(previous)
    for origin,state in previous.items():
        for kind in KINDS:
            if masks[kind][state.nodes].any():presence.add(kind)
    info=dict(status='complete',target=target,layer=layer,x=x,z=z,biome=reply['biome'],surface_y=reply['surface_y'],
              ignited=False,natural_fire_seen=False,chunk_trajectories=0,seconds=0)

    def flush(key):
        if key not in buffers:return
        rows,valid,source=buffers.pop(key);origin,kind=key
        index=serial.get(key,0);serial[key]=index+1
        trajectory=f'{seed}/{site}/{origin[0]},{origin[1]},{origin[2]}/{kind}'
        sources=sorted({name for state in rows for name in np.asarray(db.names)[np.unique(state.nodes)]
                        if kind=='vegetation' and ('bamboo' in name or 'vine' in name)})
        db.clip(trajectory+f'/{index}',kind,rows,seed,source,
                dict(trajectory_id=trajectory,native_evolution_observed=valid,preferred_vegetation_nodes=sources))
    try:
        while response['sim_time']-begin<config['site_seconds']:
            probe.step()
            if probe.s.elapsed < probe.next_sample:continue
            probe.next_sample=probe.s.elapsed+config['sample_seconds']
            current,response=probe.sample(seed,site,lookup,config['biome_groups']);latest.update(current);observed.update(current)
            for origin,b in current.items():
                for kind in KINDS:
                    if masks[kind][b.nodes].any():presence.add(kind)
                if origin not in previous:continue
                a=previous[origin]
                for kind in change_kinds(a,b,masks):
                    # Burning vegetation after a hand ignition is not natural vegetation growth.
                    if ignited and kind!='fire':continue
                    if kind in ('water','lava') and not evolve_flags(kind,a,b,masks,db.names,db.groups):continue
                    changed_seen.add(kind)
                    if kind=='fire' and not ignited:info['natural_fire_seen']=True
                    key=(origin,kind)
                    valid=evolve_flags(kind,a,b,masks,db.names,db.groups)
                    if key not in buffers:buffers[key]=([a,b],valid,'ignited' if kind=='fire' and ignited else 'natural')
                    else:
                        rows,seen,source=buffers[key];rows.append(b);buffers[key]=(rows,seen or valid,source)
                    if len(buffers[key][0])>=64:flush(key)
            previous=current
            if config['allow_ignition'] and db.remaining('fire')>0 and target=='fire' and not ignition_attempted and not info['natural_fire_seen'] and 'fire' not in presence and response['sim_time']-begin>=config['fire_ignite_after_seconds']:
                ignition_attempted=True
                fire=probe.command('ignite')
                if fire['status']=='ignited':
                    ignited=True;ignition=fire;info['ignited']=True
                    # The hand-placed ignition is not counted as native propagation.
                    previous,response=probe.sample(seed,site,lookup,config['biome_groups'])
                    latest.update(previous)
                else:info['ignition_result']=fire['status']
            if snapshot_cap and sum(db.counts().values())>=snapshot_cap:break
    except BaseException:
        info['status']='interrupted'
        raise
    finally:
        for key in list(buffers):flush(key)
        # Once per distinct observed fixed chunk; ordinary retains all non-target environment contents.
        for origin,state in latest.items():
            if snapshot_cap and sum(db.counts().values())>=snapshot_cap:break
            db.ordinary(state)
        info.update(seconds=response['sim_time']-begin,wall_seconds=time.monotonic()-started,
                    chunk_trajectories=len(observed),presence=sorted(presence),changed=sorted(changed_seen),
                    ignition=ignition)
        db.site(seed,site,info)
    return info


def collect(args):
    config=json.loads(args.config.read_text())
    for key in ('site_seconds','sites_per_world','sample_seconds','fire_ignite_after_seconds'):
        value=getattr(args,key,None)
        if value is not None:config[key]=value
    if args.no_ignition:config['allow_ignition']=False
    expected={'ordinary':100000,'water':25000,'lava':25000,'fire':25000,'vegetation':25000}
    if config['quotas']!=expected or len(config['biome_groups'])!=25:raise ValueError('Expected original 200K quotas and 25 biome groups')
    if (config['chunk_size'],config['clip_states'],config['min_clip_states'])!=(16,8,5):raise ValueError('Expected 16-cubed chunks and 5..8-state clips')
    if args.full:
        if not args.pilot_report:raise ValueError('--full requires the report from a successful native pilot')
        pilot=json.loads(args.pilot_report.read_text())
        if not pilot.get('collector_ok'):raise ValueError('Pilot has not demonstrated a working native collector')
    elif args.max_worlds>4:raise ValueError('Pilot is limited to <=4 worlds; full collection is a separate manual command')
    root=args.output
    if (root/'dataset.sqlite').exists() and not args.resume:raise FileExistsError('Output already has data; use --resume or a new output')
    db=Dataset(root,config);db.limit=None if args.full else args.snapshot_cap
    start=time.monotonic();deadline=start+args.wall_seconds
    rng=np.random.default_rng(config['seed']);status='completed';error=None
    try:
        seeds=set(row[0] for row in db.db.execute('SELECT seed FROM worlds'))
        for _ in range(args.max_worlds):
            if time.monotonic()>=deadline:status='wall_budget';break
            while True:
                seed=int(rng.integers(1,2147483647))
                if seed not in seeds:break
            seeds.add(seed);world_start=time.monotonic();info=dict(status='started',sites=0)
            try:
                with GTSession('procedural',seed,args.renderer,args.craftium_dir,scene_script=HERE/'scene.lua',
                               image_size=(64,64),native_settings={'active_block_range':3}) as session:
                    boot=json.loads((session.world/'pilot_boot.json').read_text())
                    lookup=db.register_nodes(boot['nodes'])
                    (root/'registered_biomes.json').write_text(js(boot['biomes'])+'\n')
                    probe=Probe(session,deadline);probe.lookup=lookup;probe.ignore_id=db.ids['ignore'];probe.next_sample=session.elapsed
                    for site in range(config['sites_per_world']):
                        probe.check()
                        gather_site(probe,db,seed,site,KINDS[site%4],config,rng,None if args.full else args.snapshot_cap)
                        info['sites']+=1
                        current=db.counts()
                        print(f'seed={seed} sites={info["sites"]} snapshots={current}',flush=True)
                        if all(current.get(k,0)>=n for k,n in config['quotas'].items()):break
                        if not args.full and sum(current.values())>=args.snapshot_cap:break
                    info.update(status='complete',engine_sim_seconds=session.elapsed)
            except BaseException:
                info['status']='interrupted'
                if 'session' in locals():info['engine_sim_seconds']=session.elapsed
                raise
            finally:
                info['wall_seconds']=time.monotonic()-world_start;db.world(seed,info)
            current=db.counts()
            if all(current.get(k,0)>=n for k,n in config['quotas'].items()):break
            if not args.full and sum(current.values())>=args.snapshot_cap:break
    except TimeoutError as exc:status='wall_budget';error=str(exc)
    except BaseException as exc:
        status='failed';error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        db.close()
        report(root,status=status,error=error,wall_seconds=time.monotonic()-start)
    return root


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--craftium-dir',type=Path,default=DEFAULT_CRAFTIUM)
    p.add_argument('--renderer',choices=('auto','window','software'),default='auto')
    p.add_argument('--max-worlds',type=int,default=1)
    p.add_argument('--sites-per-world',type=int)
    p.add_argument('--site-seconds',type=float)
    p.add_argument('--sample-seconds',type=float)
    p.add_argument('--fire-ignite-after-seconds',type=float)
    p.add_argument('--wall-seconds',type=float,default=180)
    p.add_argument('--snapshot-cap',type=int,default=256)
    p.add_argument('--no-ignition',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--full',action='store_true')
    p.add_argument('--pilot-report',type=Path)
    a=p.parse_args(argv)
    if min(a.max_worlds,a.wall_seconds,a.snapshot_cap)<=0:p.error('Budgets must be positive')
    for key in ('sites_per_world','site_seconds','sample_seconds','fire_ignite_after_seconds'):
        if getattr(a,key) is not None and getattr(a,key)<=0:p.error(key+' must be positive')
    return a


def main(argv=None):collect(parse_args(argv))
if __name__=='__main__':main()
