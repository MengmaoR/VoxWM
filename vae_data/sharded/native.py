"""Minimal 48³ adapter reusing the verified P0 decoder and voxel-only engine overlay."""
from pathlib import Path
import json,subprocess,tempfile,uuid,time,resource,sys
import numpy as np
from vae_data.engine import World,overlay,SETTINGS
from .clock import TickProbe,verify_clock,PATCH_VERSION
from voxwm_gt.session import GTSession
from .protocol import Vocabulary,assemble,BIOMES,biome_group,split_for_seed,check_sources,SIMULATION_HZ,STATIC_CENTER_MIN,STATIC_CENTER_MAX

HERE=Path(__file__).parent

class Native(World):
    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='voxwm-shard-')
        root=overlay(self.source,Path(self.temp.name)/'engine')
        self.session=GTSession('procedural',self.seed,self.renderer,root,scene_script=HERE/'native.lua',
            scene_assets=(HERE.parents[1]/'vae_p0/native.lua',HERE/'scene_geometry.lua',HERE/'biome_search.lua'),native_settings=dict(SETTINGS),image_size=(32,32),fixed_step=True,voxel_observation=False,render_rgb=False)
        try:
            self.session.__enter__();self.probe=TickProbe(self.session,self.deadline,self.v,self.seed)
            self.build={name:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip() for name,path in
                (('craftium',self.source),('voxelibre',self.source/'craftium-envs/openworld-creative/games/VoxeLibre'))}
            self.probe.runtime_build=self.build
            repo=HERE.parents[1]
            self.collector_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            self.collector_dirty=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip())
            boot=json.loads((self.session.world/'pilot_boot.json').read_text());self.defs={n['name']:n for n in boot['nodes']}
            self.terrain=[n['name']for n in boot['nodes']if n.get('terrain_solid')]
            self.scene_selection=None
            self.flammable=[n['name']for n in boot['nodes']if n['groups'].get('flammable',0)>0]
            self.floodable=[n['name']for n in boot['nodes']if n.get('floodable')]
            self.buildable=[n['name']for n in boot['nodes']if n.get('buildable_to')]
            self.wet=[n['name']for n in boot['nodes']if n['groups'].get('puts_out_fire',0)>0]
            self.falling=[n['name']for n in boot['nodes']if n['groups'].get('falling_node',0)>0]
            self.protected_indices=[]
            if float(boot['liquid_update'])!=1.0:raise ValueError('This protocol expects native liquid_update=1.0; no override applied')
            for kind,value in(('water',7),('lava',3)):
                if self.defs['mcl_core:'+kind+'_source']['liquid_range']!=value:raise ValueError('Native liquid_range differs from the fixed protocol')
            self.rollout=uuid.uuid4().hex;self.center=None;self._last_tick=None;self._performance=[];return self
        except BaseException:self.__exit__(None,None,None);raise
    def checked(self,result):
        self.last_result=result
        return super().checked(result)
    def locate(self,x,z):return self.checked(self.probe.command('locate',x=x,z=z))
    def find_biome(self,group,site):
        result=self.checked(self.probe.command('find_biome',biome_group=group,biome_groups=BIOMES,
            x=int(site[0]),z=int(site[1]),resolution=64,max_checks=16384))
        # Keep a small search summary, not an index of all queried positions.
        keys=('status','method','biome_group','origin','resolution','max_checks','checked_points','biome_queries','site','biome','query_y')
        return {key:result[key]for key in keys if key in result}
    def find_centers(self,site,height_type,pool_kind=''):
        return self.checked(self.probe.command('find_centers',x=int(site[0])//16*16,z=int(site[1])//16*16,height_type=height_type,pool_kind=pool_kind))
    def prepare_at(self,center,surface_y,static_window=False):
        raw=np.asarray(center)
        if raw.shape!=(3,)or raw.dtype.kind not in'iuf'or not np.all(np.isfinite(raw)&(raw==np.floor(raw))):raise ValueError('Sampling origin must be integer-valued ENU')
        origin=raw.astype(int)
        if np.any(origin[:2]%16)or(not static_window and origin[2]%16):raise ValueError('Only Static height may be offset from the engine grid')
        if static_window and not STATIC_CENTER_MIN<=origin[2]+8<=STATIC_CENTER_MAX:raise ValueError('Full Static region exceeds overworld bounds')
        self.scene_selection=None;self._last_tick=None;self.static_window=static_window
        self.center=origin.tolist();self.surface_y=surface_y;self.rollout=uuid.uuid4().hex
        result=self.checked(self.probe.command('prepare',center=self.center,surface_y=surface_y,static_window=static_window))
        if result['status']!='prepared':raise RuntimeError(result['status'])
        self.protected_indices=result.get('protected_source_indices')or[]
        return self.frame(result)
    def frame(self,result):
        started=time.perf_counter();verify_clock(result,self.session.info)
        chunks=self.decode(result);central=chunks[tuple(self.center)];meta=dict(central.meta)
        meta.update(simulation_hz=SIMULATION_HZ,capture_method=result['capture_method'],pixel_capture=False,frame_index=result.get('frame_index'),relative_sim_time=result.get('relative_sim_time'),
            absolute_sim_time=int(result['simulation_step'])/SIMULATION_HZ,native_simulation_step=int(result['simulation_step']),
            native_clock=dict(result['native_clock'],client_completed_tick=self.session.info['native_client_tick'],client_dt=self.session.info['native_client_dt'],lua_tick=result['lua_tick']),
            sim_time=int(result['simulation_step'])/SIMULATION_HZ,simulation_step=int(result['simulation_step']),engine_us=int(result['engine_us']),
            rollout_id=self.rollout,world_seed=self.seed,center_origin=self.center,region_origin=(np.asarray(self.center)-16).tolist(),
            native_provenance=dict(engine_build=self.build,settings=dict(SETTINGS,voxwm_fixed24=True,voxwm_skip_rgb=True,frameskip=1,sync_env_mode=True,vsync=False),mapgen='v7',voxel_only=result['audit'],liquid_update=1.0,collector_revision=self.collector_revision,collector_worktree_dirty=self.collector_dirty,timing_patch=PATCH_VERSION),
            scene_selection=self.scene_selection,terrain_names=self.terrain,protected_source_indices=self.protected_indices,buildable_names=self.buildable,wet_names=self.wet,falling_names=self.falling,
            sampling_window_kind='static'if getattr(self,'static_window',False)else'dynamic',
            region_capture_version=result.get('region_capture_version'),capture_reference_equal=result.get('capture_reference_equal'),engine_mapblock_count=result.get('engine_mapblock_count'),
            sample_surface_y=self.surface_y,flammable_names=self.flammable,floodable_names=self.floodable,biome_group=biome_group(central.meta['surface_biome'],BIOMES),
            split=split_for_seed(self.seed,[80,10,10]),biome_points=result['chunks'][13]['biome_points'])
        if self.scene_selection:meta.update(height_type=self.scene_selection['height_type'],stratum=self.scene_selection['height_type'])
        meta['verified_native_voxels']=sum(int(info['verified'])for info in result['chunks'])
        meta['biome_groups_present']=sorted({group for info in result['chunks']for name in info['biome_points']if(group:=biome_group(name,BIOMES))})
        meta['biome_transition']=len(meta['biome_groups_present'])>1
        frame=assemble(chunks,self.center,meta)
        result['decode_seconds']=time.perf_counter()-started
        return frame,result
    def capture48(self):
        if self._last_tick is not None:raise ValueError('After initialization, use tick() for every capture')
        return self.frame(self.checked(self.probe.command('capture')))
    def place(self,actions):
        check_sources(actions,self.center)
        result=self.checked(self.probe.command('intervene',actions=actions))
        if result['status']=='intervention_outside_center_14':raise ValueError(result['status'])
        check_sources(result.get('actions')or[],self.center)
        frame,result=self.frame(result)
        if frame.meta['frame_index']!=0:raise ValueError('Intervention baseline must be relative tick zero')
        self._last_tick=frame.step;self._performance=[]
        return frame,result
    def tick(self):
        if self._last_tick is None:raise ValueError('Initialize sources before dynamic ticks')
        start=time.perf_counter();ipc=self.probe.communication_seconds
        result=self.checked(self.probe.command('advance'))
        verify_clock(result,self.session.info,self._last_tick)
        frame,result=self.frame(result);self._last_tick=frame.step
        self._performance.append(dict(total=time.perf_counter()-start,simulation=result['native_clock']['physics_us']/1e6,
            capture=result['capture_us']/1e6,communication=self.probe.communication_seconds-ipc,decode=result['decode_seconds']))
        return frame,result
    def performance(self):
        rows=self._performance;totals={k:sum(r[k]for r in rows)for k in('total','simulation','capture','communication','decode')}
        n=len(rows);wall=totals['total']
        return dict(simulation_hz=SIMULATION_HZ,ticks=n,measured_tick_wall_seconds=wall,
            seconds_per_tick={k:v/n if n else None for k,v in totals.items()},
            simulation_ticks_per_wall_second=n/wall if wall else None,simulation_seconds_per_wall_second=n/SIMULATION_HZ/wall if wall else None,
            capture_to_simulation_ratio=totals['capture']/totals['simulation']if totals['simulation']else None,
            peak_python_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin'else 1024),
            scope='Per-tick wall includes client synchronization/blank packet, IPC and decode; excludes planning, setup and Trace persistence. Native capture is post-update and before completion release.')
    def wait(self,target):
        # Setup-only stability observation; production dynamics always use tick().
        while True:
            result=self.checked(self.probe.command('clock'))
            if int(result['simulation_step'])>=int(np.ceil(target*SIMULATION_HZ)):return result
    def inspect(self,seed):
        result=self.checked(self.probe.command('inspect',seed=int(seed)%2147483647))
        for key in('attached','suspended','fire'):result[key]=result.get(key)or[]
        return result
