"""Minimal 48³ adapter reusing the verified P0 decoder and voxel-only engine overlay."""
from pathlib import Path
import json,subprocess,tempfile,uuid
import numpy as np
from vae_data.engine import World,overlay,SETTINGS
from vae_p1.backend import Probe
from voxwm_gt.session import GTSession
from .protocol import Vocabulary,assemble,BIOMES,biome_group,split_for_seed

HERE=Path(__file__).parent

class Native(World):
    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='voxwm-shard-')
        root=overlay(self.source,Path(self.temp.name)/'engine')
        self.session=GTSession('procedural',self.seed,self.renderer,root,scene_script=HERE/'native.lua',
            scene_assets=(HERE.parents[1]/'vae_p0/native.lua',),native_settings=dict(SETTINGS),image_size=(128,128))
        try:
            self.session.__enter__();self.probe=Probe(self.session,self.deadline,self.v,self.seed)
            self.build={name:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip() for name,path in
                (('craftium',self.source),('voxelibre',self.source/'craftium-envs/openworld-creative/games/VoxeLibre'))}
            self.probe.runtime_build=self.build
            repo=HERE.parents[1]
            self.collector_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            self.collector_dirty=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip())
            boot=json.loads((self.session.world/'pilot_boot.json').read_text());self.defs={n['name']:n for n in boot['nodes']}
            self.flammable=[n['name']for n in boot['nodes']if n['groups'].get('flammable',0)>0]
            self.floodable=[n['name']for n in boot['nodes']if n.get('floodable')]
            if float(boot['liquid_update'])!=1.0:raise ValueError('This protocol expects native liquid_update=1.0; no override applied')
            for kind,value in(('water',7),('lava',3)):
                if self.defs['mcl_core:'+kind+'_source']['liquid_range']!=value:raise ValueError('Native liquid_range differs from the fixed protocol')
            self.rollout=uuid.uuid4().hex;self.center=None;return self
        except BaseException:self.__exit__(None,None,None);raise
    def locate(self,x,z):return self.checked(self.probe.command('locate',x=x,z=z))
    def prepare_at(self,center,surface_y):
        self.center=list(map(int,center));self.surface_y=surface_y;self.rollout=uuid.uuid4().hex
        result=self.checked(self.probe.command('prepare',center=self.center,surface_y=surface_y))
        if result['status']!='prepared':raise RuntimeError(result['status'])
        return self.frame(result)
    def frame(self,result):
        chunks=self.decode(result);central=chunks[tuple(self.center)];meta=dict(central.meta)
        meta.update(sim_time=result['sim_time'],simulation_step=int(result['simulation_step']),engine_us=int(result['engine_us']),
            rollout_id=self.rollout,world_seed=self.seed,center_origin=self.center,region_origin=(np.asarray(self.center)-16).tolist(),
            native_provenance=dict(engine_build=self.build,settings=SETTINGS,mapgen='v7',voxel_only=result['audit'],liquid_update=1.0,collector_revision=self.collector_revision,collector_worktree_dirty=self.collector_dirty),
            sample_surface_y=self.surface_y,flammable_names=self.flammable,floodable_names=self.floodable,biome_group=biome_group(central.meta['surface_biome'],BIOMES),
            split=split_for_seed(self.seed,[80,10,10]),biome_points=result['chunks'][13]['biome_points'])
        meta['verified_native_voxels']=sum(int(info['verified'])for info in result['chunks'])
        meta['biome_groups_present']=sorted({group for info in result['chunks']for name in info['biome_points']if(group:=biome_group(name,BIOMES))})
        meta['biome_transition']=len(meta['biome_groups_present'])>1
        frame=assemble(chunks,self.center,meta)
        return frame,result
    def capture48(self):return self.frame(self.checked(self.probe.command('capture')))
    def place(self,actions):return self.frame(self.checked(self.probe.command('intervene',actions=actions)))
    def inspect(self,seed):
        result=self.checked(self.probe.command('inspect',seed=int(seed)%2147483647))
        for key in('attached','suspended','fire'):result[key]=result.get(key)or[]
        return result
