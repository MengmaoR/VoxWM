"""Canned array fixtures for orchestration/export only. This is NOT a fluid simulator."""
import numpy as np
from vae_p0.sampling import State
from vae_pilot.sampling import split_for_seed
from .tracking import PLANT_NAMES

class DryWorld:
    def __init__(self,scene,vocabulary,**kwargs):
        self.scene=scene;self.v=vocabulary;self.time=10.;self.step=200;self.samples=0;self.initialized=False
        self.origins=[(-16,0,0)];self.prepared=None
        self.air=self.v.add('air',0);self.stone=self.v.add('mcl_core:stone',0)
        names={'D1':'mcl_core:water_flowing','D2':'mcl_core:lava_flowing','D3':'mcl_fire:fire',
               'D4':PLANT_NAMES.get(scene['subtype'],'mcl_bamboo:bamboo')}
        self.target=self.v.add(names[scene['category']],0)
    def __enter__(self):return self
    def __exit__(self,*exc):pass
    def clock(self):return dict(sim_time=self.time,simulation_step=self.step,engine_us=int(self.time*1e6),wall_time_unix=0)
    def prepare(self):
        self.prepared=dict(status='ready',origins=[list(o) for o in self.origins],surface_y=0,
            boot=dict(time_origin='SYNTHETIC FIXTURE'),lifecycle={k:self.clock() for k in
                ('emerge_requested','emerge_completed','activation_requested','activation_completed')})
        return self.prepared
    def capture(self):
        tokens=np.full((16,16,16),self.air,np.uint16);tokens[:,:,0]=self.stone
        is_control=self.scene['setting']=='C';native=not is_control or self.initialized
        if native:
            n=1+min(self.samples,7);tokens[3,3,1:1+n]=self.target;self.samples+=1
        self.step+=1
        meta=dict(world_seed=self.scene['world_seed'],chunk_origin=list(self.origins[0]),chunk_index=[-1,0,0],
            timestamp=self.time,simulation_step=self.step,biome='SYNTHETIC',biome_group='SYNTHETIC',stratum='fixture',
            split=split_for_seed(self.scene['world_seed'],[80,10,10]),axes='ENU',data_source='synthetic_dry_run')
        return {self.origins[0]:State(tokens,np.zeros(tokens.shape,np.uint8),meta)},self.clock()
    def discover(self,budget,selection_seed):
        candidates=[]
        for i in range(20):
            candidates.append(dict(position=[-15+i%4*3,1+i//4*3,2],under=[-15+i%4*3,1+i//4*3,1],
                subtype=self.scene['subtype'],features=dict(biome='SYNTHETIC',terrain='fixture',context='pool_edge',
                    contexts=['pool_edge','vertical_drop','narrow_channel','obstacle','chunk_boundary']),stratum_key=f'fixture/{i%3}'))
        return dict(status='discovered',discovered=20,examined=20,eligible=20,rejected={},candidates=candidates)
    def recheck(self,candidates):return dict(can_continue_growth=True,eligible=len(candidates))
    def intervene(self,candidates,fallback=False):
        self.initialized=True;self.samples=0;frames,clock=self.capture()
        clock['interventions']=[dict(position=c['position'],success=True,clock=self.clock()) for c in candidates]
        return frames,clock
    def wait_until(self,target):self.time=target;self.step+=1
