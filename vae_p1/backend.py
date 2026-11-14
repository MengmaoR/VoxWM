"""Thin native adapter around P0 capture. Imported only by explicit --native."""
from pathlib import Path
import numpy as np
import subprocess
from vae_p0.run import P0Probe
from vae_p0.sampling import State
from vae_pilot.sampling import biome_group,split_for_seed
from voxwm_gt.session import GTSession

HERE=Path(__file__).parent

class Probe(P0Probe):
    def decode_capture(self,result):
        raw=(self.s.world/'p0_capture.bin').read_bytes();chunks=result['chunks']
        if len(raw)!=len(chunks)*4096*4:raise ValueError('Incomplete P0 native capture payload')
        dtype=np.dtype([('node','<u2'),('light','u1'),('param2','u1')])
        array=np.frombuffer(raw,dtype).reshape(len(chunks),16,16,16);states={}
        for cells,info in zip(array,chunks):
            origin=tuple(int(i) for i in info['origin'])
            if any(i%16 for i in origin) or info['missing'] or info['mismatches']:raise ValueError('Invalid fixed world Chunk capture')
            self.verified_nodes+=int(info['verified']);tokens=self.v.encode(cells['node'],cells['param2'],self.names)
            groups=[biome_group(n,self.groups) for n in info['biome_points']]
            meta=dict(world_seed=self.seed,chunk_origin=list(origin),chunk_index=[i//16 for i in origin],axes='ENU',
                biome=info['biome'],biome_group=groups[0],surface_biome=info['surface_biome'],stratum=info['stratum'],
                biome_transition=len(set(g for g in groups if g))>1,surface_height=info.get('surface_height'),height=info['height'],
                timestamp=result['sim_time'],simulation_step=int(result['simulation_step']),engine_us=int(result['engine_us']),
                split=split_for_seed(self.seed,[80,10,10]),build=self.runtime_build,data_source='native')
            states[origin]=State(tokens,cells['light'].copy(),meta)
        return states

class NativeWorld:
    def __init__(self,scene,vocabulary,deadline,renderer,craftium_dir):
        self.scene=scene;self.v=vocabulary;self.deadline=deadline;self.renderer=renderer;self.craftium_dir=craftium_dir

    def __enter__(self):
        self.session=GTSession('procedural',self.scene['world_seed'],self.renderer,self.craftium_dir,
            scene_script=HERE/'scene.lua',scene_assets=(HERE.parent/'vae_p0/native.lua',HERE/'eligibility.lua',HERE/'p1_native.lua'),
            native_settings={'active_block_range':3},image_size=(128,128))
        try:
            self.session.__enter__();self.probe=Probe(self.session,self.deadline,self.v,self.scene['world_seed'])
            builds={}
            for name,path in (('craftium',Path(self.craftium_dir)),('voxelibre',Path(self.craftium_dir)/'craftium-envs/openworld-creative/games/VoxeLibre')):
                r=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],text=True,capture_output=True)
                builds[name]=r.stdout.strip() if r.returncode==0 else None
            self.scene['source_build']=builds;self.probe.runtime_build=builds;self.v.runtime_build=builds
            return self
        except BaseException:
            self.session.close();raise

    def __exit__(self,*exc):self.session.__exit__(*exc)

    def prepare(self):
        self.prepared=self.probe.command('p1_prepare',site=self.scene['site']);return self.prepared

    def capture(self):
        result=self.probe.command('p0_capture',origins=self.prepared['origins'],surface_y=self.prepared['surface_y'])
        return self.probe.decode_capture(result),result

    def discover(self,budget,selection_seed):
        return self.probe.command('p1_candidates',category=self.scene['category'],subtype=self.scene['subtype'],
            setting=self.scene['setting'],interaction_context=self.scene['interaction_context'],budget=budget,selection_seed=selection_seed)

    def recheck(self,candidates):
        return self.probe.command('p1_recheck',category=self.scene['category'],subtype=self.scene['subtype'],
            setting='A',interaction_context=self.scene['interaction_context'],candidates=candidates)

    def intervene(self,candidates,fallback=False):
        result=self.probe.command('p1_intervene',scene_id=self.scene['scene_id'],category=self.scene['category'],
            subtype=self.scene['subtype'],setting='C',interaction_context=self.scene['interaction_context'],
            candidates=candidates,origins=self.prepared['origins'],fallback=fallback)
        return self.probe.decode_capture(result),result

    def wait_until(self,target):
        while True:
            result=self.probe.command('p1_clock')
            if result['sim_time']>=target:return
