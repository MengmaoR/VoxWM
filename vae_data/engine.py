"""Isolated runtime builtin overlay plus P0's verified native chunk decoder."""
import json
from pathlib import Path
import shutil
import tempfile
import subprocess
from vae_p1.backend import Probe
from voxwm_gt.session import GTSession

HERE=Path(__file__).parent
SETTINGS=dict(mobs_spawn=False,mobs_griefing=False,mobs_drop_items=False,mcl_doTileDrops=False,
    enable_particles=False,enable_fire=True,enable_damage=False,enable_pvp=False,active_block_range=3,
    max_forceloaded_blocks=64,debug_log_level='info')

def overlay(source,destination):
    source=Path(source).resolve();destination=Path(destination);destination.mkdir()
    for p in source.iterdir():
        if p.name!='builtin':(destination/p.name).symlink_to(p,target_is_directory=p.is_dir())
    b=destination/'builtin';shutil.copytree(source/'builtin',b)
    game_init=b/'game/init.lua'
    game_init.write_text(game_init.read_text()+"\ndofile(core.get_builtin_path()..'voxel_only.lua')\n")
    shutil.copy2(HERE/'native/voxel_only.lua',b/'voxel_only.lua')
    return destination

class World:
    def __init__(self,seed,vocabulary,deadline,craftium_dir,renderer='auto',region=None):
        self.seed=seed;self.v=vocabulary;self.deadline=deadline;self.source=Path(craftium_dir).resolve();self.renderer=renderer;self.region=region
        self.session=None
    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='voxwm-voxel-only-')
        root=overlay(self.source,Path(self.temp.name)/'engine');settings=dict(SETTINGS)
        if self.region:settings['vae_replay_region']=json.dumps(self.region)
        self.session=GTSession('procedural',self.seed,self.renderer,root,scene_script=HERE/'native/scene.lua',
            scene_assets=(HERE.parent/'vae_p0/native.lua',HERE/'native/plants.lua'),native_settings=settings,image_size=(128,128))
        try:
            self.session.__enter__();self.probe=Probe(self.session,self.deadline,self.v,self.seed)
            builds={}
            for k,p in (('craftium',self.source),('voxelibre',self.source/'craftium-envs/openworld-creative/games/VoxeLibre')):
                r=subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True);builds[k]=r.stdout.strip() if r.returncode==0 else None
            self.probe.runtime_build=builds;self.v.runtime_build=builds;self.build=builds
            self.probe.session_path=self.session.world
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*exc):
        if self.session:self.session.close()
        self.temp.cleanup()
    def verify_scene(self,scene):
        if self.build!=scene['build'] or SETTINGS!=scene['config']:
            raise ValueError('Indexed scene build/settings differ from the reconstruction runtime')
    def checked(self,result):
        audit=result.get('audit')
        if audit is None:raise RuntimeError('Voxel-only pre-mod guard not active')
        if not audit['compatible']:raise RuntimeError('Excluded entity/falling-node transition: '+json.dumps(audit))
        self.audit=audit;return result
    def decode(self,result,file='p0_capture.bin'):
        if file!='p0_capture.bin':shutil.copy2(self.session.world/file,self.session.world/'p0_capture.bin')
        return self.probe.decode_capture(result)
    def prepare(self,site=None):
        if self.region:
            result=json.loads((self.session.world/'bootstrap_capture.json').read_text())
            self.checked(result);self.prepared=result;return result
        result=self.checked(self.probe.command('prepare',site=site));self.prepared=result;return result
    def capture(self):
        result=self.checked(self.probe.command('capture'));return self.decode(result),result
    def inspect(self,seed):
        result=self.checked(self.probe.command('inspect',seed=seed))
        for key in ('fire','attached','suspended'):result[key]=result.get(key)or[]
        result['plants']={key:value or[] for key,value in result['plants'].items()}
        return result
    def intervene(self,actions):
        result=self.checked(self.probe.command('intervene',actions=actions));result['actions']=result.get('actions')or[]
        return self.decode(result),result
    def initial(self):
        result=self.prepared
        frames=self.decode(result,result.get('file','p0_capture.bin'))
        early=[]
        for row in (result.get('early')or{}).values():
            data=dict(row['result'],**row['clock'])
            if not data['chunks'][0]['missing']:early.append((self.decode(data,row['file']),data,'mapgen_callback'))
        trace=self.session.world/'early_trace.jsonl'
        if trace.exists():
            for line in trace.read_text().splitlines():
                r=json.loads(line);self.checked(r);early.append((self.decode(r,r['file']),r,'early_native'))
        return frames,result,sorted(early,key=lambda row:row[1]['sim_time'])
    def wait(self,target):
        while True:
            r=self.checked(self.probe.command('clock'))
            if r['sim_time']>=target:return r
    def entity_probe(self):return self.checked(self.probe.command('entity_probe'))
