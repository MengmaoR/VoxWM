"""Native engine session for agent policies, manual play and GT recording."""
import copy
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .encoding import Encoder
from .native import close_env, parse_nodes, renderer_config
from .storage import read_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRAFTIUM = ROOT / 'third_party/craftium'
SCENES = Path(__file__).parent / 'scenes'


def prepare_template(destination, craftium_dir, scene, world_file=None):
    base = craftium_dir / 'craftium-envs' / ('openworld-creative' if scene=='procedural' else 'openworld-dataset')
    game = base / 'games/VoxeLibre'
    if not (game/'game.conf').is_file():
        raise FileNotFoundError('Craftium/VoxeLibre source is missing; run scripts/setup_gt.sh first')
    (destination/'games').mkdir(parents=True)
    (destination/'games/VoxeLibre').symlink_to(game.resolve(), target_is_directory=True)
    mods = destination/'mods'; (mods/'craftium_env').mkdir(parents=True)
    (mods/'voxel_api').symlink_to((base/'mods/voxel_api').resolve(), target_is_directory=True)
    shutil.copy2(base/'mods/craftium_env/mod.conf', mods/'craftium_env/mod.conf')
    shutil.copy2(SCENES/('test_a.lua' if scene=='test_a' else 'native_world.lua'), mods/'craftium_env/init.lua')
    world = destination/'worlds/world'; world.mkdir(parents=True)
    config = (base/'worlds/world/world.mt').read_text()
    if scene!='test_a': config=config.replace('creative_mode = false','creative_mode = true').replace('enable_damage = true','enable_damage = false')
    (world/'world.mt').write_text(config)
    if scene=='custom':
        if world_file is None: raise ValueError('custom requires --world scene.npz')
        with np.load(world_file,allow_pickle=False) as data:
            voxels=np.asarray(data['voxels'])
            names=np.asarray(data['node_names'])
            params=np.asarray(data['node_param2']) if 'node_param2' in data else np.zeros(len(names),int)
            origin=np.asarray(data['origin_engine'],int)
            spawn=np.asarray(data['spawn_engine'],float)
            look=np.asarray(data['look_degrees'],float) if 'look_degrees' in data else np.zeros(2)
        if voxels.ndim!=3 or voxels.dtype.kind not in 'iu' or voxels.min()<0 or voxels.max()>=len(names):
            raise ValueError('voxels must be a 3D integer array indexing node_names')
        if names.ndim!=1 or names.dtype.kind not in 'US' or params.shape!=names.shape or np.any((params<0)|(params>255)):
            raise ValueError('node_names must be strings; node_param2 must match and fit uint8')
        if origin.shape!=(3,) or spawn.shape!=(3,) or look.shape!=(2,):
            raise ValueError('origin_engine/spawn_engine must be [3]; look_degrees=[yaw,pitch]')
        payload=dict(shape=list(voxels.shape),origin=origin.tolist(),spawn=spawn.tolist(),look=look.tolist(),
                     node_names=names.astype(str).tolist(),node_param2=params.tolist(),voxels=voxels.ravel().tolist())
        (world/'custom_world.json').write_text(json.dumps(payload,allow_nan=False))


class GTSession:
    """reset is performed on entry; step accepts the four-group native discrete action.

    packet=(rgb,info) always comes from ONE engine observation; voxel_obs_center is
    supplied by get_voxel_data_cpp, never estimated from player position.
    """
    def __init__(self, scene='test_a', seed=42, renderer='auto', craftium_dir=DEFAULT_CRAFTIUM,
                 world_file=None, mapgen='v7', scene_script=None, native_settings=None, image_size=(640,360), scene_assets=()):
        self.scene,self.seed,self.renderer=scene,seed,renderer
        self.craftium_dir=Path(craftium_dir).resolve()
        self.world_file=Path(world_file).resolve() if world_file else None
        self.mapgen=mapgen
        self.scene_script=Path(scene_script).resolve() if scene_script else None
        self.native_settings=dict(native_settings or {})
        self.image_size=image_size
        self.scene_assets=tuple(Path(p) for p in scene_assets)
        self.env=None; self.work=None
        self.elapsed=0.

    def __enter__(self):
        import gymnasium as gym
        import craftium
        from craftium.craftium_env import CraftiumEnv
        from craftium.wrappers import NueToEnuVoxelObs
        from .actions import MultiDiscreteActionWrapper
        self.work=tempfile.TemporaryDirectory(prefix='voxwm-gt-')
        try:
            folder=Path(self.work.name)/'scene'
            prepare_template(folder,self.craftium_dir,self.scene,self.world_file)
            if self.scene_script: shutil.copy2(self.scene_script,folder/"mods/craftium_env/init.lua")
            for asset in self.scene_assets: shutil.copy2(asset,folder/"mods/craftium_env"/asset.name)
            mode,environment=renderer_config(self.renderer)
            name='Craftium/OpenWorldCreative-v0' if self.scene=='procedural' else 'Craftium/OpenWorldDataset-v0'
            kwargs=copy.deepcopy(gym.spec(name).kwargs)
            kwargs.update(env_dir=str(folder),minetest_dir=str(self.craftium_dir),
                          obs_width=self.image_size[0],obs_height=self.image_size[1],enable_voxel_obs=True,
                          voxel_obs_rx=24,voxel_obs_ry=24,voxel_obs_rz=24,
                          init_frames=200,fps_max=24,frameskip=1,pmul=1,sync_mode=True,
                          max_timesteps=None,seed=self.seed,run_dir_prefix=self.work.name,
                          offscreen_sdl=mode=='software')
            conf=dict(fov=72,creative_mode=True,mg_name=self.mapgen if self.scene=='procedural' else 'singlenode',
                      fixed_map_seed=self.seed,world_start_time=12000,viewing_range=30,enable_fog=True,
                      enable_particles=False,mcl_logging_mobs_spawn=False,
                      num_emerge_threads=1,max_block_generate_distance=8,max_block_send_distance=8,
                      emergequeue_limit_total=64,emergequeue_limit_diskonly=32,emergequeue_limit_generate=32)
            if self.scene!='test_a':
                conf.update(enable_damage=False,enable_pvp=False,view_bobbing_amount=0.,
                            starting_inventory_creative='mcl_core:dirt 99,mcl_core:stone 99,mcl_core:tree 99,mcl_core:water_source 99')
            if __import__('sys').platform=='darwin': conf['video_driver']='opengl3'
            conf.update(self.native_settings)
            kwargs['minetest_conf']=conf
            self.env=MultiDiscreteActionWrapper(NueToEnuVoxelObs(CraftiumEnv(**kwargs)))
            self.env.unwrapped.mt.proc_env=environment
            self.rgb,self.info=self.env.reset(seed=self.seed)
            self.world=Path(self.env.unwrapped.mt.run_dir)/'worlds/world'
            noop=np.zeros(4,dtype=np.int64)
            for _ in range(300):
                if (self.world/'gt_ready').is_file(): break
                self.step(noop)
            else: raise RuntimeError('Scene did not emerge; see native engine output')
            (self.world/'gt_arm').write_text('arm\n')
            for _ in range(24): self.step(noop)
            (self.world/'gt_start').write_text('start\n')
            for _ in range(120):
                self.step(noop)
                if (self.world/'gt_nodes.json').is_file(): break
            else: raise RuntimeError('Engine node dictionary was not published')
            self.encoder=Encoder(parse_nodes(read_json(self.world/'gt_nodes.json')))
            self.elapsed=0.
            return self
        except BaseException:
            self.close()
            raise

    def step(self,action):
        self.rgb,reward,self.terminated,self.truncated,self.info=self.env.step(action)
        self.elapsed+=float(self.info['mt_dtime'])
        return self.rgb,reward,self.terminated,self.truncated,self.info

    def close(self):
        try:
            if self.env is not None: close_env(self.env)
        finally:
            if self.work is not None: self.work.cleanup()
            self.env=None;self.work=None

    def __exit__(self,*exc): self.close()
