"""Standalone GT: prescribed Test A, procedural engine worlds, custom voxel maps."""
import argparse
from pathlib import Path
from .session import DEFAULT_CRAFTIUM


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('generate','play'):
        g=sub.add_parser(name)
        g.add_argument('--scene',choices=('test_a','procedural','custom'),default='test_a')
        g.add_argument('--seed',type=int,default=42)
        g.add_argument('--craftium-dir',type=Path,default=DEFAULT_CRAFTIUM)
        g.add_argument('--renderer',choices=('auto','window','software'),default='auto')
        g.add_argument('--mapgen',choices=('v7','valleys','carpathian','v6','flat'),default='v7')
        g.add_argument('--world',type=Path,help='Custom .npz voxel map with node_names and spawn_engine')
        g.add_argument('--output',type=Path,required=name=='generate',help='New directory; play records only when supplied')
    g=sub.choices['generate']
    g.add_argument('--max-frames',type=int,help='Default: Test A <=1200; procedural/custom 240')
    g.add_argument('--policy',choices=('auto','noop','rotate','random'),default='auto')
    g.add_argument('--turns',type=float,default=2.5)
    g.add_argument('--min-frames',type=int,default=384)
    g.add_argument('--quiet-frames',type=int,default=96)
    g.add_argument('--tail-frames',type=int,default=48)
    g.add_argument('--actions',type=Path,help='Optional [T,4] native integer action .npy; T transitions, recording has T+1 frames')
    a=p.parse_args(argv)
    if a.scene=='custom' and a.world is None: p.error('--scene custom requires --world')
    if a.scene!='custom' and a.world is not None: p.error('--world is only used by --scene custom')
    if a.command=='generate':
        if a.max_frames is None: a.max_frames=1200 if a.scene=='test_a' else 240
        if min(a.max_frames,a.min_frames,a.quiet_frames,a.tail_frames)<1: p.error('frame counts must be positive')
    return a


def generate(a):
    import numpy as np
    from .session import GTSession
    from .recording import Recorder
    from .policy import RotatePolicy
    from .native import water_signature
    if a.output.exists(): raise FileExistsError(f'{a.output} already exists; use a new directory')
    planned=np.load(a.actions,allow_pickle=False) if a.actions else None
    if planned is not None and (planned.ndim!=2 or planned.shape[1]!=4 or planned.dtype.kind not in 'iu'):
        raise ValueError('--actions must contain an integer [T,4] array')
    policy_name=('rotate' if a.scene=='test_a' else 'noop') if a.policy=='auto' else a.policy
    legacy_stop=a.scene=='test_a' and a.policy=='auto' and planned is None
    rng=np.random.default_rng(a.seed)
    with GTSession(a.scene,a.seed,a.renderer,a.craftium_dir,a.world,a.mapgen) as s:
        if planned is not None and ((planned<0).any() or (planned>=s.env.action_space.nvec).any()):
            raise ValueError('--actions contains an action outside the native action space')
        rotate=RotatePolicy(s.info,a.turns)
        action=np.zeros(4,dtype=np.int64);previous=None;quiet=0;initial=maximum=0
        limit=min(a.max_frames,len(planned)+1) if planned is not None else a.max_frames
        with Recorder(a.output,s) as r:
            r.record.update(policy='file' if planned is not None else policy_name,turns=a.turns if policy_name=='rotate' else None)
            for t in range(limit):
                voxel,origin=r.append(action)
                signature,water=water_signature(voxel,origin,s.encoder.classes)
                if t==0:
                    initial=water
                    if legacy_stop and water!=9: raise ValueError(f'Test A expects nine initial water voxels, got {water}')
                maximum=max(maximum,water)
                quiet=quiet+1 if signature==previous else 0;previous=signature
                if s.terminated or s.truncated: r.record['stop_reason']='native_engine_terminated';break
                if legacy_stop and rotate.done and t+1>=a.min_frames and maximum>initial and quiet>=a.quiet_frames+a.tail_frames:
                    r.record['stop_reason']='water_settled_with_redundancy';break
                if t+1==limit: r.record['stop_reason']='action_file_end' if planned is not None and limit==len(planned)+1 else 'frame_cap';break
                if planned is not None: action=planned[t]
                elif policy_name=='rotate': action=rotate.action(s.env,s.info)
                elif policy_name=='random': action=np.array([rng.integers(n) for n in s.env.action_space.nvec],np.int64)
                else: action=np.zeros(4,dtype=np.int64)
                s.step(action)
            r.record.update(water_initial_count=initial,water_max_count=maximum,water_quiet_frames=quiet,
                            rotation_degrees=rotate.rotation if policy_name=='rotate' else None)
    print('GT and RGB saved:',a.output.resolve())


def main(argv=None):
    a=parse_args(argv)
    if a.command=='generate': generate(a)
    else:
        from .play import play
        play(a)
