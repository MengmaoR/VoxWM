"""Identical recording for automatic policies and keyboard play."""
import shutil
import time
from pathlib import Path
import numpy as np
from .native import VideoWriter, camera_record
from .storage import DeltaWriter, write_json

class Recorder:
    def __init__(self,output,session):
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.session=session
        self.voxels=DeltaWriter(self.root/'voxels.bin')
        self.video=VideoWriter(self.root/'rgb.mp4')
        self.rows={k:[] for k in ('camera','action','native_action','player_position','player_yaw','player_pitch',
                                  'cam_pos','cam_dir','obs_voxel_center','origin','crop_origin','dt','time_seconds')}
        self.start=time.perf_counter()
        self.record=dict(scene=session.scene,seed=session.seed,mapgen=session.mapgen if session.scene=='procedural' else 'singlenode',
                         kind='engine_gt',status='running',fps=24,shape=[48,48,48],rgb_shape=[360,640,3],
                         camera_convention='center_minus_half',axes='ENU (engine x,z,y)',
                         voxel_capture='get_voxel_data_cpp',capture_version=3,
                         craftium_commit='ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49')
        write_json(self.root/'record.json',self.record)

    def append(self,action):
        s=self.session;info=s.info
        voxel=s.encoder(info['voxel_obs'])
        camera,origin=camera_record(info)
        if not self.rows['camera']: np.save(self.root/'initial_rgb.npy',s.rgb,allow_pickle=False)
        self.voxels.append(voxel);self.video.append(s.rgb)
        values=dict(camera=camera,action=s.env.multihot(action).astype(np.float32),native_action=action,
                    player_position=info['player_pos'],player_yaw=info['player_yaw'],player_pitch=info['player_pitch'],
                    cam_pos=info['cam_pos'],cam_dir=info['cam_dir'],obs_voxel_center=info['voxel_obs_center'],
                    origin=origin,crop_origin=origin,dt=info['mt_dtime'],time_seconds=s.elapsed)
        for key,value in values.items(): self.rows[key].append(np.array(value,copy=True))
        return voxel,origin

    def __enter__(self): return self

    def __exit__(self,typ,error,trace):
        self.voxels.close();self.video.close()
        s=self.session
        if self.rows['camera']:
            np.savez_compressed(self.root/'track.npz',**{k:np.asarray(v) for k,v in self.rows.items()},
                                frame_indices=np.arange(len(self.rows['camera']),dtype=np.int64))
        write_json(self.root/'nodes.json',s.encoder.nodes)
        write_json(self.root/'classdict.json',s.encoder.classdict)
        write_json(self.root/'classes.json',s.encoder.classes)
        self.record.update(status='failed' if error else 'complete',frames=len(self.rows['camera']),
                           duration_seconds=s.elapsed,generation_wall_seconds=time.perf_counter()-self.start)
        if error: self.record['error']=str(error)
        write_json(self.root/'record.json',self.record)
        run=Path(s.env.unwrapped.mt.run_dir)
        for log in ('stdout.txt','stderr.txt'):
            if (run/log).is_file(): shutil.copy2(run/log,self.root/('engine_'+log))
