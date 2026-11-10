"""USER-RUN engine tests: generate the real original Test A or native seed-42 world."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('test_a','seed42'),required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--renderer',choices=('auto','window','software'),default='auto')
    p.add_argument('--craftium-dir',type=Path)
    p.add_argument('--max-frames',type=int,help='Optional short smoke; omit to generate the full Test A protocol')
    a=p.parse_args(argv)
    from voxwm_gt.cli import main as generate
    output=a.output or Path('outputs')/a.case
    command=['generate','--scene','test_a' if a.case=='test_a' else 'procedural',
             '--seed','42','--renderer',a.renderer,'--output',str(output)]
    if a.craftium_dir: command+=['--craftium-dir',str(a.craftium_dir)]
    if a.max_frames: command+=['--max-frames',str(a.max_frames)]
    generate(command)
    import av
    import numpy as np
    from voxwm_gt.storage import DeltaReader, read_json
    voxel=DeltaReader(output/'voxels.bin')
    with np.load(output/'track.npz',allow_pickle=False) as data:
        assert len(data['camera'])==voxel.frames and data['camera'].shape[1]==10
        assert np.isfinite(data['camera']).all()
        assert np.array_equal(data['frame_indices'],np.arange(voxel.frames))
    with av.open(str(output/'rgb.mp4')) as video:
        count=sum(1 for _ in video.decode(video=0))
    assert count==voxel.frames and voxel.shape==(48,48,48)
    voxel.frame(voxel.frames-1)
    record=read_json(output/'record.json')
    assert record['status']=='complete' and record['seed']==42
    if a.case=='seed42': assert record['mapgen']=='v7'
    print(f'PASS: {a.case}, {voxel.frames} real GT/video frames, saved to {output.resolve()}')


if __name__=='__main__': main()
