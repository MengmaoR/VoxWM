"""Tiny synthetic logic checks only; never produces or claims native GT."""
import json
from pathlib import Path
import tempfile
import numpy as np
from .sampling import State,distinct_progress,progress_sample,aligned_chunks,split_for_seed,class_masks
from .storage import Dataset,iter_snapshots
from .collect import evolve_flags


def main():
    config=json.loads((Path(__file__).parent/'config.json').read_text())
    frames=[]
    for t in range(12):
        nodes=np.zeros((16,16,16),np.uint16);nodes.ravel()[:t]=1
        frames.append(State(nodes,np.zeros_like(nodes,dtype=np.uint8),dict(seed=42,origin_enu=[0,0,0],biome='Forest',biome_group='Forest',terrain='surface',height=0,sim_time=float(t))))
    unique,total=distinct_progress([frames[0],frames[0],*frames[1:],frames[0]])
    assert len(unique)==12 and total==22
    selected=progress_sample(unique,8)
    assert len(selected)==8 and selected[0].info['sim_time']==0 and selected[-1].info['sim_time']==11
    assert [s.info['sim_time'] for s in selected]==sorted(s.info['sim_time'] for s in selected)
    with tempfile.TemporaryDirectory() as folder:
        db=Dataset(folder,config)
        db.register_nodes([dict(id=0,name='air',groups={}),dict(id=1,name='test_water',groups={'water':1})])
        assert db.clip('synthetic-short','water',frames[:4],42,'natural')==0
        assert db.clip('synthetic-valid','water',frames,42,'natural')==8
        assert not db.insert(selected[0],'ordinary',None)
        assert db.counts()['water']==8
        db.close()
        rows=list(iter_snapshots(folder,split_for_seed(42,config['split_percent'])))
        assert len(rows)==8 and rows[0]['voxel_ids'].shape==(16,16,16)
    raw=np.zeros((49,49,49,2),np.uint16)
    assert len(list(aligned_chunks(raw,[24,24,24])))==27
    assert len(list(aligned_chunks(raw,[25,25,25])))==8
    names=['air','vine','fire'];groups=[{}, {'vine':1},{'fire':1}]
    masks=class_masks(names,groups)
    plant=frames[1];burned=State(plant.nodes*0,plant.param2,plant.info)
    assert not evolve_flags('vegetation',plant,burned,masks,names,groups)
    print('PASS: progress sampling, duplicate removal, 5-state minimum, split isolation, exact storage and crop alignment; burning is not plant growth. Synthetic only.')

if __name__=='__main__':main()
