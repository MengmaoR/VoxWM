"""Synthetic geometry ONLY; never exported as native GT or accepted dataset evidence."""
import numpy as np
from ..protocol import Frame,Vocabulary

def fixture(forest=False,large=False):
    v=Vocabulary();air=v.add('air',0);stone=v.add('mcl_core:stone',0);wood=v.add('mcl_core:tree',0)
    a=np.full((48,)*3,stone,np.uint16)
    if large:a[:,:,17:]=air
    else:a[17:29,18:28,18:28]=air
    if forest:
        a[20:27:2,20:27:2,18:26]=wood
    meta=dict(data_source='synthetic_fixture',center_origin=[0,0,0],region_origin=[-16,-16,-16],
        simulation_step=0,sim_time=0.,engine_us=0,rollout_id='geometry-fixture',world_seed=42,
        flammable_names=['mcl_core:tree'],floodable_names=[],falling_names=[],buildable_names=[],wet_names=[])
    return Frame(a,np.zeros(a.shape,np.uint8),meta),v
