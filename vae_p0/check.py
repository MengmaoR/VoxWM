"""One tiny CPU-only representation/sampling/storage smoke, not a game simulation."""
import tempfile
from pathlib import Path
import numpy as np
from .codec import Vocabulary
from .sampling import Event,State
from .storage import Output

def main():
    v=Vocabulary();names={10:'air',11:'mcl_core:water_flowing'}
    nodes=np.full((16,16,16),10,np.uint16);params=np.zeros(nodes.shape,np.uint8)
    nodes[1,2,3]=11;params[1,2,3]=7
    tokens=v.encode(nodes,params,names);decoded,p=v.decode(tokens)
    assert decoded[1,2,3]==names[11] and p[1,2,3]==7
    changed=params.copy();changed[1,2,3]=8
    assert v.encode(nodes,changed,names)[1,2,3]!=tokens[1,2,3]
    for action in (lambda:v.encode(np.full(nodes.shape,999),params,names),lambda:v.decode(np.full(nodes.shape,65535,np.uint16))):
        try:action()
        except (ValueError,KeyError):pass
        else:raise AssertionError('Unknown pair/token was accepted')
    bounded=Vocabulary();bounded.limit=1;bounded.add('air',0)
    try:bounded.add('other',0)
    except OverflowError:pass
    else:raise AssertionError('Vocabulary overflow was accepted')
    light=np.zeros(tokens.shape,np.uint8)
    meta=dict(world_seed=42,chunk_origin=[-16,0,32],timestamp=0.,simulation_step=10,split='train')
    initial=State(tokens,light,meta);event=Event(initial,'D1',v)
    event.append(State(tokens.copy(),light+1,dict(meta,timestamp=1.,simulation_step=11)))
    assert len(event.records)==1 and event.light_only==1
    for i in range(1,10):
        t=tokens.copy();t[1,2,3+i]=v.add('mcl_core:water_flowing',i)
        event.append(State(t,light,dict(meta,timestamp=float(i+1),simulation_step=11+i)))
        if i==3:assert not event.select()
    selected=event.select();assert len(selected)==8
    assert len({s.tokens.tobytes() for s in selected})==8
    assert all(b.meta['simulation_step']>a.meta['simulation_step'] for a,b in zip(selected,selected[1:]))
    air=v.add('air',0);vine=v.add('mcl_core:vine',3)
    base=np.full(tokens.shape,air,np.uint16);base[0,0,:4]=vine
    declining=Event(State(base,light,meta),'D4',v)
    for i in range(4):
        frame=base.copy();frame[0,0,:i+1]=air
        declining.append(State(frame,light,dict(meta,timestamp=float(i+1),simulation_step=30+i)))
    assert len(declining.select())==5 and not declining.has_native_evolution()
    frame=frame.copy();frame[0,0,5]=vine
    declining.append(State(frame,light,dict(meta,timestamp=5.,simulation_step=35)))
    assert declining.has_native_evolution()
    fire=v.add('mcl_fire:fire',0);frame=np.full(tokens.shape,air,np.uint16)
    propagation=Event(State(frame.copy(),light,meta),'D3',v)
    frame[0,0,0]=fire;propagation.append(State(frame.copy(),light,meta))
    assert not propagation.has_native_evolution('controlled_random')
    frame[0,0,1]=fire;propagation.append(State(frame.copy(),light,meta))
    assert propagation.has_native_evolution('controlled_random')
    with tempfile.TemporaryDirectory() as tmp:
        output=Output(Path(tmp)/'data',v)
        for s in selected:assert output.append(s,'D1','one-event')
        assert not output.append(selected[0],'D1','one-event')
        assert output.append(selected[0],'D1','independent-event')
        output.flush()
        import pyarrow.parquet as pq
        assert pq.read_table(output.root/'metadata.parquet').num_rows==9
        assert output.tokens.shape==(9,16,16,16) and output.tokens.dtype==np.uint16
        assert output.light.dtype==np.uint8
        saved=Vocabulary(output.root/'vocabulary.json');assert saved.entries==v.entries
    print('PASS: Combined Token round trip / unknowns / overflow; target-progress distinct states; light-only exclusion; growth vs decay / ignition vs propagation; Zarr+Parquet.')

if __name__=='__main__':main()
