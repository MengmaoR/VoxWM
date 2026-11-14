"""Small Zarr/Zstd arrays with Parquet metadata; no type/state model channels."""
from pathlib import Path
import numpy as np
from .codec import BUILD

class Output:
    def __init__(self,root,vocabulary):
        import zarr
        from numcodecs import Blosc
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.vocabulary=vocabulary;self.metadata=[]
        g=zarr.open_group(str(self.root/'snapshots.zarr'),mode='w')
        compressor=Blosc(cname='zstd',clevel=3,shuffle=Blosc.SHUFFLE)
        self.tokens=g.create_dataset('tokens',shape=(0,16,16,16),chunks=(8,16,16,16),dtype='<u2',compressor=compressor)
        self.light=g.create_dataset('light',shape=(0,16,16,16),chunks=(8,16,16,16),dtype='u1',compressor=compressor)
        g.attrs.update(dict(axes=['sample','east','north','up'],build=BUILD,combined_token=True,light_is_auxiliary=True))
        self.seen_context={}

    def append(self,state,category,event_id,ignition_mode='natural'):
        # Only bound repetitions within the same seed/chunk context, never globally.
        context=(state.meta['world_seed'],tuple(state.meta['chunk_origin']),event_id or 'ordinary')
        key=state.tokens.tobytes();seen=self.seen_context.setdefault(context,set())
        if key in seen:return False
        seen.add(key)
        n=len(self.metadata);self.tokens.resize((n+1,16,16,16));self.light.resize((n+1,16,16,16))
        self.tokens[n]=state.tokens;self.light[n]=state.light
        row=dict(snapshot_id=n,category=category,event_id=event_id,ignition_mode=ignition_mode,
                 change_progress=state.progress,**state.meta)
        self.metadata.append(row)
        return True

    def flush(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.vocabulary.save(self.root/'vocabulary.json')
        if self.metadata:pq.write_table(pa.Table.from_pylist(self.metadata),self.root/'metadata.parquet',compression='zstd')
        else:
            schema=pa.schema([('snapshot_id',pa.int64()),('category',pa.string()),('event_id',pa.string()),
                              ('world_seed',pa.int64()),('timestamp',pa.float64()),('simulation_step',pa.int64())])
            pq.write_table(pa.Table.from_pylist([],schema=schema),self.root/'metadata.parquet')
