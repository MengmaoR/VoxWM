"""Reuse P0 training arrays; preserve every captured token/light state as sparse raw deltas."""
import json
from pathlib import Path
import struct
import numpy as np
from vae_p0.storage import Output
from vae_p0.sampling import State

CELL=np.dtype([('index','<u2'),('token','<u2'),('light','u1')])

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class Trajectory:
    def __init__(self,root):
        self.root=Path(root);self.data=open(self.root/'raw_trajectory.bin','wb');self.data.write(b'VWP1')
        self.index=open(self.root/'raw_trajectory.jsonl','w');self.previous={};self.count=0

    def append(self,state,scene,phase):
        key=(scene['scene_id'],tuple(state.meta['chunk_origin']));old=self.previous.get(key)
        mask=np.ones(4096,bool) if old is None else (state.tokens.ravel()!=old[1].ravel())|(state.light.ravel()!=old[2].ravel())
        changes=np.flatnonzero(mask);cells=np.empty(len(changes),CELL)
        cells['index']=changes;cells['token']=state.tokens.ravel()[changes];cells['light']=state.light.ravel()[changes]
        offset=self.data.tell();self.data.write(struct.pack('<H',len(cells)));self.data.write(cells.tobytes())
        row=dict(record_id=self.count,scene_id=scene['scene_id'],instance_id=scene['instance_id'],setting=scene['setting'],
            category=scene['category'],subtype=scene['subtype'],data_source=state.meta.get('data_source','native'),world_seed=scene['world_seed'],chunk_origin=list(key[1]),
            timestamp=state.meta['timestamp'],simulation_step=state.meta['simulation_step'],phase=phase,
            offset=offset,changed_cells=len(cells),previous_record=old[0] if old else None)
        self.index.write(json.dumps(row,separators=(',',':'))+'\n')
        self.previous[key]=(self.count,state.tokens.copy(),state.light.copy());self.count+=1
        state.meta['raw_record_id']=row['record_id'];return row['record_id']

    def flush(self):self.data.flush();self.index.flush()
    def close(self):self.data.close();self.index.close()

class Reader:
    def __init__(self,root):
        self.root=Path(root);self.rows=[json.loads(line) for line in (self.root/'raw_trajectory.jsonl').read_text().splitlines()]

    def frame(self,record_id):
        chain=[];row=self.rows[record_id]
        while True:
            chain.append(row)
            if row['previous_record'] is None:break
            row=self.rows[row['previous_record']]
        tokens=np.empty(4096,np.uint16);light=np.empty(4096,np.uint8)
        with open(self.root/'raw_trajectory.bin','rb') as stream:
            if stream.read(4)!=b'VWP1':raise ValueError('Not a P1 trajectory')
            for row in reversed(chain):
                stream.seek(row['offset']);n=struct.unpack('<H',stream.read(2))[0]
                cells=np.frombuffer(stream.read(n*CELL.itemsize),CELL)
                tokens[cells['index']]=cells['token'];light[cells['index']]=cells['light']
        return State(tokens.reshape(16,16,16),light.reshape(16,16,16),self.rows[record_id])

class Store:
    def __init__(self,root,vocabulary):
        self.standard=Output(root,vocabulary);self.root=self.standard.root;self.v=vocabulary
        self.aux=Output(self.root/'aux_short_event',vocabulary);self.raw=Trajectory(self.root)
        self.events=[]

    def save_event(self,event,scene):
        unique=event.distinct();native=event.has_native_evolution();sampled=event.select() if native else []
        short=unique if 2<=len(unique)<=4 and event.updates>0 else []
        event_id=scene['scene_id']+'/'+','.join(map(str,event.records[0].meta['chunk_origin']))
        for destination,states in ((self.standard,sampled),(self.aux,short)):
            for state in states:
                state.meta.update(is_initial_baseline=state is event.records[0],setting=scene['setting'],subtype=scene['subtype'],scene_id=scene['scene_id'],
                    native_growth_or_spread=native,event_type='growth' if scene['category']=='D4' and event.native_additions else
                    'decay' if scene['category']=='D4' else 'flow_and_reaction' if scene['category']=='D2' else 'native_dynamics')
                reserved={'snapshot_id','category','event_id','ignition_mode','change_progress'}
                saved=State(state.tokens,state.light,{k:v for k,v in state.meta.items() if k not in reserved},state.progress)
                destination.append(saved,scene['category'],event_id,'controlled_initialization' if scene['setting']=='C' else 'natural')
        row=dict(event.result(),scene_id=scene['scene_id'],event_id=event_id,setting=scene['setting'],category=scene['category'],
            subtype=scene['subtype'],world_seed=scene['world_seed'],chunk_origin=event.records[0].meta['chunk_origin'],
            standard_states=len(sampled),aux_short_states=len(short),native_evolution=native)
        self.events.append(row);return row,sampled or short or unique[:8]

    def flush(self):
        self.standard.flush();self.aux.flush();self.raw.flush();dump(self.root/'events.json',self.events)
    def close(self):self.flush();self.raw.close()
