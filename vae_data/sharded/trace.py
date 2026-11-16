"""Streaming lossless native-only debug observations; separate from five-state shard data."""
from pathlib import Path
import json,struct
import numpy as np
from .protocol import Frame,dump
from .storage import atomic_json

CELL=np.dtype([('index','<u4'),('token','<u2'),('light','u1')])
class Trace:
    def __init__(self,root,vocabulary,unique_type):
        self.root=Path(root);self.root.mkdir(parents=True);self.v=vocabulary;self.type=unique_type
        self.stream=open(self.root/'observations.bin','wb');self.stream.write(b'VWT1')
        self.rows=[];self.by_key={};self.previous=None;self.events=[];self.selection={};self.vsize=-1
    def observe(self,frame,stage,events=()):
        if frame.meta.get('data_source')!='native':raise ValueError('Debug Trace accepts observed native states only')
        if frame.key in self.by_key:
            row=self.rows[self.by_key[frame.key]]
            if stage not in row['stages']:row['stages'].append(stage)
            return row['index']
        old=self.previous;mask=np.ones(frame.tokens.size,bool)if old is None else((frame.tokens!=old.tokens)|(frame.light!=old.light)).ravel()
        indices=np.flatnonzero(mask);cells=np.empty(len(indices),CELL);cells['index']=indices
        cells['token']=frame.tokens.ravel()[indices];cells['light']=frame.light.ravel()[indices]
        offset=self.stream.tell();self.stream.write(struct.pack('<I',len(cells)));self.stream.write(cells.tobytes());self.stream.flush()
        changed=np.zeros(frame.tokens.shape,bool)if old is None else frame.tokens!=old.tokens
        center=int(changed[16:32,16:32,16:32].sum());total=int(changed.sum())
        row=dict(index=len(self.rows),frame_id=frame.key,offset=offset,changed_cells=len(cells),stages=[stage],
            center_changes=center,neighbor_changes=total-center,metadata=dict(frame.meta))
        self.by_key[frame.key]=row['index'];self.rows.append(row);self.events.extend(events)
        self.previous=frame
        self.checkpoint();return row['index']
    def checkpoint(self):
        if len(self.v.entries)!=self.vsize:self.v.save(self.root/'vocabulary.json');self.vsize=len(self.v.entries)
        atomic_json(self.root/'trace.json',dict(format='VWT1',unique_type=self.type,shape=[48]*3,source='real_native',frames=self.rows,
            native_events=self.events,selected_indices=self.selection.get('indices',[]),selection=self.selection))
    def planning(self,planner,selection):
        save_planning(self.root/'planning',planner,selection)
    def finish(self,collector=None,chain=None,status=None):
        if self.stream.closed:return
        if collector:
            for event in collector.events:
                for role in('pre','reaction','post'):
                    key=event.get(role)
                    if role=='pre'and event.get('pre_observation'):key=event['pre_observation']['frame_id']
                    if key in self.by_key:
                        row=self.rows[self.by_key[key]];row['stages'].append(role)
            for f in collector.registry.frames:
                if f.key in self.by_key:self.rows[self.by_key[f.key]]['metadata'].update(f.meta)
        if chain:self.selection=dict(indices=[self.by_key[f.key]for f in chain.frames],frame_ids=[f.key for f in chain.frames],events=chain.events)
        self.selection['status']=status;self.checkpoint();self.stream.close()
    def recover(self,world):
        result=getattr(world,'last_result',{})
        if result.get('chunks'):
            try:
                frame,_=world.frame(result);self.observe(frame,'native_failure_observation',result.get('reaction_events')or[])
            except Exception:pass

class TraceReader:
    def __init__(self,root):self.root=Path(root);self.meta=json.loads((self.root/'trace.json').read_text());self.rows=self.meta['frames']
    def frames(self):
        tokens=np.empty((48,)*3,np.uint16);light=np.empty((48,)*3,np.uint8)
        with open(self.root/'observations.bin','rb')as stream:
            if stream.read(4)!=b'VWT1':raise ValueError('Unknown native trace')
            for row in self.rows:
                stream.seek(row['offset']);count=struct.unpack('<I',stream.read(4))[0]
                cells=np.frombuffer(stream.read(count*CELL.itemsize),CELL)
                tokens.ravel()[cells['index']]=cells['token'];light.ravel()[cells['index']]=cells['light']
                yield Frame(tokens.copy(),light.copy(),dict(row['metadata']))


def save_planning(directory,planner,selection):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    for kind,result in planner.results.items():
        np.savez_compressed(directory/(kind+'.npz'),legal_positions=np.asarray(np.unravel_index(planner.geometry.legal[result['kind']],planner.geometry.shape)).T,
            stats=result['stats'],qualified=result['qualified'])
    selected=[]
    for i,detail in enumerate(selection.get('details',[])):
        grid=detail['estimated_arrival_seconds'];grid.astype('u1').tofile(directory/f'selected_{i}.bin')
        selected.append({k:v for k,v in detail.items()if k!='estimated_arrival_seconds'})
    summary={k:v for k,v in selection.items()if k!='details'};summary['selected_sources']=selected
    dump(directory/'summary.json',summary)
