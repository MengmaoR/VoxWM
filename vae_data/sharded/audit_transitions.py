"""Read original VWT1 observations and report world-aligned transitions; never runs an engine."""
from pathlib import Path
from collections import Counter
import csv,json
import numpy as np
from .trace import TraceReader
from .protocol import Vocabulary,dump
from .transitions import classify,definition
from voxwm_gt.storage import DeltaReader

DIRECTIONS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

def neighbors(frame,vocabulary,world):
    q=np.asarray(world)-frame.meta['region_origin'];result=[]
    for d in DIRECTIONS:
        at=q+d
        if np.all(at>=0)and np.all(at<48):
            name,param=vocabulary.entries[int(frame.tokens[tuple(at)])]
            result.append(dict(world_xyz=(at+frame.meta['region_origin']).tolist(),name=name,param2=param))
    return result

def audit(trace,start,end,output):
    trace=Path(trace);reader=TraceReader(trace);v=Vocabulary(trace/'vocabulary.json');output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folder=trace.parents[2]/'viewer'/trace.parent.name/trace.name
    exported=DeltaReader(folder/'voxels.bin')if(folder/'voxels.bin').exists()else None
    viewer_vocab=Vocabulary(folder/'vocabulary.json')if(folder/'vocabulary.json').exists()else None
    old_labels=json.loads((folder/'classes.json').read_text())if(folder/'classes.json').exists()else None
    rows=[];intervals=[];previous=None;first=None;count=0;raw_equal=True;mapping_equal=True;source_history=[]
    sources=reader.meta.get('selection',{}).get('status',{}).get('intervention',{}).get('actual_actions',[])
    for index,current in enumerate(reader.frames()):
        # Decode EVERY voxel as its original native pair, not category-color labels.
        names,params=v.decode(current.tokens)
        if first is None:first=current
        if exported is not None:raw_equal&=np.array_equal(exported.frame(index),current.tokens)
        table=viewer_vocab.entries if viewer_vocab else[(c['name'],c['param2'])for c in old_labels]if old_labels else None
        if table is not None:
            actual_names=np.asarray([x[0]for x in table],object)[current.tokens]
            actual_params=np.asarray([x[1]for x in table],np.uint8)[current.tokens]
            mapping_equal&=np.array_equal(names,actual_names)and np.array_equal(params,actual_params)
        source_history.append(dict(frame=index,sim_time=current.time,states=[dict(world_xyz=list(map(int,a['position'])),node_state=v.entries[int(current.tokens[tuple(np.asarray(a['position'],int)-current.meta['region_origin'])])])for a in sources]))
        if current.meta['region_origin']!=first.meta['region_origin']or current.meta['center_origin']!=first.meta['center_origin']:raise ValueError('Region moved: compare through explicit world coordinates before auditing')
        if current.meta['rollout_id']!=first.meta['rollout_id']:raise ValueError('Mixed rollout identities')
        if previous is not None:
            if not(current.step>previous.step and current.time>previous.time):raise ValueError('Non-increasing native time/index')
            if previous.time<=end and current.time>=start:
                changed=np.argwhere(previous.tokens!=current.tokens)
                intervals.append(dict(before_frame=index-1,after_frame=index,before_time=previous.time,after_time=current.time,changed_voxels=len(changed)))
                for q in changed:
                    pos=(q+current.meta['region_origin']).tolist();before=v.entries[int(previous.tokens[tuple(q)])];after=v.entries[int(current.tokens[tuple(q)])]
                    events=[e for e in reader.meta.get('native_events',[])if e['position']==pos and previous.time<float(e['clock']['sim_time'])<=current.time]
                    center=np.asarray(current.meta['center_origin']);in_center=bool(np.all(np.asarray(pos)>=center)and np.all(np.asarray(pos)<center+16))
                    rows.append(dict(sim_time=current.time,world_xyz=pos,before_name=before[0],before_param2=before[1],after_name=after[0],after_param2=after[1],
                        transition_type=classify(before,after,bool(events)),before_frame=index-1,after_frame=index,before_time=previous.time,after_time=current.time,
                        region_local=q.tolist(),center_chunk_local=(np.asarray(pos)-center).tolist()if in_center else None,in_center=in_center,
                        before_neighbors=neighbors(previous,v,pos),after_neighbors=neighbors(current,v,pos),confirmed_cooling_events=events))
        previous=current;count+=1
    if not(first.time<=start<=end<=previous.time):raise ValueError(f'Requested interval not covered: actual trace {first.time:.6f}..{previous.time:.6f}s; no substitute simulation was run')
    header=('sim_time','world_xyz','before_name','before_param2','after_name','after_param2','transition_type')
    with(output/'transitions.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=header,lineterminator="\n");writer.writeheader()
        for row in rows:writer.writerow({key:row[key]for key in header})
    suspicious=[r for r in rows if r['transition_type']in('source_loss_requires_write_evidence','solid_to_liquid_requires_write_evidence')]
    result=dict(status='NEEDS_FURTHER_NATIVE_WRITE_TRACING'if suspicious else'NO_SOURCE_LOSS_OR_UNEXPLAINED_SOLID_TO_LIQUID_IN_OBSERVED_WINDOW',
        trace=str(trace),unique_type=reader.meta['unique_type'],attempt=trace.name,rollout_id=first.meta['rollout_id'],requested_interval=[start,end],
        observed_range=[first.time,previous.time],observed_frames=count,region_origin=first.meta['region_origin'],center_origin=first.meta['center_origin'],
        transition_counts=dict(Counter(r['transition_type']for r in rows)),transitions=rows,observation_intervals=intervals,source_history=source_history,
        checks=dict(raw_viewer_tensor_equal=bool(raw_equal)if exported is not None else None,
                    viewer_pair_mapping_equal=bool(mapping_equal)if table is not None else None,
                    world_origin_fixed=True,native_time_strictly_increasing=True,frame_indices_match=bool(exported.frames==count)if exported is not None else None),
        captured_cooling_events=reader.meta.get('native_events',[]),full_native_write_journal_available=False,
        limitations=['sim_time in the table is AFTER-observation time; each actual transition occurred within (before_time, after_time].',
         'Native trace records observed states and cooling writes, not all Lua/C++ node writes. Intermediate air after mushroom destruction is not observed.',
         'No engine regeneration, map alteration or GT/token modification. Absence of a cooling log is not proof of every unobserved write.'])
    dump(output/'audit.json',result)
    text=f"# Native transition audit — {reader.meta['unique_type']} / {trace.name}\n\n{result['status']}\n\n"
    text+=f"Original trace: `{trace}`. Rollout `{first.meta['rollout_id']}`. Requested {start}–{end}s; {len(rows)} observed voxel transitions.\n\n"
    text+='All times below are after-observation times. Exact write time is bounded by the corresponding before/after observations in `audit.json`.\n\n'
    text+='| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'
    for r in rows:text+='| '+' | '.join(str(r[key])for key in header)+' |\n'
    text+='\n## Evidence and limits\n\n'
    text+='- Both initialization source positions are tracked at every original observation in `audit.json`.\n'
    text+='- Raw VWT1 to ABV1 and (Node Name,param2) viewer mappings: '+str(result['checks'])+'.\n'
    text+='- Red mushroom has native `destroy_by_lava_flow=1` in `mcl_mushrooms/small.lua`; `mcl_core/functions.lua` permits removal beside/under feeding lava. This supports a legal destruction/refill explanation, but the trace does not identify the precise callback or contain an intermediate air observation.\n'
    text+='- Flowing-lava withdrawal is distinguished from source deletion. No transition is labelled normal merely because its new name contains lava.\n'
    text+='- Full node-write/map-update journal is unavailable; cooling-only evidence cannot exclude every other native write.\n'
    (output/'README.md').write_text(text)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',required=True,type=Path);p.add_argument('--start',required=True,type=float);p.add_argument('--end',required=True,type=float);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=audit(a.trace,a.start,a.end,a.output);print(json.dumps({k:r[k]for k in('status','transition_counts','checks')},indent=2))
