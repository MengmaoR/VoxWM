"""Render real captured phases from a lossless rollout; no simulator is started."""
import argparse
import json
from pathlib import Path
from vae_p1.storage import Reader
from vae_p1.visualize import save_event
from .core import Vocabulary,Event


def phases(dataset,rollout,category,subtype='',origin=None):
    root=Path(dataset);reader=Reader(root/'trajectories'/rollout)
    vocabulary=Vocabulary(root/'vocabulary.json');groups={}
    for row in reader.rows:groups.setdefault(tuple(row['chunk_origin']),[]).append(row)
    if origin is None:
        def target_count(item):
            rows=item[1];row=next((r for r in rows if r['phase']=='intervention_before'),rows[0])
            state=reader.frame(row['record_id'])
            return int((vocabulary.plant_mask(state.tokens,subtype) if category=='D4' else vocabulary.mask(state.tokens,category)).sum())
        origin=max(groups.items(),key=target_count)[0]
    rows=groups[tuple(origin)];selected=[]
    for phase in ('intervention_before','intervention_after','intervention_settling','native_observation'):
        candidates=[r for r in rows if r['phase']==phase]
        if candidates:selected.append(candidates[-1] if phase in ('intervention_settling','native_observation') else candidates[0])
    if not selected:selected=[rows[0],rows[-1]]
    states=[reader.frame(r['record_id']) for r in selected]
    event=Event(states[0],category,vocabulary,subtype)
    return save_event(root,dict(scene_id=rollout,category=category,subtype=subtype,synthetic=False),event,states,suffix='_raw_phases')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--rollout',required=True)
    p.add_argument('--category',choices=('D1','D2','D3','D4'),required=True)
    p.add_argument('--subtype',choices=('bamboo','reeds','vine'),default='')
    p.add_argument('--origin',type=int,nargs=3)
    a=p.parse_args();print(phases(a.dataset,a.rollout,a.category,a.subtype,a.origin))

if __name__=='__main__':main()
