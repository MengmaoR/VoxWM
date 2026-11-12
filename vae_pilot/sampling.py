"""Voxel-change progress sampling; no repeated-frame padding."""
from dataclasses import dataclass
import numpy as np

KINDS=('water','lava','fire','vegetation')

@dataclass
class State:
    nodes: np.ndarray
    param2: np.ndarray
    info: dict
    progress: int = 0

    def payload(self):
        return self.nodes.astype('<u2',copy=False).tobytes()+self.param2.astype('u1',copy=False).tobytes()


def changed_voxels(a,b):
    return int(np.count_nonzero((a.nodes!=b.nodes)|(a.param2!=b.param2)))


def biome_group(name,groups):
    matches=[(len(alias),group) for group,aliases in groups.items() for alias in aliases
             if name==alias or name.startswith(alias+'_')]
    return max(matches)[1] if matches else None


def split_for_seed(seed,percent):
    # Arithmetic partition only; every chunk/trajectory from one seed shares a split.
    slot=((int(seed)*1103515245+12345)&0x7fffffff)%100
    return 'train' if slot<percent[0] else ('validation' if slot<sum(percent[:2]) else 'test')


def class_masks(names,groups):
    masks={k:[] for k in KINDS}
    for name,g in zip(names,groups):
        masks['water'].append(g.get('water',0)>0)
        masks['lava'].append(g.get('lava',0)>0)
        masks['fire'].append(g.get('fire',0)>0)
        preferred='bamboo' in name or 'vine' in name
        growth=any(g.get(k,0)>0 for k in ('sapling','plant','flora','crop'))
        masks['vegetation'].append(preferred or growth)
    return {k:np.asarray(v,bool) for k,v in masks.items()}


def change_kinds(a,b,masks):
    changed=(a.nodes!=b.nodes)|(a.param2!=b.param2)
    return [kind for kind,mask in masks.items() if np.any(changed&(mask[a.nodes]|mask[b.nodes]))]


def distinct_progress(states):
    # Cumulative updates include A->B->A; only distinct node+param2 states can be selected.
    result=[];seen=set();previous=None;progress=0
    for state in states:
        if previous is not None: progress+=changed_voxels(previous,state)
        previous=state
        payload=state.payload()
        if payload in seen: continue
        seen.add(payload)
        result.append(State(state.nodes,state.param2,state.info,progress))
    return result,progress


def progress_sample(states,n=8):
    if len(states)<=n:return list(states)
    # Preserve first/last distinct states, select nearest unused cumulative-progress stages.
    targets=np.linspace(states[0].progress,states[-1].progress,n)
    selected={0,len(states)-1}
    for target in targets[1:-1]:
        candidates=[i for i in range(1,len(states)-1) if i not in selected]
        selected.add(min(candidates,key=lambda i:(abs(states[i].progress-target),i)))
    return [states[i] for i in sorted(selected)]


def clip_size(available,remaining):
    # With lengths 5..8, remaining 1..4 or 9 cannot be completed without padding.
    for n in range(min(8,available,remaining),4,-1):
        if remaining-n not in (1,2,3,4,9):return n
    return 0


def aligned_chunks(raw,center):
    # obs center is a native node centre. right-crop first; use actual centre, not player pos.
    first=np.rint(np.asarray(center)-24).astype(int)
    raw=np.asarray(raw)[:48,:48,:48]
    starts=((first+15)//16)*16
    for x in range(starts[0],first[0]+33,16):
        for y in range(starts[1],first[1]+33,16):
            for z in range(starts[2],first[2]+33,16):
                offset=np.array([x,y,z])-first
                chunk=raw[tuple(slice(v,v+16) for v in offset)]
                if chunk.shape==(16,16,16,2): yield (int(x),int(y),int(z)),chunk
