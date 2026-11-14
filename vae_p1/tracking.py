"""Native-only event progress, conservative reaction evidence and fire-component merges."""
import numpy as np
import json
from pathlib import Path
from vae_p0.codec import Vocabulary as P0Vocabulary
from vae_p0.sampling import Event as P0Event

PLANT_NAMES={'reeds':'mcl_core:reeds','vine':'mcl_core:vine'}
PRODUCTS=('mcl_core:obsidian','mcl_core:cobble','mcl_core:stone')

class Vocabulary(P0Vocabulary):
    def save(self,path):
        super().save(path)
        if hasattr(self,'runtime_build'):
            data=json.loads(Path(path).read_text());data['configured_build_pins']=data['build'];data['build']=self.runtime_build
            Path(path).write_text(json.dumps(data,indent=2)+'\n')

    def mask(self,tokens,kind):
        if kind!='D4':return super().mask(tokens,kind)
        return self.plant_mask(tokens,'')

    def plant_mask(self,tokens,subtype):
        allowed=[]
        for name,_ in self.entries:
            bamboo=name in ('mcl_bamboo:bamboo','mcl_bamboo:bamboo_1','mcl_bamboo:bamboo_2','mcl_bamboo:bamboo_3','mcl_bamboo:bamboo_endcap')
            allowed.append(bamboo if subtype=='bamboo' else name==PLANT_NAMES[subtype] if subtype else bamboo or name in PLANT_NAMES.values())
        return np.asarray(allowed,bool)[tokens]

    def names(self,tokens):return np.asarray([n for n,_ in self.entries],object)[tokens]

class Event(P0Event):
    def __init__(self,initial,kind,vocabulary,subtype=''):
        super().__init__(initial,kind,vocabulary)
        self.subtype=subtype;self.first_native_time=None;self.first_growth_time=None;self.losses=0
        self.added=np.zeros((16,16,16),bool);self.reactions=0
        self._distinct=[initial];self._seen={initial.tokens.tobytes()}

    def mask(self,state):
        return self.v.plant_mask(state.tokens,self.subtype) if self.kind=='D4' else self.v.mask(state.tokens,self.kind)

    def append(self,state,reaction=None):
        old=self.previous;self.previous=state;self.observations+=1
        delta=old.tokens!=state.tokens
        if not delta.any():
            self.identical+=1;self.light_only+=int(np.any(old.light!=state.light));return
        old_mask,new_mask=self.mask(old),self.mask(state)
        reacted=np.zeros(delta.shape,bool) if reaction is None else reaction
        target=delta&(old_mask|new_mask|reacted)
        count=int(target.sum())
        if not count:return
        self.updates+=count;self.changed_positions|=target
        additions=new_mask&~old_mask
        self.native_additions+=int(additions.sum());self.added|=additions
        if additions.any() and self.first_growth_time is None:self.first_growth_time=state.meta['timestamp']
        self.losses+=int((old_mask&~new_mask).sum());self.reactions+=int(reacted.sum())
        if self.first_native_time is None:self.first_native_time=state.meta['timestamp']
        state.progress=self.updates;self.records.append(state)
        key=state.tokens.tobytes()
        if key not in self._seen:self._seen.add(key);self._distinct.append(state)

    def has_native_evolution(self,ignition_mode='natural'):
        # The first record is AFTER every initial intervention, never before it.
        if self.kind in ('D3','D4'):return bool(self.added.any())
        return self.updates>0

    def distinct(self):return self._distinct

    def result(self):
        summary=super().summary()
        if not self.has_native_evolution():summary.update(sampled_states=0,selected_progress=[],selected_timestamps=[],selected_steps=[])
        return dict(summary,first_native_time=self.first_native_time,first_growth_time=self.first_growth_time,native_added_positions=int(self.added.sum()),
            native_loss_updates=self.losses,confirmed_reactions=self.reactions,subtype=self.subtype)


def assemble(frames):
    origins=list(frames);low=np.min(origins,axis=0);high=np.max(origins,axis=0)+16
    grid=np.zeros(tuple(high-low),np.uint16);valid=np.zeros(grid.shape,bool)
    for origin,state in frames.items():
        at=np.asarray(origin)-low;sl=tuple(slice(int(i),int(i+16)) for i in at)
        grid[sl]=state.tokens;valid[sl]=True
    return low,grid,valid

def shifted(mask,offset):
    """At p, return mask[p+offset], with no wrapping across array edges."""
    result=np.zeros(mask.shape,mask.dtype);dest=[];source=[]
    for n,d in zip(mask.shape,offset):
        dest.append(slice(max(0,-d),min(n,n-d)));source.append(slice(max(0,d),min(n,n+d)))
    result[tuple(dest)]=mask[tuple(source)];return result

class Reactions:
    def __init__(self,vocabulary):self.v=vocabulary;self.contacts=[];self.products=[];self.seen=set();self.series=[]

    def observe(self,previous,current):
        low,old,valid=assemble(previous);low2,new,newvalid=assemble(current)
        if not np.array_equal(low,low2) or old.shape!=new.shape:raise ValueError('Reaction coordinates moved')
        old_names=self.v.names(old);new_names=self.v.names(new)
        water=self.v.mask(old,'D1')&valid;lava=self.v.mask(old,'D2')&valid
        flow=np.array(['flowing' in n for n,_ in self.v.entries],bool)[old]
        t=next(iter(current.values())).meta['timestamp'];step=next(iter(current.values())).meta['simulation_step']
        previous_time=next(iter(previous.values())).meta['timestamp']
        self.series.append(dict(timestamp=t,simulation_step=step,counts={name:int(np.count_nonzero(new_names==name)) for name in
            ('mcl_core:lava_source','mcl_core:lava_flowing','mcl_core:water_source','mcl_core:water_flowing',*PRODUCTS)}))
        confirmed=np.zeros(new.shape,bool)
        # Matches the pinned native Lava cooling ABM, not a generic stone detector.
        for offset in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            contact=lava&shifted(water,offset)
            for p in np.argwhere(contact):
                q=p+offset;world=tuple((p+low).tolist());neighbor=tuple((q+low).tolist());key=(world,neighbor)
                if key not in self.seen:
                    self.seen.add(key);self.contacts.append(dict(lava=list(world),water=list(neighbor),offset_enu=list(offset),
                        timestamp=previous_time,detection_sim_time=t,lava_node=str(old_names[tuple(p)]),water_node=str(old_names[tuple(q)])))
                if offset==(0,0,-1):
                    at=tuple(q);expected=PRODUCTS[2];before=old_names[at]
                else:
                    at=tuple(p);expected=PRODUCTS[1] if flow[at] else PRODUCTS[0];before=old_names[at]
                if newvalid[at] and new_names[at]==expected and old_names[at]!=expected:
                    if not confirmed[at]:
                        confirmed[at]=True;self.products.append(dict(position=(np.array(at)+low).tolist(),product=expected,
                            before=str(before),timestamp=t,simulation_step=step,transition_interval=[previous_time,t],contact=dict(lava=list(world),water=list(neighbor))))
        per_chunk={}
        for origin in current:
            at=np.asarray(origin)-low;sl=tuple(slice(int(i),int(i+16)) for i in at);per_chunk[origin]=confirmed[sl]
        return per_chunk

class FireMerges:
    def __init__(self,vocabulary,initial_positions):
        self.v=vocabulary;self.labels={tuple(p):{i} for i,p in enumerate(initial_positions)};self.merges=[];self.seen=set()

    def observe(self,frames):
        low,grid,valid=assemble(frames)
        points={tuple((p+low).tolist()) for p in np.argwhere(self.v.mask(grid,'D3')&valid)}
        remaining=set(points);next_labels={};t=next(iter(frames.values())).meta['timestamp']
        while remaining:
            start=remaining.pop();stack=[start];component={start};labels=set()
            while stack:
                p=stack.pop();labels.update(self.labels.get(p,set()))
                for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    q=tuple(a+b for a,b in zip(p,d));labels.update(self.labels.get(q,set()))
                    if q in remaining:remaining.remove(q);component.add(q);stack.append(q)
            for p in component:next_labels[p]=set(labels)
            key=tuple(sorted(labels))
            if len(labels)>1 and key not in self.seen:
                self.seen.add(key);self.merges.append(dict(timestamp=t,initial_fire_ids=list(key),component_size=len(component)))
        self.labels=next_labels
