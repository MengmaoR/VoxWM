"""Within-clip distinct Combined Tokens sampled on target-related change progress."""
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    tokens:np.ndarray
    light:np.ndarray
    meta:dict
    progress:int=0

class Event:
    def __init__(self,initial,kind,vocabulary):
        self.kind=kind;self.v=vocabulary;self.previous=initial;self.records=[initial]
        self.updates=0;self.changed_positions=np.zeros((16,16,16),bool)
        self.observations=1;self.identical=0;self.light_only=0;self.native_additions=0

    def append(self,state):
        old=self.previous;self.previous=state;self.observations+=1
        delta=old.tokens!=state.tokens
        if not delta.any():
            self.identical+=1;self.light_only+=int(np.any(old.light!=state.light));return
        old_mask=self.v.mask(old.tokens,self.kind);new_mask=self.v.mask(state.tokens,self.kind)
        target=delta&(old_mask|new_mask)
        count=int(target.sum())
        if not count:return
        self.updates+=count;self.changed_positions|=target
        self.native_additions+=int(np.count_nonzero(new_mask&~old_mask))
        state.progress=self.updates;self.records.append(state)

    def has_native_evolution(self,ignition_mode="natural"):
        if self.kind=="D4":return self.native_additions>0
        if self.kind=="D3":return self.native_additions>(1 if ignition_mode=="controlled_random" else 0)
        return self.updates>0

    def distinct(self):
        # Do not purge repeated local structures across independent seeds/contexts.
        unique=[];seen=set()
        for s in self.records:
            key=s.tokens.tobytes()
            if key in seen:continue
            seen.add(key);unique.append(s)
        return unique

    def select(self):
        states=self.distinct()
        if len(states)<5:return []
        n=min(8,len(states))
        if len(states)==n:return states
        targets=np.linspace(0,states[-1].progress,n)
        selected={0,len(states)-1}
        for target in targets[1:-1]:
            selected.add(min((i for i in range(1,len(states)-1) if i not in selected),key=lambda i:abs(states[i].progress-target)))
        return [states[i] for i in sorted(selected)]

    def summary(self):
        selected=self.select()
        return dict(observations=self.observations,consecutive_identical=self.identical,light_only_changes=self.light_only,
            distinct_states=len(self.distinct()),sampled_states=len(selected),cumulative_target_changes=self.updates,
            unique_changed_positions=int(self.changed_positions.sum()),
            net_token_difference=int(np.count_nonzero(self.records[0].tokens!=self.previous.tokens)),
            native_target_additions=self.native_additions,selected_progress=[s.progress for s in selected],
            selected_timestamps=[s.meta['timestamp'] for s in selected],selected_steps=[s.meta['simulation_step'] for s in selected])
