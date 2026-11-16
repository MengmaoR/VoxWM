"""At most ten center-distinct candidates; D4 roles share physical frame identities."""
import numpy as np
from .protocol import Chain,CONFIG,target_changes

class Registry:
    def __init__(self,anchor=None):
        self.anchor_key=anchor.key if anchor else None;self.frames=[];self.by_key={};self.signatures={}
    def add(self,frame,event_id=None,role=None):
        if frame.key==self.anchor_key:return None
        if frame.key in self.by_key:
            saved=self.by_key[frame.key]
            if not np.array_equal(saved.tokens,frame.tokens):raise ValueError('Conflicting states for one physical frame')
        else:
            signature=frame.center.tobytes()
            if signature in self.signatures:
                if role is None:return None
                saved=self.by_key[self.signatures[signature]]
            elif len(self.frames)>=10:return None
            else:
                saved=frame;self.frames.append(frame);self.by_key[frame.key]=frame;self.signatures[signature]=frame.key
        if role:
            roles=saved.meta.setdefault('event_roles',[]);item=dict(event_id=event_id,role=role)
            if item not in roles:roles.append(item)
        return saved.key

class Collector:
    def __init__(self,anchor,category,vocabulary,max_seconds=None):
        self.anchor=anchor;self.category=category;self.v=vocabulary;self.previous=anchor
        self.registry=Registry(anchor if category=='D4'else None)
        if category!='D4':self.registry.add(anchor)
        self.started=anchor.time;self.last_change=anchor.time;self.changes=0;self.transition_steps=0;self.events=[]
        self.max_seconds=max_seconds if max_seconds is not None else CONFIG['max_observation_seconds'][category]
        self.reason=None
    def observe(self,frame,reactions=()):
        if frame.step<=self.previous.step or frame.time<=self.previous.time:raise ValueError('Native time/step must increase')
        changed=target_changes(self.previous,frame,self.v,self.category)
        if changed:self.last_change=frame.time;self.changes+=changed;self.transition_steps+=1
        if self.category!='D4':
            if changed:self.registry.add(frame)
        else:
            if changed:
                for event in self.events:
                    if event.get('post')is None and frame.step>event['reaction_step']:
                        key=self.registry.add(frame,event['id'],'post')
                        if key and self.registry.by_key[key].step>event['reaction_step']:event['post']=key
            confirmed=[event for event in reactions if event.get('confirmed') and self._inside(event['position'])]
            if confirmed:
                event_id=self.anchor.meta['rollout_id']+'/reaction_'+str(len(self.events))
                pre=self.registry.add(self.previous,event_id,'pre');reaction=self.registry.add(frame,event_id,'reaction')
                self.events.append(dict(id=event_id,pre=pre,reaction=reaction,post=None,reaction_step=frame.step,
                    transitions=confirmed,center_origin=frame.meta['center_origin'],
                    pre_observation=dict(frame_id=self.previous.key,sim_time=self.previous.time,simulation_step=self.previous.step,engine_us=self.previous.meta['engine_us'],
                        equal_center_to_reference=bool(pre and np.array_equal(self.previous.center,self.registry.by_key[pre].center)))))
        self.previous=frame
        if len(self.registry.frames)>=10:self.reason='candidate_limit'
        elif frame.time-self.last_change>=2.0:self.reason='center_no_change_2s'
        elif frame.time-self.started>=self.max_seconds:self.reason='maximum_sim_time'
        return self.reason is not None
    def _inside(self,position):
        low=np.asarray(self.anchor.meta['center_origin']);p=np.asarray(position)
        return bool(np.all(p>=low)and np.all(p<low+16))
    def windows(self):
        frames=sorted(self.registry.frames,key=lambda frame:frame.step);windows=[]
        for index,frame in enumerate(frames):frame.meta.update(candidate_index=index,candidate_count=len(frames))
        for i in range(len(frames)-4):
            window=frames[i:i+5];keys={f.key for f in window}
            if self.category=='D4':
                valid=[e for e in self.events if e['pre']in keys and e['reaction']in keys and e['post']in keys]
                by_key={f.key:f for f in window}
                valid=[e for e in valid if by_key[e['pre']].time<by_key[e['reaction']].time<by_key[e['post']].time]
                if not valid:continue
            windows.append(window)
        return windows
    def select(self,rng,intervention=None,subtype=''):
        from .validate import validate_chain
        valid=[]
        for chosen in self.windows():
            keys={f.key for f in chosen}
            events=[e for e in self.events if any(e.get(role)in keys for role in('pre','reaction','post'))]
            candidate=Chain(self.category,chosen,self.anchor if self.category=='D4'else None,events,intervention or{},subtype)
            try:validate_chain(candidate,self.v,require_native=self.anchor.meta.get('data_source')=='native')
            except ValueError:continue
            valid.append(candidate)
        return valid[int(rng.integers(len(valid)))]if valid else None
    def summary(self):
        return dict(candidate_states=len(self.registry.frames),center_target_changes=self.changes,center_transition_observations=self.transition_steps,
                    simulated_seconds=self.previous.time-self.started,stop_reason=self.reason,legal_windows=len(self.windows()))
