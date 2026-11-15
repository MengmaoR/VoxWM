"""Distinct-state sampling preserves the terminal state once, without capping the raw process."""
import numpy as np
from vae_p0.sampling import State

def select_states(event,maximum=8):
    states=list(event.distinct())
    if len(states)<5:return []
    final=event.previous;first=states[0]
    # If final repeats the initial state, one initial sample represents both; raw time is retained.
    if not np.array_equal(final.tokens,first.tokens):
        states=[s for s in states if not np.array_equal(s.tokens,final.tokens)]
        states.append(State(final.tokens,final.light,final.meta,event.updates))
    n=min(maximum,len(states))
    if len(states)==n:return states
    targets=np.linspace(0,states[-1].progress,n);selected={0,len(states)-1}
    for target in targets[1:-1]:
        options=[i for i in range(1,len(states)-1) if i not in selected]
        selected.add(min(options,key=lambda i:abs(states[i].progress-target)))
    return [states[i] for i in sorted(selected)]
