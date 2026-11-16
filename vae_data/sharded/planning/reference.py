"""Small, independent single-source reference BFS; never used for formal all-point planning."""
from collections import deque
import numpy as np

def reference(geometry,kind,source,contacts=None):
    graph=geometry.graph(kind);fire=kind=='fire';radius=7 if kind=='water'else 3;viscosity=7 if kind=='lava'else 1
    initial=(0 if fire else radius+1,int(source));queue=deque([(initial,0)]);seen={initial};arrival={int(source):0};cool={};phase=0;first=255
    if not graph['reverse'][source]:return np.array([0,0,0,255,0,0,0],np.int32),np.full(geometry.shape,255,np.uint8)
    while queue:
        (r,p),t=queue.popleft()
        if contacts is not None:
            ptr,product,partner=contacts
            for i in range(ptr[p],ptr[p+1]):
                time=max(t,int(partner[i]))+1
                if time<=10:cool[int(product[i])]=min(cool.get(int(product[i]),255),time)
        if t==10:continue
        for e in range(graph['ptr'][p],graph['ptr'][p+1]):
            typ=graph['types'][e];q=int(graph['edges'][e]);rr=0
            if not fire:
                if typ==3 and r!=radius+1:continue
                if typ==1:rr=radius
                else:
                    if r==0:continue
                    rr=radius-1 if r==radius+1 else r-1
            state=(rr,q)
            if state in seen:continue
            seen.add(state);queue.append((state,t+1));arrival[q]=min(arrival.get(q,255),t+1)
    centre=geometry.center.ravel();centers=0
    for p,t in arrival.items():
        if centre[p]:
            centers+=1
            if t:
                phase|=1<<t;first=min(first,t)
                if viscosity>1 and t>=2:
                    for k in range(t+1,min(10,t+viscosity-1)+1):phase|=1<<k
    product_phase=0
    for time in cool.values():product_phase|=1<<time;first=min(first,time)
    phase|=product_phase;grid=np.full(geometry.size,255,np.uint8)
    for p,t in arrival.items():grid[p]=t
    return np.array([len(arrival),centers,phase,first,len(cool),product_phase,sum(bool(geometry.contact_halo.ravel()[p])for p in arrival)],np.int32),grid.reshape(geometry.shape)
