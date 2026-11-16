# Historical pre-exhaustive planner, retained only for compatibility fixtures.
# The formal generator uses vae_data.sharded.planning.Planner.
"""Geometry-only native-fluid reachability filter, never a fluid simulator or GT source."""
from collections import deque
import numpy as np
from vae_data.sharded.protocol import CENTER,kind_mask

DIRECTIONS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0))

def potential(passable,source,liquid_range):
    """Gravity first; horizontal level decays, downward flow restores range.

    Same-fluid cells are passable, solid/other-fluid cells are barriers/contact surfaces.
    This bounded geometric envelope ignores timing, viscosity, competition and source renewal.
    It is only a feasibility screen; native contact/write observations confirm reactions.
    """
    shape=passable.shape;levels=np.full(shape,-1,np.int8);source=tuple(map(int,source))
    if not all(0<=p<n for p,n in zip(source,shape)):return levels>=0
    levels[source]=liquid_range;queue=deque([source])
    while queue:
        p=queue.popleft();level=int(levels[p]);below=(p[0],p[1],p[2]-1)
        if below[2]>=0 and passable[below]:
            if levels[below]<liquid_range:levels[below]=liquid_range;queue.append(below)
            # Down-flowing nodes do not emit a horizontal stream in the pinned engine.
            if p!=source:continue
        if level<=0:continue
        for dx,dy,dz in DIRECTIONS:
            q=(p[0]+dx,p[1]+dy,p[2]+dz)
            if all(0<=a<n for a,n in zip(q,shape)) and passable[q]and levels[q]<level-1:
                levels[q]=level-1;queue.append(q)
    return levels>=0

def contact_points(water,lava):
    """Six-neighbor contact with at least one member in the center chunk."""
    contacts=np.zeros(water.shape,bool)
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        contacts[tuple(a)]|=water[tuple(a)]&lava[tuple(b)]
        contacts[tuple(b)]|=water[tuple(b)]&lava[tuple(a)]
        contacts[tuple(a)]|=lava[tuple(a)]&water[tuple(b)]
        contacts[tuple(b)]|=lava[tuple(b)]&water[tuple(a)]
    return int(contacts[CENTER].sum())

def pool_components(mask):
    """Retain connected natural-source pools with a real two-dimensional footprint."""
    remaining={tuple(p)for p in np.argwhere(mask)};accepted=np.zeros(mask.shape,bool)
    while remaining:
        first=remaining.pop();stack=[first];component=[first]
        while stack:
            p=stack.pop()
            for axis in range(3):
                for offset in(-1,1):
                    q=list(p);q[axis]+=offset;q=tuple(q)
                    if q in remaining:remaining.remove(q);stack.append(q);component.append(q)
        footprint={(p[0],p[1])for p in component}
        if len(footprint)>=4 and len({p[0]for p in footprint})>=2 and len({p[1]for p in footprint})>=2:
            for p in component:accepted[p]=True
    return accepted

def plan(frame,vocabulary,subtype,rng,budget=32,stable_pool=None):
    names=np.asarray([name for name,_ in vocabulary.entries],object)[frame.tokens]
    air=names=='air';floodable=np.isin(names,frame.meta.get('floodable_names',[]));open_nodes=air|floodable;water=kind_mask(frame.tokens,vocabulary,'water');lava=kind_mask(frame.tokens,vocabulary,'lava')
    source_water=np.array(['water'in n and 'source'in n for n,_ in vocabulary.entries],bool)[frame.tokens]
    source_lava=np.array(['lava'in n and 'source'in n for n,_ in vocabulary.entries],bool)[frame.tokens]
    if subtype=='A'and source_water.sum()<4:return None
    if subtype=='C'and source_lava.sum()<4:return None
    if subtype=='B'and(source_water.sum()>=4 or source_lava.sum()>=4):return None
    # Include upper context so falling streams can contact the center. Bounded candidate work.
    allowed=np.zeros(air.shape,bool);allowed[12:36,12:36,16:40]=True
    candidates=np.argwhere(air&allowed);rng.shuffle(candidates);candidates=candidates[:budget]
    routes={}
    if subtype=='B':
        routes={(tuple(p),kind):potential(open_nodes,tuple(p),amount)for p in candidates for kind,amount in(('water',7),('lava',3))}
    qualified=[]
    for p in candidates:
        p=tuple(p)
        if subtype in('A','C'):
            kind='lava'if subtype=='A'else'water';base=water if subtype=='A'else lava
            if stable_pool is not None:base=base&stable_pool
            route=potential(open_nodes|(lava if kind=='lava'else water),p,3 if kind=='lava'else 7)
            score=contact_points(base,route)
            if score:qualified.append(([(kind,p)],score))
        else:
            for q in candidates:
                q=tuple(q)
                distance=np.linalg.norm(np.array(p)-q)
                if not 2<=distance<=8:continue
                w=routes[(p,'water')];l=routes[(q,'lava')];score=contact_points(w,l)
                if score:qualified.append(([('water',p),('lava',q)],score));break
    if not qualified:return None
    actions,score=qualified[int(rng.integers(len(qualified)))];origin=np.asarray(frame.meta['region_origin'])
    return dict(actions=[dict(type=kind,position=(origin+point).tolist())for kind,point in actions],
        subtype=subtype,predicted_center_contact_cells=score,screen='gravity/range/obstacle geometry; not physical evidence',
        tested_candidates=len(candidates),qualified_plans=len(qualified))
