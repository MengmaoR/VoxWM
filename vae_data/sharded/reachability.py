"""Connected natural-source pools. Formal source reachability lives in planning/."""
import numpy as np

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
