"""Shared geometry, full legal enumeration, CSR adjacency and optimistic reverse bounds."""
from collections import deque
import time
import numpy as np
from ..protocol import CENTER,kind_mask

CARDINAL=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

def shifted(mask,d):
    out=np.zeros(mask.shape,mask.dtype);a=[];b=[]
    for size,delta in zip(mask.shape,d):
        a.append(slice(max(0,-delta),min(size,size-delta)));b.append(slice(max(0,delta),min(size,size+delta)))
    out[tuple(a)]=mask[tuple(b)];return out

def csr(source,destination,types=None,size=None):
    source=np.asarray(source,np.int32);destination=np.asarray(destination,np.int32);order=np.argsort(source,kind='stable')
    source=source[order];destination=destination[order];n=size if size is not None else int(source.max(initial=-1)+1)
    ptr=np.zeros(n+1,np.int32);ptr[1:]=np.cumsum(np.bincount(source,minlength=n))
    return ptr,destination,np.asarray(types,np.uint8)[order]if types is not None else None

class Geometry:
    def __init__(self,frame,vocabulary):
        start=time.perf_counter();self.frame=frame;self.v=vocabulary;self.shape=frame.tokens.shape;self.size=frame.tokens.size
        self.ids=np.arange(self.size,dtype=np.int32).reshape(self.shape)
        self.names=np.asarray([n for n,_ in vocabulary.entries],object)[frame.tokens]
        self.air=self.names=='air';self.replaceable=np.isin(self.names,frame.meta.get('buildable_names',[]))&~(kind_mask(frame.tokens,vocabulary,'water')|kind_mask(frame.tokens,vocabulary,'lava'))
        self.protected=np.zeros(self.shape,bool)
        self.protected.ravel()[frame.meta.get('protected_source_indices',[])]=True
        self.fuel=np.isin(self.names,frame.meta.get('flammable_names',[]))
        self.wet=np.isin(self.names,frame.meta.get('wet_names',[]))|kind_mask(frame.tokens,vocabulary,'water')
        self.floodable=np.isin(self.names,frame.meta.get('floodable_names',[]))
        self.falling=np.isin(self.names,frame.meta.get('falling_names',[]))
        self.fuel_adj=np.zeros(self.shape,bool);self.wet_adj=np.zeros(self.shape,bool)
        for d in CARDINAL:self.fuel_adj|=shifted(self.fuel,d);self.wet_adj|=shifted(self.wet,d)
        self.center=np.zeros(self.shape,np.uint8);self.center[CENTER]=1
        self.contact_halo=self.center.astype(bool).copy()
        for d in CARDINAL:self.contact_halo|=shifted(self.center,d).astype(bool)
        ambient=kind_mask(frame.tokens,vocabulary,'water')|kind_mask(frame.tokens,vocabulary,'lava')
        self.ambient_contact=ambient.copy()
        for d in CARDINAL:self.ambient_contact|=shifted(ambient,d)
        self.kernel_targets=self.center+2*self.contact_halo.astype(np.uint8)+4*self.ambient_contact.astype(np.uint8)
        self.legal={'water':np.flatnonzero(((self.air|self.replaceable)&~self.protected).ravel()).astype(np.int32),'lava':np.flatnonzero(((self.air|self.replaceable)&~self.protected).ravel()).astype(np.int32),
            'fire':np.flatnonzero((self.air&~shifted(kind_mask(frame.tokens,vocabulary,'water'),(0,0,-1))&~self.protected).ravel()).astype(np.int32)}
        self.falling_risk=self.fuel&shifted(self.falling,(0,0,1));self.graphs={};self.build_seconds=time.perf_counter()-start
    def graph(self,kind):
        if kind in self.graphs:return self.graphs[kind]
        start=time.perf_counter();src=[];dst=[];types=[]
        if kind=='fire':
            valid=(self.air&self.fuel_adj)|self.fuel
            offsets={(dx,dy,dz)for dx in(-1,0,1)for dy in(-1,0,1)for dz in(-1,0,1,2,3,4)if(dx,dy,dz)!=(0,0,0)}
            for d in offsets|set(CARDINAL):
                target=shifted(self.air&self.fuel_adj,d)if d in offsets else np.zeros(self.shape,bool)
                # Native air-cube spread plus adjacent structural burning, no age-only edges.
                if d in CARDINAL:target|=shifted(self.fuel,d)
                mask=valid&target;source=self.ids[mask];destination=(shifted(self.ids,d))[mask]
                src.append(source);dst.append(destination);types.append(np.full(len(source),2,np.uint8))
        else:
            passable=self.air|self.floodable|kind_mask(self.frame.tokens,self.v,kind)
            valid=passable|self.replaceable
            down=shifted(passable,(0,0,-1))
            for d in CARDINAL:
                if d==(0,0,1):continue
                mask=valid&shifted(passable,d);source=self.ids[mask];destination=shifted(self.ids,d)[mask]
                typ=np.ones(len(source),np.uint8)if d==(0,0,-1)else np.where(down[mask],3,0).astype(np.uint8)
                src.append(source);dst.append(destination);types.append(typ)
        p,e,t=csr(np.concatenate(src),np.concatenate(dst),np.concatenate(types),self.size)
        # Optimistic graph removes range, flow-down suppression and timing restrictions.
        # Destructible materials are not claimed globally impossible: disable pruning if
        # the liquid scene includes flammable/ice/snow removal that the graph omits.
        uncertain=kind!='fire'and bool(self.fuel.any()or np.array(['ice'in name or'snow'in name for name,_ in self.v.entries],bool)[self.frame.tokens].any())
        if uncertain:reverse=np.ones(self.size,np.uint8)
        else:
            sorted_sources=np.repeat(np.arange(self.size,dtype=np.int32),np.diff(p))
            q,r,_=csr(e,sorted_sources,size=self.size);reverse=np.zeros(self.size,np.uint8)
            target_mask=self.center.astype(bool).copy()
            if kind!='fire':
                for d in CARDINAL:target_mask|=shifted(self.center,d).astype(bool)
            targets=np.flatnonzero(target_mask.ravel()&valid.ravel());queue=deque(map(int,targets));reverse[targets]=1
            while queue:
                node=queue.popleft()
                for predecessor in r[q[node]:q[node+1]]:
                    if not reverse[predecessor]:reverse[predecessor]=1;queue.append(int(predecessor))
        graph=dict(ptr=p,edges=e,types=t,reverse=reverse,uncertain_pruning_disabled=uncertain,
            build_seconds=time.perf_counter()-start,kind=kind)
        self.graphs[kind]=graph;return graph
    def fire_metrics(self):
        if hasattr(self,'_fire_metrics'):return self._fire_metrics
        remaining=set(map(int,np.flatnonzero(self.fuel.ravel())));components=[]
        while remaining:
            p=remaining.pop();queue=[p];count=0
            while queue:
                p=queue.pop();count+=1;xyz=np.array(np.unravel_index(p,self.shape))
                for d in CARDINAL:
                    q=xyz+d
                    if np.all(q>=0)and np.all(q<self.shape):
                        key=int(np.ravel_multi_index(q,self.shape))
                        if key in remaining:remaining.remove(key);queue.append(key)
            components.append(count)
        self._fire_metrics=dict(center_flammable_voxels=int(self.fuel[16:32,16:32,16:32].sum()),
            flammable_components=len(components),largest_flammable_component=max(components,default=0),
            center_spread_air_voxels=int((self.air&self.fuel_adj)[16:32,16:32,16:32].sum()),
            falling_risk_voxels=int(self.falling_risk.sum()),falling_policy='native falling/entity attempt rejects rollout; risk is not frozen')
        return self._fire_metrics
    def contacts(self,active_kind,partner_arrival):
        """CSR active-cell -> actual product position, not just either contact endpoint."""
        src=[];products=[];times=[];partner=np.asarray(partner_arrival).reshape(self.shape)
        for d in CARDINAL:
            available=shifted(partner,d)<255
            if active_kind=='lava':
                # Water below lava produces stone AT THE WATER; other contacts AT LAVA.
                product_center=shifted(self.center,d)if d==(0,0,-1)else self.center
            else:
                # Water below lava produces stone AT WATER; other contacts AT LAVA.
                product_center=self.center if d==(0,0,1)else shifted(self.center,d)
            mask=available&product_center.astype(bool)
            source=self.ids[mask]
            product=(shifted(self.ids,d)[mask]if(active_kind=='lava'and d==(0,0,-1))or(active_kind=='water'and d!=(0,0,1))else source)
            src.append(source);products.append(product);times.append(shifted(partner,d)[mask])
        sources=np.concatenate(src);order=np.argsort(sources,kind='stable')
        p=np.zeros(self.size+1,np.int32);p[1:]=np.cumsum(np.bincount(sources,minlength=self.size))
        return p,np.concatenate(products)[order].astype(np.int32),np.concatenate(times)[order].astype(np.uint8)
