"""Complete legal-source evaluation using a shared sparse bitset propagation kernel."""
import ctypes as ct
import fcntl,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from .graph import Geometry,csr
from ..protocol import kind_mask,CENTER

_LIBRARY=None

def library():
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    source=Path(__file__).with_name('kernel.cpp');cache=Path(tempfile.gettempdir())/f'voxwm_planner_{os.getuid()}'
    cache.mkdir(exist_ok=True);target=cache/(str(source.stat().st_mtime_ns)+('.dylib'if sys.platform=='darwin'else'.so'))
    with open(cache/'build.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            temporary=target.with_suffix('.'+str(os.getpid())+'.tmp')
            subprocess.run(['c++','-O3','-std=c++17',*(['-dynamiclib']if sys.platform=='darwin'else['-shared','-fPIC']),str(source),'-o',str(temporary)],check=True,capture_output=True)
            os.replace(temporary,target)
    lib=ct.CDLL(str(target));lib.vw_wave.restype=ct.c_int
    lib.vw_wave.argtypes=[ct.c_int]*6+[ct.c_void_p]*13
    _LIBRARY=lib;return lib

def pointer(array):return None if array is None else np.asarray(array).ctypes.data

def phase_ticks(mask):return [tick for tick in range(1,11)if int(mask)&(1<<tick)]

def four_stages(mask):
    times=phase_ticks(mask)
    if len(times)<4 or times[0]>2:return False
    return all(b-a<=2 for a,b in zip([0,*times[:4]],times[:4]))

def bernoulli_sources(indices,rng,maximum=4):
    ids=np.asarray(indices,dtype=np.int32);n=len(ids)
    if not n:return ids,dict(N=0,p=None,selected=[],fallback=False,cap=maximum)
    draws=rng.random(n);chosen=ids[draws<1/n];fallback=not len(chosen)
    if fallback:chosen=np.array([ids[int(rng.integers(n))]],np.int32)
    if len(chosen)>maximum:chosen=rng.choice(chosen,maximum,replace=False).astype(np.int32)
    return chosen,dict(N=n,p=1/n,selected=chosen.tolist(),fallback=fallback,cap=maximum,distribution='independent Bernoulli then uniform fallback/cap')

class Planner:
    def __init__(self,frame,vocabulary):
        self.geometry=Geometry(frame,vocabulary);self.frame=frame;self.v=vocabulary;self.results={};self.work_log=[]
    def evaluate(self,kind,partner=None,source_ids=None,detail=False):
        g=self.geometry;graph=g.graph(kind);legal=g.legal[kind]
        ids=np.arange(len(legal),dtype=np.int32)if source_ids is None else np.asarray(source_ids,np.int32)
        sources=np.ascontiguousarray(legal[ids],np.int32)
        if detail and len(ids)>64:raise ValueError('Detailed reach grids are streamed only for selected sources (max64)')
        contacts=g.contacts(kind,partner)if partner is not None else(None,None,None)
        stats=np.empty((len(ids),7),np.int32);arrival=np.empty((len(ids),g.size),np.uint8)if detail else None
        operations=np.zeros(1,np.uint64);workspace=np.zeros(1,np.uint64)
        lib=library();start=time.perf_counter()
        if len(ids):
            result=lib.vw_wave(g.size,len(ids),-1 if kind=='fire'else 7 if kind=='water'else 3,
                7 if kind=='lava'else 1,10,64,*[pointer(a)for a in(graph['ptr'],graph['edges'],graph['types'],g.kernel_targets.ravel(),graph['reverse'],sources,*contacts,stats,arrival,operations,workspace)])
            if result:raise RuntimeError('Shared propagation kernel failed')
        elapsed=time.perf_counter()-start
        # Four occupied time bands, not four voxels in one time band. Time is a
        # scheduled-wave estimate; chance/humidity/queue effects remain uncertain.
        qualified=np.array([four_stages(row[2])and(partner is None or row[5]>0)for row in stats],bool)
        metrics=dict(legal_source_count=len(legal),evaluated_source_count=len(ids),
            reachable_candidate_count=int(np.count_nonzero((stats[:,1]>0)|(stats[:,4]>0))),qualified_candidate_count=int(qualified.sum()),
            mathematically_pruned_count=int(np.count_nonzero(graph['reverse'][sources]==0)),
            visited_nodes=int(operations[0]),reachability_seconds=elapsed,shared_graph_build_seconds=g.build_seconds+graph['build_seconds'],
            kernel_working_bytes=int(workspace[0]),batch_size=64,uncertain_pruning_disabled=graph['uncertain_pruning_disabled'],
            candidates_per_second=len(ids)/elapsed if elapsed else None,
            semantics='10 scheduled-wave seconds: optimistic potential, NOT deterministic native GT',
            uncertainty='native queue ordering, fluid renewal/competition and ABM randomness are not fixed; lower bound0, no finite stochastic upper bound')
        if kind=='fire':metrics.update(g.fire_metrics())
        self.work_log.append(metrics)
        result=dict(kind=kind,source_ids=ids,sources=sources,stats=stats,qualified=qualified,metrics=metrics,arrivals=arrival)
        if source_ids is None:self.results[kind+('_partner'if partner is not None else'')]=result
        return result
    def details(self,kind,ids):
        g=self.geometry
        result=self.evaluate(kind,source_ids=ids,detail=True)
        return [dict(kind=kind,source_index=int(source_id),position=(np.array(np.unravel_index(int(p),self.geometry.shape))+self.frame.meta['region_origin']).tolist(),
            reachable_voxels=int(row[0]),center_reachable_voxels=int(row[1]),potential_transition_times=phase_ticks(row[2]),
            falling_risk_reachable_voxels=int(np.count_nonzero((result['arrivals'][i].reshape(g.shape)<255)&g.falling_risk))if kind=='fire'else 0,
            estimated_arrival_seconds=result['arrivals'][i].reshape(self.geometry.shape))for i,(source_id,p,row)in enumerate(zip(result['source_ids'],result['sources'],result['stats']))]
    def select(self,category,rng,subtype='',stable_pool=None):
        started=time.perf_counter();g=self.geometry;selection=dict(planner_version='shared-wave64-v1',rng_state=rng.bit_generator.state,algorithm='all-source shared graph / sparse 64-source bitsets',pair_compatibility_seconds=0.)
        selected={};details=[];relations=[]
        if category!='D4':
            kind={'D1':'water','D2':'lava','D3':'fire'}[category];result=self.evaluate(kind)
            chosen,sampling=bernoulli_sources(np.flatnonzero(result['qualified']),rng);selection[kind]=sampling
            selected[kind]=chosen
        elif subtype in('A','C'):
            kind='lava'if subtype=='A'else'water';other='water'if kind=='lava'else'lava'
            pool=kind_mask(self.frame.tokens,self.v,other)if stable_pool is None else stable_pool
            if np.count_nonzero(pool)<4:
                result=self.evaluate(kind,np.full(g.shape,255,np.uint8))
                return None,dict(reason='no_static_pool',metrics={kind:result['metrics']},**selection)
            partner=np.where(pool,0,255).astype(np.uint8);result=self.evaluate(kind,partner)
            chosen,sampling=bernoulli_sources(np.flatnonzero(result['qualified']),rng);selected[kind]=chosen;selection[kind]=sampling
        else:
            water=self.evaluate('water');lava=self.evaluate('lava')
            from ..reachability import pool_components
            names=np.asarray([name for name,_ in self.v.entries],object)
            has_pool=any(pool_components(np.array([kind in name and 'source'in name for name in names],bool)[self.frame.tokens]).any()for kind in('water','lava'))
            if has_pool:
                return None,dict(reason='existing_pool_in_D4_B',metrics={k:r['metrics']for k,r in self.results.items()},**selection)
            # Both complete sets are evaluated. The compatibility graph is implicit:
            # a queried edge exists only if the center-product/time constraints pass.
            possible_w=np.flatnonzero((water['stats'][:,6]>0)&(water['stats'][:,0]>1));possible_l=np.flatnonzero((lava['stats'][:,6]>0)&(lava['stats'][:,0]>1))
            w,ws=bernoulli_sources(possible_w,rng);l,ls=bernoulli_sources(possible_l,rng)
            selection.update(water=ws,lava=ls,unary_filter='10s reach intersects center or its one-cell product-contact halo; four-stage potential is checked JOINTLY',compatibility_graph='lazy shared contact-factor graph over complete source sets',conditional_distribution_changes=[])
            pair_start=time.perf_counter();pairs=[]
            for wi in w:
                wd=self.details('water',[wi])[0]
                joint=self.evaluate('lava',wd['estimated_arrival_seconds'],source_ids=l)
                for j,li in enumerate(l):
                    union=int(joint['stats'][j,2])|int(water['stats'][wi,2])
                    if joint['stats'][j,5]and four_stages(union):pairs.append((int(wi),int(li)))
            if not pairs and len(w)and len(possible_l):
                # One bounded conditional fallback. All lava points remain eligible
                # for the shared query; no Cartesian N_water x N_lava scan is built.
                wi=int(w[int(rng.integers(len(w)))]);wd=self.details('water',[wi])[0]
                joint=self.evaluate('lava',wd['estimated_arrival_seconds'])
                good=[int(li)for li in possible_l if joint['stats'][li,5]and four_stages(int(joint['stats'][li,2])|int(water['stats'][wi,2]))]
                if good:
                    li=good[int(rng.integers(len(good)))];pairs=[(wi,li)]
                    selection['conditional_distribution_changes'].append(dict(method='uniform compatible lava conditional on sampled water',water_source=wi,compatible_lava_count=len(good),conditional_probability=1/len(good)))
            selection['pair_compatibility_seconds']=time.perf_counter()-pair_start
            if pairs:
                pair=pairs[int(rng.integers(len(pairs)))];selected={'water':np.array([pair[0]],np.int32),'lava':np.array([pair[1]],np.int32)}
                relations=[dict(water=pair[0],lava=pair[1],confirmed_prediction_only=True)];selection['selected_compatible_edges']=relations
            else:selected={}
        for kind,ids in selected.items():details.extend(self.details(kind,ids))
        selection['precompute_wall_seconds']=time.perf_counter()-started
        selection['metrics']={kind:result['metrics']for kind,result in self.results.items()}
        # Partner-mode results are still complete evaluations; include their metrics.
        if category=='D4'and subtype in('A','C'):selection['metrics']={kind:result['metrics']}
        if not details:return None,dict(selection,reason='no_qualified_temporal_center_potential')
        actions=[]
        for d in details:
            action=dict(type=d['kind'],position=d['position'])
            if d['kind']=='fire':
                p=np.unravel_index(int(g.legal['fire'][d['source_index']]),g.shape)
                for offset in((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    q=np.array(p)+offset
                    if np.all(q>=0)and np.all(q<g.shape)and g.fuel[tuple(q)]:action['under']=(q+self.frame.meta['region_origin']).tolist();break
            actions.append(action)
        selection['details']=details
        return actions,selection
