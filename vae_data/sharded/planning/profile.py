"""Measured CPU planning only, using existing observations and labelled fixtures; no simulator."""
import argparse,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
from .planner import Planner,library
from .reference import reference
from .fixtures import fixture
from ..protocol import Frame,Vocabulary,dump
from ..trace import TraceReader


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin'else 1024)

def load(path):
    path=Path(path)
    if (path/'trace.json').exists():
        r=TraceReader(path);return next(r.frames()),Vocabulary(path/'vocabulary.json')
    import zarr
    meta=json.loads((path/'metadata.json').read_text());a=zarr.open_group(str(path/'snapshots.zarr'),mode='r')
    v=Vocabulary(path.parents[1]/'vocabulary.json')
    return Frame(a['tokens'][0],a['light'][0],meta['frames'][0]),v

def measure(kind,scene):
    started=time.perf_counter();library();compile_seconds=time.perf_counter()-started
    f,v=fixture(forest=True)if scene=='fixture'else load(scene)
    before=rss_mb();started=time.perf_counter();p=Planner(f,v);pair=0.;query_count=0
    if kind=='pair':
        _,selection=p.select('D4',np.random.default_rng(42),'B')
        pair=selection.get('pair_compatibility_seconds',0.)
        query_count=int('lava_partner'in p.results)
    else:p.evaluate(kind)
    elapsed=time.perf_counter()-started;metrics=p.work_log
    return dict(task=kind,scene=scene,source='synthetic_fixture'if scene=='fixture'else'existing_real_native_GT',
        legal_source_count={k:len(p.geometry.legal[k])for k in(('water','lava')if kind=='pair'else(kind,))},
        reachable_candidate_count={k:r['metrics']['reachable_candidate_count']for k,r in p.results.items()},
        qualified_candidate_count={k:r['metrics']['qualified_candidate_count']for k,r in p.results.items()},
        precompute_wall_seconds=elapsed,precompute_peak_memory_mb=rss_mb(),incremental_peak_memory_mb=max(0,rss_mb()-before),
        visited_nodes=sum(m['visited_nodes']for m in metrics),shared_graph_build_seconds=p.geometry.build_seconds+sum(g['build_seconds']for g in p.geometry.graphs.values()),
        reachability_seconds=sum(m['reachability_seconds']for m in metrics),pair_compatibility_seconds=pair,
        compatibility_all_source_queries=query_count,candidates_per_second=sum(m['evaluated_source_count']for m in metrics)/elapsed,
        kernel_peak_working_mb=max(m['kernel_working_bytes']for m in metrics)/1024**2,batch_size=64,
        compile_or_load_seconds=compile_seconds,evaluation_calls=len(metrics),naive_vs_optimized_speedup=None,
        caveat='Absolute process peak RSS includes Python/imports/input/graphs; pair runs the actual independent samplers and bounded conditional fallback; no Cartesian pair enumeration.')

def comparison(kind):
    f,v=fixture(forest=True);p=Planner(f,v);k='lava'if kind=='pair'else kind
    ids=np.linspace(0,len(p.geometry.legal[k])-1,24,dtype=np.int32);partner=None
    if kind=='pair':partner=np.full(f.tokens.shape,255,np.uint8);partner[22:25,22:25,19]=1
    p.geometry.graph(k);contacts=p.geometry.contacts(k,partner)if partner is not None else None
    start=time.perf_counter();optimized=p.evaluate(k,partner,ids,True);opt=time.perf_counter()-start
    start=time.perf_counter();expected=[reference(p.geometry,k,int(p.geometry.legal[k][i]),contacts)for i in ids];naive=time.perf_counter()-start
    for j,(stats,arrival)in enumerate(expected):
        np.testing.assert_array_equal(stats,optimized['stats'][j]);np.testing.assert_array_equal(arrival.ravel(),optimized['arrivals'][j])
    return dict(task=kind,source='synthetic_fixed_fixture',source_count=len(ids),equal=True,
        naive_local_bfs_seconds=naive,optimized_seconds=opt,naive_vs_optimized_speedup=naive/opt,
        scope='same24sources with earliest-arrival outputs; shared graph already built for both; no native speedup claim')

def run(output,real,fire=None,pair=None):
    library();rows=[]
    for scene in('fixture',str(real)):
        for kind in('water','lava','fire','pair'):
            selected=str(fire)if fire and scene!='fixture'and kind=='fire'else str(pair)if pair and scene!='fixture'and kind=='pair'else scene
            command=[sys.executable,'-m','vae_data.sharded.planning.profile','--worker',kind,'--scene',selected]
            child=subprocess.run(command,capture_output=True,text=True,check=True)
            rows.append(json.loads(child.stdout))
    result=dict(native_simulations_started=0,profiles=rows,reference_comparisons=[comparison(k)for k in('water','lava','fire','pair')])
    dump(output,result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=('water','lava','fire','pair'));parser.add_argument('--scene')
    parser.add_argument('--real',type=Path);parser.add_argument('--fire',type=Path);parser.add_argument('--pair',type=Path);parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.worker:result=measure(a.worker,a.scene)
    else:
        if not a.real or not a.output:parser.error('--real and --output required')
        result=run(a.output,a.real,a.fire,a.pair)
    print(json.dumps(result,indent=2))
