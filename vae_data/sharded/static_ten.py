"""User-requested ten-biome Static preview, not a production shard."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from .native import Native
from .protocol import BIOMES,Vocabulary,dump
from .generate import prepare_static
from .static_height import validate_static
from .validate import validate_native
from .trace import Trace
from .storage import atomic_json

TARGETS={'surface':4,'elevated_complex':2,'shallow_underground':2,'deep_cave':2}
PREFERRED={
    'surface':['Plains','Desert','Swampland','MushroomIsland','Savanna','IcePlains'],
    'elevated_complex':['MegaTaiga','RoofedForest','Forest','Taiga','Jungle','BambooJungle'],
    'shallow_underground':['Taiga','Jungle','Forest','BirchForest'],
    'deep_cave':['Mesa','IcePlains','ExtremeHills','ColdTaiga'],
}
MAX_ATTEMPTS=40


def run(args):
    root=Path(args.dataset)
    if root.exists():raise FileExistsError('Use a new ten-Static output directory')
    root.mkdir(parents=True);started=time.perf_counter();deadline=time.monotonic()+args.wall_seconds-10
    rng=np.random.default_rng(args.seed);accepted=[];attempts=[];used=set();heights=set();positions=set();world=None
    result=dict(status='INCONCLUSIVE',purpose='Ten distinct biomes, height allocation 4/2/2/2 only',
        sampling_seed=args.seed,targets=TARGETS,successes=accepted,attempts=attempts,max_attempts=MAX_ATTEMPTS,wall_budget=args.wall_seconds,
        formal_shard_generated=False,dynamic_rerun=False)
    try:
        # Tree biomes first, so required elevated structures do not lose all their
        # candidates to the less restrictive strata. No per-biome height quota.
        for stratum in('elevated_complex','surface','shallow_underground','deep_cave'):
            candidates=list(dict.fromkeys(PREFERRED[stratum]+list(BIOMES)))
            for group in candidates:
                if sum(s['stratum']==stratum for s in accepted)>=TARGETS[stratum]:break
                if group in used:continue
                for trial in range(3):
                    if len(attempts)>=MAX_ATTEMPTS or time.monotonic()+20>=deadline:break
                    row=dict(attempt=len(attempts)+1,target_biome=group,target_stratum=stratum,trial=trial+1)
                    attempts.append(row)
                    try:
                        if world is None:
                            seed=191664964 if len(attempts)==1 else int(rng.integers(1,2147483647))
                            world=Native(seed,Vocabulary(),deadline,args.craftium_dir,args.renderer);world.__enter__()
                        row['world_seed']=world.seed
                        origin=(rng.integers(-8192,8193,2)//16*16).tolist()
                        frame=prepare_static(world,dict(biome=group,stratum=stratum),origin,rng,row)
                        validate_native(frame);validate_static(frame,world.v)
                        c=frame.meta['selected_chunk_center_y'];position=(world.seed,*frame.meta['center_origin'])
                        if c in heights:raise ValueError('Repeated center height in this ten-scene demonstration')
                        if position in positions:raise ValueError('Repeated world position')
                        name=f'scene_{len(accepted)+1:02d}_{group}_{stratum}'
                        frame.meta.update(category='Static',preview_scene=name)
                        trace=Trace(root/'traces/Static'/name,world.v,'Static')
                        trace.observe(frame,'static_snapshot');trace.finish(status=dict(status='accepted_static',reason=None))
                        sample=dict(scene=name,biome_group=frame.meta['biome_group'],native_biome=frame.meta['surface_biome'],
                            stratum=stratum,world_seed=world.seed,center_y=c,center_origin=frame.meta['center_origin'],
                            region_origin=frame.meta['region_origin'],surface_reference_y=frame.meta['surface_reference_y'],
                            region_height_min=frame.meta['region_height_min'],region_height_max=frame.meta['region_height_max'],
                            air_fraction=frame.meta['air_fraction'],void_fraction=frame.meta['void_fraction'],
                            height_attempt_count=frame.meta['height_attempt_count'],trace=str(trace.root.relative_to(root)))
                        accepted.append(sample);used.add(group);heights.add(c);positions.add(position)
                        row.update(status='accepted',scene=name,center_y=c)
                        print('STATIC',len(accepted),group,stratum,'C='+str(c),flush=True)
                        break
                    except Exception as exc:
                        row.update(status='rejected',reason=f'{type(exc).__name__}: {exc}')
                        print('STATIC REJECT',group,stratum,row['reason'],flush=True)
                        if world:world.__exit__(None,None,None);world=None
                    finally:atomic_json(root/'static_results.json',result)
                if len(attempts)>=MAX_ATTEMPTS or time.monotonic()+20>=deadline:break
            if len(attempts)>=MAX_ATTEMPTS or time.monotonic()+20>=deadline:break
    finally:
        if world:world.__exit__(None,None,None)
    counts=Counter(s['stratum']for s in accepted)
    result.update(counts=dict(counts),distinct_biomes=len(used),distinct_heights=len(heights),wall_seconds=time.perf_counter()-started)
    result['status']='GO'if len(accepted)==10 and len(used)==10 and len(heights)==10 and dict(counts)==TARGETS else'INCONCLUSIVE'
    atomic_json(root/'static_results.json',result)
    from .export_trace import export,publish_static
    export(root)
    if args.preview_dataset:
        publish_static(root,args.preview_dataset);result['preview_dataset']=str(args.preview_dataset)
        atomic_json(root/'static_results.json',result)
    return result
