"""Bounded read-only native clock test; does not create a dynamic chain or shard."""
from pathlib import Path
import json,time,traceback,resource,sys
import numpy as np
from .native import Native
from .protocol import Vocabulary,SIMULATION_HZ,dump
from .trace import Trace,TraceReader


def run(args):
    root=Path(args.dataset)
    if root.exists():raise FileExistsError('Use a fresh clock-test directory')
    root.mkdir(parents=True);started=time.perf_counter()
    result=dict(status='INCONCLUSIVE',purpose='clock verification only, not six-type native acceptance',simulation_hz=SIMULATION_HZ,
        requested_ticks=args.ticks,dynamic_examples_generated=0,formal_shard_generated=False,consumer_work_ms=args.consumer_work_ms)
    trace=None
    try:
        with Native(args.seed,Vocabulary(),time.monotonic()+args.wall_seconds-10,args.craftium_dir,args.renderer)as world:
            world.prepare_at([96,1968,0],1);anchor,_=world.place([])
            trace=Trace(root/'trace',world.v,'clock_verification');trace.observe(anchor,'clock_baseline')
            loop=time.perf_counter();changed=0;previous=anchor;clocks=[]
            for index in range(1,args.ticks+1):
                if args.consumer_work_ms:
                    # Diagnostic CPU workload ONLY; never a production FPS limiter/sleep.
                    until=time.perf_counter()+args.consumer_work_ms/1000
                    while time.perf_counter()<until:sum(i*i for i in range(100))
                frame,_=world.tick()
                if frame.meta['frame_index']!=index:raise ValueError('Relative tick mismatch')
                changed+=int(np.count_nonzero(frame.tokens!=previous.tokens));previous=frame
                trace.observe(frame,'native_tick');clocks.append(frame.meta['native_clock'])
            elapsed=time.perf_counter()-loop
            result.update(clock_verified=True,consecutive_native_ticks=True,ticks_observed=args.ticks,
                clock_checkpoints={str(n):n/SIMULATION_HZ for n in(24,48,120,240)if n<=args.ticks},
                first_absolute_tick=anchor.step,last_absolute_tick=previous.step,source_initialization='none; read-only existing natural scene',
                native_voxel_changes=changed,performance=world.performance(),trace_delta_write_seconds=trace.write_seconds,
                loop_wall_seconds=elapsed,ticks_per_wall_second_including_trace=args.ticks/elapsed,
                simulation_seconds_per_wall_second_including_trace=args.ticks/SIMULATION_HZ/elapsed,
                server_client_lua_equal=all(c['server_tick']==c['client_request_tick']==c['client_completed_tick']==c['lua_tick']for c in clocks),
                same_tick_p0_vs_cpp_capture_equal=True)
            trace.finish(status=result)
            result['trace_delta_write_seconds']=trace.write_seconds
        result['peak_child_rss_mb']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024**2 if sys.platform=='darwin'else 1024)
        if args.compare_to:
            reference=TraceReader(Path(args.compare_to)/'trace');current=TraceReader(root/'trace')
            equal=len(reference.rows)==len(current.rows);initial=None;mismatch=None
            for i,(a,b)in enumerate(zip(reference.frames(),current.frames())):
                same=np.array_equal(a.tokens,b.tokens)and np.array_equal(a.light,b.light)
                if i==0:initial=same
                if not same and mismatch is None:mismatch=i
                aligned=all(a.meta.get(k)==b.meta.get(k)for k in('frame_index','relative_sim_time','world_seed','center_origin'))
                equal&=same and aligned
            result['throughput_comparison']=dict(reference=str(args.compare_to),initial_state_equal=initial,
                observed_voxel_sequence_equal=bool(equal),first_mismatch_tick=mismatch,
                controlled_rng_replay=False,interpretation='Two real runs only; native global RNG/emergence are not snapshotted. Equality here does not prove universal determinism.')
        result['limits']='Clock/capture synchronization verified only. New fixed24 Water/Lava/Fire trajectories, controlled-RNG replay and pixel equivalence need separate bounded native experiments; six-type examples were not rerun.'
    except Exception as exc:
        result.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}',last_native_response=getattr(locals().get('world'),'last_result',None));(root/'error.txt').write_text(traceback.format_exc())
        if trace and not trace.stream.closed:trace.finish(status=result)
    result['total_wall_seconds']=time.perf_counter()-started;dump(root/'clock_results.json',result)
    return result
