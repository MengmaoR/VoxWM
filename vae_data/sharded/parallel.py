"""Multiple independent world processes feeding one exclusive writer per shard."""
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import json,os,time
import numpy as np
from voxwm_gt.runtime import InfrastructureError,stop_tree
from .protocol import Vocabulary,Chain
from .storage import Writer,atomic_json


def failure_path(root):
    return Path(root)/'.runtime'/('infrastructure_'+os.environ.get('VOXWM_RUN_ID','manual')+'.json')


def infrastructure_failure(root,shard,error):
    path=failure_path(root);path.parent.mkdir(exist_ok=True)
    atomic_json(path,dict(status='FAIL',shard=shard,pid=os.getpid(),reason=str(error)),sync=False)


def world_process(channel,deadline,craftium,renderer,sites_per_world,tick_batch_size=24):
    # No Native objects, sockets or mutable vocabulary are shared across worlds.
    from .native import Native
    from .generate import prepare_static,collect
    from .centers import prepare_dynamic,choose_height
    world=None;uses=0
    try:
        while True:
            job=channel.recv()
            if job is None:break
            rng=np.random.default_rng();rng.bit_generator.state=job['rng'];row={};chain=None;fatal=False
            try:
                if world is None or uses>=sites_per_world:
                    if world:world.__exit__(None,None,None);world=None
                    world=Native(int(rng.integers(1,2147483647)),Vocabulary(),deadline,craftium,renderer);world.__enter__();world.tick_batch_size=tick_batch_size;uses=0
                uses+=1;row['world_seed']=world.seed
                site=rng.integers(-8192,8193,2)//16*16;category=job['category'];subtype=job['subtype']
                if category=='Static':chain=Chain('Static',[prepare_static(world,job['target'],site,rng,row)])
                else:
                    height=choose_height(category,rng);row.update(height_type=height,subtype=subtype)
                    frame,_=prepare_dynamic(world,site,category,rng,subtype,height)
                    chain,details=collect(world,frame,category,rng,subtype);row.update(details)
            except Exception as exc:
                fatal=isinstance(exc,InfrastructureError);row.update(status='infrastructure_error'if fatal else'failed',reason=f'{type(exc).__name__}: {exc}')
                if world:world.__exit__(None,None,None);world=None
            channel.send(dict(row=row,chain=chain,pairs=list(world.v.entries)if chain is not None else[],rng=rng.bit_generator.state,fatal=fatal))
            if fatal:break
            if world and world.retire_after_batch:world.__exit__(None,None,None);world=None
    except (EOFError,BrokenPipeError):pass
    finally:
        if world:world.__exit__(None,None,None)
        channel.close()


def generate_parallel(root,shard,resume,deadline,craftium,renderer,seed,max_attempts,sites_per_world,worlds,tick_batch_size=24):
    from .generate import choose_category,choose_static_target
    final=Path(root)/'shards'/f'shard_{shard:04d}'
    if(final/'READY').exists():return dict(shard=shard,status='skipped_ready')
    ctx=mp.get_context('spawn');processes=[];channels=[];pending={};started=time.monotonic()
    try:
        with Writer(root,shard,resume)as writer:
            cursor=writer.path/'cursor.json';old=json.loads(cursor.read_text())if resume and cursor.exists()else{}
            attempts=old.get('attempts',0);issued=0;states=dict(old.get('world_rng',{}));rngs=[]
            for slot in range(worlds):
                rng=np.random.default_rng(seed+shard)if slot==0 else np.random.default_rng(np.random.SeedSequence([seed,shard,slot]))
                saved=states.get(str(slot),old.get('rng')if slot==0 else None)
                if saved:rng.bit_generator.state=saved
                rngs.append(rng);states[str(slot)]=rng.bit_generator.state
                parent,child=ctx.Pipe();process=ctx.Process(target=world_process,args=(child,deadline,craftium,renderer,sites_per_world,tick_batch_size))
                process.start();child.close();processes.append(process);channels.append(parent)
            def dispatch(slot):
                nonlocal issued,attempts
                if issued>=max_attempts or time.monotonic()+15>=deadline:return False
                rng=rngs[slot];category,missing=choose_category(writer,rng)
                if category is None:return False
                target=choose_static_target(missing['static'],rng)if category=='Static'else None
                choices=[k for k,n in missing['d4'].items()if n>0]if category=='D4'else[]
                subtype=choices[int(rng.integers(len(choices)))]if choices else''
                issued+=1;attempts+=1
                row=dict(attempt=attempts,category=category,world_slot=slot,source='real_native')
                channels[slot].send(dict(category=category,target=target,subtype=subtype,rng=rng.bit_generator.state))
                pending[channels[slot]]=(slot,row);return True
            for slot in range(worlds):dispatch(slot)
            while pending and time.monotonic()+5<deadline:
                if not any(writer.missing()['categories'].values()):break
                for channel in wait(list(pending),timeout=1):
                    slot,row=pending.pop(channel)
                    try:reply=channel.recv()
                    except EOFError:
                        error=InfrastructureError(f'World worker {slot} exited unexpectedly in shard {shard}')
                        infrastructure_failure(root,shard,error);raise error
                    row.update(reply['row']);rngs[slot].bit_generator.state=reply['rng'];states[str(slot)]=reply['rng']
                    if reply['fatal']:
                        infrastructure_failure(root,shard,row['reason'])
                        writer.attempt(row);raise InfrastructureError(row['reason'])
                    chain=reply['chain']
                    if chain is not None:
                        vocabulary=Vocabulary()
                        for name,param in reply['pairs']:vocabulary.add(name,param)
                        try:row['status']='committed'if writer.submit(chain,vocabulary)else'rejected_duplicate_or_quota'
                        except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                    writer.attempt(row)
                    atomic_json(cursor,dict(attempts=attempts,rng=states['0'],world_rng=states))
                    if attempts%10==0:writer.export()
                    dispatch(slot)
            # Stop simulation processes before doing potentially slow final QA/export.
            for channel in channels:
                try:channel.send(None)
                except (BrokenPipeError,EOFError,OSError):pass
            grace=time.monotonic()+2
            for process in processes:process.join(timeout=max(0,grace-time.monotonic()))
            for process in processes:
                if process.is_alive():stop_tree(process.pid);process.join(timeout=1)
            if not any(writer.missing()['categories'].values()):return dict(shard=shard,status='ready',worlds=worlds,quality=writer.publish())
            return dict(shard=shard,status='incomplete',worlds=worlds,statistics=writer.export(),wall_seconds=time.monotonic()-started)
    except InfrastructureError as exc:
        infrastructure_failure(root,shard,exc);return dict(shard=shard,status='infrastructure_error',error=str(exc))
    except Exception as exc:return dict(shard=shard,status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for process in processes:
            if process.is_alive():stop_tree(process.pid);process.join(timeout=1)
        for channel in channels:channel.close()
