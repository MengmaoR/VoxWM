"""Small process/IPC fixtures only: no engine, X server or native generation."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import contextlib,io,json,multiprocessing,os,subprocess,tempfile,time,unittest
from unittest.mock import Mock,patch
import numpy as np
from vae_data.sharded.protocol import Vocabulary,Chain
from vae_data.sharded.storage import Writer,globalize,initialize,lock,stage_json,rows_at
from vae_data.sharded.tests.test_sharded import frame
from voxwm_gt.runtime import InfrastructureError


def competing_publication(root,name,barrier,channel):
    v=Vocabulary();f=frame(v,0);f.tokens[20,20,20]=v.add(name,7)
    staged=False
    def sync_before_lock(*args,**kwargs):
        nonlocal staged
        path=stage_json(*args,**kwargs)
        if not staged:staged=True;barrier.wait(timeout=5)
        return path
    try:
        with patch('vae_data.sharded.storage.stage_json',sync_before_lock):mapped,_=globalize(root,[f],v)
        decoded=Vocabulary(Path(root)/'vocabulary.json').decode(mapped[0].tokens)
        channel.send((str(decoded[0][20,20,20]),int(decoded[1][20,20,20])))
    finally:channel.close()


def fixture_world(channel,deadline,craftium,renderer,sites_per_world):
    job=channel.recv();v=Vocabulary();f=frame(v,0,seed=os.getpid())
    channel.send(dict(row=dict(worker_pid=os.getpid()),chain=Chain('Static',[f]),pairs=v.entries,rng=job['rng'],fatal=False))
    channel.recv();channel.close()


def nested_fixture_generation(root):
    from vae_data.sharded.parallel import generate_parallel
    with patch('vae_data.sharded.parallel.Writer',partial(Writer,fixture=True)),patch('vae_data.sharded.parallel.world_process',fixture_world):
        return generate_parallel(root,0,False,time.monotonic()+30,'unused','auto',42,2,8,2)


class RuntimeTests(unittest.TestCase):
    def test_known_tokens_skip_lock_and_disk_sync(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=initialize(Path(tmp)/'data',True);v=Vocabulary();f=frame(v,0)
            mapped,_=globalize(root,[f],v);before=(root/'vocabulary.json').read_bytes()
            with patch('vae_data.sharded.storage.lock',side_effect=AssertionError('unnecessary lock')),patch('os.fsync',side_effect=AssertionError('unnecessary sync')):
                again,_=globalize(root,[f],v)
            np.testing.assert_array_equal(mapped[0].tokens,again[0].tokens)
            self.assertEqual(before,(root/'vocabulary.json').read_bytes())

    def test_new_tokens_sync_without_holding_global_lock(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=initialize(Path(tmp)/'data',True);v=Vocabulary();f=frame(v,0)
            def check_sync(fd):
                with lock(root/'.vocabulary.lock',blocking=False):pass
            with patch('os.fsync',side_effect=check_sync)as sync:globalize(root,[f],v)
            self.assertEqual(sync.call_count,1)

    def test_competing_publishers_preserve_both_tokens_and_existing_ids(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=initialize(Path(tmp)/'data',True);v=Vocabulary();globalize(root,[frame(v,0)],v)
            prefix=list(Vocabulary(root/'vocabulary.json').entries);ctx=multiprocessing.get_context('spawn');barrier=ctx.Barrier(2)
            processes=[];channels=[]
            try:
                for name in('mcl_core:water_flowing','mcl_core:lava_flowing'):
                    parent,child=ctx.Pipe();proc=ctx.Process(target=competing_publication,args=(root,name,barrier,child));proc.start();child.close()
                    processes.append(proc);channels.append(parent)
                replies=[]
                for channel in channels:
                    self.assertTrue(channel.poll(8),'publication worker stalled');replies.append(channel.recv())
                for proc in processes:proc.join(2);self.assertEqual(proc.exitcode,0)
                self.assertEqual(set(replies),{('mcl_core:water_flowing',7),('mcl_core:lava_flowing',7)})
                final=Vocabulary(root/'vocabulary.json').entries
                self.assertEqual(final[:len(prefix)],prefix);self.assertEqual(len(final),len(prefix)+2)
            finally:
                for proc in processes:
                    if proc.is_alive():proc.kill();proc.join()
                for channel in channels:channel.close()

    def test_lock_wait_is_bounded(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'lock'
            with lock(path):
                start=time.monotonic()
                with self.assertRaises(TimeoutError):
                    with lock(path,timeout=.02):pass
                self.assertLess(time.monotonic()-start,.3)
            with lock(path,blocking=False):pass

    def test_worker_count_has_no_shard_or_cpu_cap(self):
        from vae_data.sharded.cli import parser,worker_count,world_allocation
        self.assertEqual(world_allocation(100,200),([2]*100,100))
        self.assertEqual(world_allocation(1,256),([256],1))
        self.assertEqual(world_allocation(4,2),([1]*4,2))
        with patch('os.sched_getaffinity',return_value=set(range(192)),create=True):
            self.assertEqual(worker_count('auto'),192)
            self.assertEqual(parser(['generate','--dataset','unused','--wall-seconds','1']).workers,192)
        args=parser(['generate','--dataset','unused','--wall-seconds','1','--workers','512'])
        self.assertEqual(args.workers,512)

    def test_two_world_processes_share_exactly_one_shard_writer(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=initialize(Path(tmp)/'data',True)
            # Same nesting as production: shard process -> independent world processes.
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'))as pool:
                result=pool.submit(nested_fixture_generation,root).result(timeout=12)
            self.assertEqual(result['status'],'incomplete',result)
            path=root/'shards/.shard_0000.partial';rows,_=rows_at(path)
            self.assertEqual(len(rows),2);self.assertEqual(len({r['world_seed']for r in rows}),2)
            self.assertTrue((path/'progress').exists())
            with Writer(root,0,True,fixture=True):
                with self.assertRaises(BlockingIOError):Writer(root,0,True,fixture=True)

    def test_private_displays_get_distinct_dynamic_numbers_and_cleanup(self):
        from voxwm_gt.display import SoftwareDisplay
        with tempfile.TemporaryDirectory()as tmp:
            numbers=iter((120,121));processes=[]
            def start(args,**kwargs):
                self.assertIn('-displayfd',args);self.assertTrue(kwargs['start_new_session'])
                self.assertNotIn(':99',args);os.write(int(args[2]),str(next(numbers)).encode()+b'\n')
                proc=Mock();proc.poll.return_value=None;processes.append(proc);return proc
            with patch('voxwm_gt.display.shutil.which',return_value='/usr/bin/Xvfb'),patch('voxwm_gt.display.subprocess.Popen',side_effect=start),patch('voxwm_gt.display.stop_process')as stop:
                a=SoftwareDisplay(tmp);b=SoftwareDisplay(tmp)
                self.assertEqual(a.start(),':120');self.assertEqual(b.start(),':121');a.close();b.close()
                self.assertEqual(stop.call_count,2)

    def test_failed_xvfb_is_cleaned_and_not_silently_retried(self):
        from voxwm_gt.display import SoftwareDisplay
        with tempfile.TemporaryDirectory()as tmp:
            proc=Mock();proc.poll.return_value=1
            with patch('voxwm_gt.display.shutil.which',return_value='/usr/bin/Xvfb'),patch('voxwm_gt.display.subprocess.Popen',return_value=proc)as start,patch('voxwm_gt.display.stop_process')as stop:
                display=SoftwareDisplay(tmp)
                with self.assertRaises(InfrastructureError):display.start()
                self.assertEqual(start.call_count,1);stop.assert_called_once_with(proc);self.assertIsNone(display.log)

    def test_unconnected_engine_still_closes_socket_pipes_and_process(self):
        from voxwm_gt.native import close_env
        channel=Mock();channel.is_open.return_value=False;engine=Mock()
        env=SimpleNamespace(unwrapped=SimpleNamespace(mt=engine,mt_chann=channel))
        with patch('voxwm_gt.native.stop_process')as stop:close_env(env)
        channel.close.assert_called_once();channel.send_kill.assert_not_called();stop.assert_called_once_with(engine.proc)
        engine.close_pipes.assert_called_once();engine.clear.assert_called_once()

    def test_display_errors_are_fatal_and_engine_exit_does_not_wait_60s(self):
        from voxwm_gt.native import startup_error,connect_engine,renderer_config
        with tempfile.TemporaryDirectory()as tmp:
            (Path(tmp)/'stderr.txt').write_text('Maximum number of clients reached\nUnable to initialize SDL: x11 not available')
            self.assertIsInstance(startup_error(SimpleNamespace(run_dir=tmp),RuntimeError('failed')),InfrastructureError)
        channel=SimpleNamespace(listen_timeout=60000);engine=SimpleNamespace(proc=Mock());engine.proc.poll.return_value=1
        with self.assertRaises(RuntimeError):connect_engine(channel,engine,Mock())
        self.assertEqual(channel.listen_timeout,60000)
        _,env=renderer_config('software','linux');self.assertEqual(env['SDL_VIDEODRIVER'],'x11');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_infrastructure_failure_stops_attempt_retries(self):
        from vae_data.sharded.generate import generate_shard
        from vae_data.sharded.parallel import failure_path
        with tempfile.TemporaryDirectory()as tmp:
            root=initialize(Path(tmp)/'data',True)
            with patch('vae_data.sharded.generate.Writer',partial(Writer,fixture=True)),patch('vae_data.sharded.generate.Native',side_effect=InfrastructureError('display full'))as native:
                result=generate_shard(root,0,False,time.monotonic()+30,'unused','auto',42,10,8)
            self.assertEqual(result['status'],'infrastructure_error');self.assertEqual(native.call_count,1);self.assertTrue(failure_path(root).exists())

    def test_progress_ignores_retry_logs_but_tracks_committed_samples(self):
        from vae_data.__main__ import progress_signature
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'shards/.shard_0000.partial';path.mkdir(parents=True)
            before=progress_signature(root,0,1);(path/'statistics.json').write_text('retry');(path/'journal.sqlite').write_text('attempt')
            self.assertEqual(before,progress_signature(root,0,1));(path/'progress').touch()
            self.assertNotEqual(before,progress_signature(root,0,1))

    def test_fast_infrastructure_exit_still_reports_failure(self):
        from vae_data.__main__ import main
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=['vae_data','generate','--dataset',tmp,'--wall-seconds','100']
            proc=Mock();proc.pid=700001;proc.wait.return_value=1
            def launch(*args,env,**kwargs):
                path=root/'.runtime'/('infrastructure_'+env['VOXWM_RUN_ID']+'.json');path.parent.mkdir()
                path.write_text(json.dumps(dict(reason='Xvfb missing')));return proc
            with patch('sys.argv',args),patch('vae_data.__main__.subprocess.Popen',side_effect=launch),patch('vae_data.__main__.stop_tree'),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(),1)
            self.assertIn('Xvfb missing',(root/'infrastructure_failure.json').read_text())

    def test_no_progress_watchdog_stops_owned_tree(self):
        from vae_data.__main__ import main
        with tempfile.TemporaryDirectory()as tmp:
            args=['vae_data','generate','--dataset',tmp,'--wall-seconds','100','--stall-seconds','1','--workers','1']
            def wait(timeout=None):
                if timeout is not None:raise subprocess.TimeoutExpired('fixture',timeout)
                return 0
            with patch('sys.argv',args),patch('vae_data.__main__.subprocess.Popen')as popen,patch('vae_data.__main__.time.monotonic',side_effect=iter(range(20))),patch('vae_data.__main__.stop_tree')as stop,contextlib.redirect_stderr(io.StringIO()):
                popen.return_value.pid=700001;popen.return_value.wait.side_effect=wait
                self.assertEqual(main(),125);stop.assert_called_once_with(700001)
            self.assertEqual(json.loads((Path(tmp)/'stalled.json').read_text())['status'],'FAIL')

if __name__=='__main__':unittest.main()
