"""Batch transport and exact-stop fixtures; no native worlds or large datasets."""
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import copy,json,os,socket,struct,subprocess,tempfile,threading,time,unittest
import numpy as np
from vae_data.sharded.batch import Batch,CELLS,DELTA,CLOCK,receive,VERSION
from vae_data.sharded.clock import RAW,TickProbe,verify_clock,PATCH_VERSION
from vae_data.sharded.generate import observation_ticks
from vae_data.sharded.native import Native
from vae_data.sharded.sampling import Collector
from vae_data.sharded.tests.test_sharded import frame,overlap_fixture
from vae_data.sharded.protocol import Vocabulary

ROOT=Path(__file__).resolve().parents[3]


def packet(states,start=100):
    data=bytearray(b'VWB1'+struct.pack('<II',len(states),CELLS)+states[0].tobytes())
    for old,new in zip(states,states[1:]):
        changed=np.flatnonzero(old.view('<u4').ravel()!=new.view('<u4').ravel())
        delta=np.empty(len(changed),DELTA);delta['index']=changed;delta['raw']=new.ravel()[changed]
        data.extend(struct.pack('<I',len(delta)));data.extend(delta.tobytes())
    for tick in range(start+1,start+1+len(states)):data.extend(struct.pack('<Qf',tick,1/24))
    return data


def clock_row(tick):
    return dict(native_clock=dict(server_tick=tick,client_request_tick=tick,server_dt=1/24,physics_us=10,patch_version=PATCH_VERSION),
        lua_tick=tick,sim_time=tick/24,simulation_step=tick,engine_us=tick*100)


class ReplayWorld:
    def __init__(self,vocabulary,states,size=48):
        self.v=vocabulary;self.states=states;self.tick_batch_size=size;self.at=0;self.captured=0;self.tail=0;self.retire_after_batch=False
    def tick(self):
        f=self.states[self.at];self.at+=1;return f,dict(reaction_events=f.meta.get('fixture_events',[]))
    def tick_batch(self,n):
        group=self.states[self.at:self.at+n];self.at+=len(group);self.captured+=len(group);consumed=0
        try:
            for f in group:
                consumed+=1;yield f,dict(reaction_events=f.meta.get('fixture_events',[]))
        finally:self.tail+=len(group)-consumed;self.retire_after_batch|=len(group)>consumed


def consume(world,monitor):
    with closing(observation_ticks(world,monitor))as observations:
        for f,result in observations:
            if monitor.observe(f,result['reaction_events']):break
    return monitor


class BatchTests(unittest.TestCase):
    def test_cpp_wire_roundtrip_and_in_engine_barrier_bookkeeping(self):
        with tempfile.TemporaryDirectory()as tmp:
            exe=Path(tmp)/'codec'
            subprocess.run(['c++','-std=c++17','-O1','-I',str(ROOT/'voxwm_gt/patches'),str(Path(__file__).with_name('batch_fixture.cpp')),'-o',str(exe)],check=True,timeout=10,capture_output=True)
            payload=subprocess.check_output([str(exe)],timeout=3)
        decoded=Batch(payload,48);self.assertLess(len(payload),CELLS*4+4096)
        for t,raw in enumerate(decoded.states(),1):
            a=raw.ravel();self.assertEqual(int(a[123]['node']),256+t);self.assertEqual(int(a[123]['param2']),128+t)
            self.assertEqual(int(a[901]['light']),t);self.assertEqual(int(a[900]['node']),0)
        self.assertEqual(decoded.clocks['tick'].tolist(),list(range(101,149)))
        np.testing.assert_allclose(decoded.clocks['dt'],1/24,atol=1e-8)

    def test_zero_delta_still_preserves_all_48_ticks(self):
        raw=np.zeros(CELLS,RAW);payload=packet([raw]*48);b=Batch(payload,48)
        self.assertEqual(len(list(b.states())),48);self.assertEqual([n for _,n in b.offsets],[0]*47)
        with self.assertRaises(ValueError):Batch(payload,24)
        with self.assertRaises(ValueError):Batch(payload[:-1],48)

    def test_transport_receives_one_length_prefixed_memory_packet(self):
        raw=np.zeros(CELLS,RAW);expected=packet([raw]*2);a,b=socket.socketpair();errors=[]
        def send():
            try:b.sendall(struct.pack('<Q',len(expected))+expected)
            except Exception as exc:errors.append(exc)
            finally:b.close()
        worker=threading.Thread(target=send);worker.start()
        try:
            self.assertTrue(os.get_blocking(a.fileno()))
            self.assertEqual(receive(SimpleNamespace(connfd=a.fileno()),time.monotonic()+3),expected)
            self.assertTrue(os.get_blocking(a.fileno()),'Duplicate fd must not leave Craftium recv nonblocking')
        finally:a.close();worker.join(3)
        self.assertFalse(errors);self.assertFalse(worker.is_alive())

    def test_48_idle_ticks_stop_at_48_all_categories(self):
        for category,limit in(('D1',48),('D2',48),('D3',120),('D4',48)):
            v=Vocabulary();anchor=frame(v,0);world=ReplayWorld(v,[frame(v,t)for t in range(1,193)])
            monitor=consume(world,Collector(anchor,category,v))
            self.assertEqual(monitor.previous.step,limit);self.assertEqual(monitor.changes,0)
            self.assertEqual(len(monitor.registry.frames),0 if category=='D4'else 1)
            self.assertEqual(world.tail,24 if category=='D3'else 0)

    def test_mid_batch_change_resets_timeout_and_tail_does_not_add_candidates(self):
        v=Vocabulary();water=v.add('mcl_core:water_flowing',1);anchor=frame(v,0);states=[]
        for t in range(1,145):
            f=frame(v,t)
            if t>=20:f.tokens[20,20,20]=water
            if t>=69:f.tokens[21,20,20]=water # Would be a candidate after the true stop (68).
            states.append(f)
        world=ReplayWorld(v,states);monitor=consume(world,Collector(anchor,'D1',v))
        self.assertEqual(monitor.previous.step,68);self.assertEqual(monitor.last_change_step,20)
        self.assertEqual([f.step for f in monitor.registry.frames],[0,20]);self.assertEqual(world.tail,28);self.assertTrue(world.retire_after_batch)

    def test_batch_matches_single_tick_chain_and_stops_at_candidate_ten(self):
        runs=[]
        for size in(1,24,48):
            v=Vocabulary();water=v.add('mcl_core:water_flowing',1);anchor=frame(v,0);states=[]
            for t in range(1,145):
                f=frame(v,t);f.tokens[20:20+min(12,t//5),20,20]=water;states.append(f)
            world=ReplayWorld(v,states,size);monitor=consume(world,Collector(anchor,'D1',v))
            chosen=monitor.select(np.random.default_rng(42));self.assertIsNotNone(chosen)
            runs.append((monitor.previous.step,[f.step for f in chosen.frames],monitor.summary()))
            self.assertEqual(len(monitor.registry.frames),10);self.assertEqual(monitor.previous.step,45)
        self.assertEqual(runs[0],runs[1]);self.assertEqual(runs[1],runs[2])

    def test_d4_real_reaction_roles_match_single_tick(self):
        runs=[]
        for size in(1,24,48):
            v,anchor,initial,events=overlap_fixture();states=initial+[frame(v,t,initial[-1].tokens)for t in range(7,145)]
            for f in states:f.meta['fixture_events']=events.get(f.step,[])
            world=ReplayWorld(v,states,size);monitor=consume(world,Collector(anchor,'D4',v));chain=monitor.select(np.random.default_rng(42),subtype='B')
            self.assertIsNotNone(chain);runs.append(([f.step for f in chain.frames],chain.events))
        self.assertEqual(runs[0],runs[1]);self.assertEqual(runs[1],runs[2])

    def test_maximum_sim_time_does_not_expand_with_batch(self):
        v=Vocabulary();water=v.add('mcl_core:water_flowing',1);anchor=frame(v,0);states=[]
        for t in range(1,193):
            f=frame(v,t)
            if(t//10)%2:f.tokens[20,20,20]=water
            states.append(f)
        world=ReplayWorld(v,states);monitor=consume(world,Collector(anchor,'D1',v,max_seconds=2.5))
        self.assertEqual(monitor.previous.step,60);self.assertEqual(monitor.reason,'maximum_sim_time');self.assertEqual(world.captured,60)

    def test_probe_checks_every_server_client_lua_clock_including_tail(self):
        p=object.__new__(TickProbe);raw=np.zeros(CELLS,RAW);payload=packet([raw]*3);p.deadline=time.monotonic()+3;p.communication_seconds=0
        p.s=SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace(mt_chann='fixture')),info=dict(native_client_tick=103,native_client_dt=1/24))
        response=dict(batch_transport_version=VERSION,status='batch_captured',count=3,frames=[clock_row(t)for t in range(101,104)])
        p.command=Mock(side_effect=lambda *a,**kw:copy.deepcopy(response))
        with patch('vae_data.sharded.batch.receive',return_value=payload):
            result,batch=p.advance_batch(3,100);self.assertEqual(batch.count,3);self.assertEqual(result['frames'][1]['batch_client_info']['native_client_tick'],102)
            response['frames'][2]['lua_tick']=102
            with self.assertRaisesRegex(ValueError,'clock mismatch'):p.advance_batch(3,100)
            response['frames'][2]=clock_row(104)
            with self.assertRaises(ValueError):p.advance_batch(3,100)
            response['frames']=[clock_row(t)for t in range(101,104)]
            response['frames'][1]['native_clock']['server_dt']=1/12
            with self.assertRaisesRegex(ValueError,'Non-fixed'):p.advance_batch(3,100)

    def test_native_generator_decodes_no_tail_and_cannot_reuse_world(self):
        raw=np.zeros(CELLS,RAW);p=Batch(packet([raw]*3),3);native=object.__new__(Native)
        native._last_tick=100;native._performance=[];native._batches=[];native.checked=lambda x:x;native.tick_batch_size=24
        response=dict(chunks=[],center_origin=[0,0,0],capture_reference_equal=True,engine_mapblock_count=27,
            frames=[dict(clock_row(t),batch_client_info=dict(native_client_tick=t,native_client_dt=1/24),capture_us=10)for t in range(101,104)])
        native.probe=SimpleNamespace(communication_seconds=0,advance_batch=lambda *args:(response,p));seen=[]
        def decode(result,cells,info):
            seen.append(result['simulation_step']);result['decode_seconds']=0
            return SimpleNamespace(step=result['simulation_step'],meta=dict(native_provenance={})),result
        native.frame=decode
        with closing(native.tick_batch(3))as batch:self.assertEqual(next(batch)[0].step,101)
        self.assertEqual(seen,[101]);self.assertTrue(native.retire_after_batch);self.assertEqual(native._batches[0]['discarded_tail_ticks'],2)
        with self.assertRaises(ValueError):next(native.tick_batch(3))

    def test_single_advance_keeps_exact_one_tick_contract(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=object.__new__(TickProbe);p.seq=0;p.communication_seconds=p.step_wall_seconds=0;p.check=lambda:None;p.s=SimpleNamespace(world=Path(tmp));count=0
            def delayed_step():
                nonlocal count
                count+=1;(Path(tmp)/'pilot_response.json').write_text(json.dumps(dict(request_id=0 if count==1 else 1)))
            p.step=delayed_step
            with self.assertRaisesRegex(ValueError,'multiple physical ticks'):p.command('advance')

    def test_cli_retains_world_count_and_exposes_24_48_transport(self):
        from vae_data.sharded.cli import parser
        for n in(1,24,48):
            a=parser(['generate','--dataset','unused','--wall-seconds','10','--workers','64','--tick-batch-size',str(n)])
            self.assertEqual(a.workers,64);self.assertEqual(a.tick_batch_size,n)
        self.assertEqual(parser(['clock-verify','--dataset','unused','--wall-seconds','10']).tick_batch_size,1)

if __name__=='__main__':unittest.main()
