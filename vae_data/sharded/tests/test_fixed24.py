"""Small timing/domain fixtures; none are published as native physics evidence."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vae_data.sharded.protocol import Frame,check_sources,NO_CHANGE_TICKS
from vae_data.sharded.planning import Planner
from vae_data.sharded.planning.fixtures import fixture
from vae_data.sharded.native import Native
from vae_data.sharded.clock import verify_clock,PATCH_VERSION,RegionEncoder,RAW
from vae_data.sharded.sampling import Collector
from vae_data.sharded.trace import Trace,TraceReader
from vae_data.sharded.tests.test_sharded import frame
from vae_data.sharded.protocol import Vocabulary

class Fixed24Tests(unittest.TestCase):
    def test_all_six_source_domains_and_all_selection_branches(self):
        for category,subtype in(('D1',''),('D2',''),('D3',''),('D4','A'),('D4','B'),('D4','C')):
            f,v=fixture(forest=True);p=Planner(f,v)
            for legal in p.geometry.legal.values():
                coordinates=np.array(np.unravel_index(legal,f.tokens.shape)).T
                self.assertTrue(np.all((coordinates>=17)&(coordinates<31)))
            pool=None
            if subtype in('A','C'):
                kind='water'if subtype=='A'else'lava';node=v.add('mcl_core:'+kind+'_source',0)
                f.tokens[20:24,20:24,18:20]=node;pool=np.zeros((48,)*3,bool);pool[20:24,20:24,18:20]=True;p=Planner(f,v)
            actions,selection=p.select(category,np.random.default_rng(42),subtype,pool)
            if actions:check_sources(actions,[0,0,0])
            for key in('water','lava','fire'):
                sample=selection.get(key)
                if sample and sample['N']:
                    self.assertEqual(sample['p'],1/sample['N'])
                    positions=p.geometry.legal[key][sample['selected']]
                    xyz=np.array(np.unravel_index(positions,f.tokens.shape)).T
                    self.assertTrue(np.all((xyz>=17)&(xyz<31)))
    def test_boundaries_and_neighbors_rejected_before_native_call(self):
        native=object.__new__(Native);native.center=[0,0,0];native.probe=SimpleNamespace(command=Mock())
        for value in(-1,0,15,16,24):
            for kind in('water','lava','fire'):
                with self.assertRaisesRegex(ValueError,'outside_center_14'):native.place([dict(type=kind,position=[value,5,5])])
        native.probe.command.assert_not_called();self.assertTrue(check_sources([dict(position=[1,14,7])],[0,0,0]))
    def test_lua_integer_valued_float_positions_are_valid_without_rounding(self):
        self.assertTrue(check_sources([dict(position=[102.,1974.,3.])],[96,1968,0]))
        for invalid in([102.5,1974.,3.],[float('nan'),1974.,3.],[float('inf'),1974.,3.],[96.,1974.,3.]):
            with self.assertRaisesRegex(ValueError,'outside_center_14'):check_sources([dict(position=invalid)],[96,1968,0])
        native=object.__new__(Native);native.center=[96,1968,0];native.checked=lambda x:x
        native.probe=SimpleNamespace(command=lambda *a,**kw:dict(status='captured',actions=[dict(type='water',position=[102.,1974.,3.])]))
        native.frame=lambda r:(SimpleNamespace(meta=dict(frame_index=0),step=100),r)
        native.place([dict(type='water',position=[102,1974,3])]);self.assertEqual(native._last_tick,100)
    def test_returned_actual_source_out_of_bounds_rejected(self):
        native=object.__new__(Native);native.center=[0,0,0];native.checked=lambda x:x
        native.probe=SimpleNamespace(command=lambda *a,**kw:dict(status='captured',actions=[dict(type='water',position=[0,5,5])]))
        with self.assertRaisesRegex(ValueError,'outside_center_14'):native.place([dict(type='water',position=[1,5,5])])
    def test_idle_timeout_exact_ticks_in_all_four_categories(self):
        for category,limit in NO_CHANGE_TICKS.items():
            v=Vocabulary();monitor=Collector(frame(v,0),category,v)
            for tick in range(1,limit):self.assertFalse(monitor.observe(frame(v,tick)))
            self.assertTrue(monitor.observe(frame(v,limit)));self.assertEqual(monitor.summary()['simulated_ticks'],limit)
    def test_fire_age_and_neighbor_activity_do_not_reset_120_ticks(self):
        v=Vocabulary();start=frame(v,0);start.tokens[20,20,20]=v.add('mcl_fire:fire',0);monitor=Collector(start,'D3',v)
        for tick in range(1,121):
            current=frame(v,tick,start.tokens);current.tokens[20,20,20]=v.add('mcl_fire:fire',tick)
            current.tokens[2,2,2]=v.add('mcl_fire:fire',tick)
            stopped=monitor.observe(current)
            self.assertEqual(stopped,tick==120)
        self.assertEqual(monitor.changes,0);self.assertEqual(len(monitor.registry.frames),1)
    def test_real_center_structure_change_resets_fire_clock(self):
        v=Vocabulary();start=frame(v,0);monitor=Collector(start,'D3',v)
        for tick in range(1,201):
            current=frame(v,tick)
            if tick>=80:current.tokens[20,20,20]=v.add('mcl_fire:fire',0)
            stopped=monitor.observe(current)
            self.assertEqual(stopped,tick==200)
        self.assertEqual(monitor.last_change_step,80)
    def test_fire_transition_classes_and_age_are_separate(self):
        from vae_data.sharded.protocol import fire_change_counts
        v=Vocabulary();a=frame(v,0);fire=v.add('mcl_fire:fire',0);leaves=v.add('mcl_core:leaves',0)
        a.tokens[20,20,20]=leaves;a.tokens[21,20,20]=fire;a.tokens[22,20,20]=fire;a.tokens[23,20,20]=leaves
        b=frame(v,1,a.tokens);b.meta['flammable_names']=['mcl_core:leaves']
        b.tokens[20,20,20]=fire;b.tokens[21,20,20]=v.add('mcl_fire:fire',2);b.tokens[22,20,20]=v.add('air',0)
        b.tokens[23,20,20]=v.add('air',0);b.tokens[23,21,20]=fire
        counts=fire_change_counts(a,b,v)
        for key in('new_fire','fuel_to_fire','fire_age_update','fire_extinguished','fuel_removed_near_fire'):self.assertEqual(counts[key],1)
    def test_sampler_ignores_wall_throughput(self):
        summaries=[]
        for wall_value in(.001,10000.):
            v=Vocabulary();monitor=Collector(frame(v,0),'D3',v)
            with patch('time.monotonic',return_value=wall_value):
                for tick in range(1,121):monitor.observe(frame(v,tick))
            summaries.append(monitor.summary())
        self.assertEqual(summaries[0],summaries[1])
    def test_missing_or_merged_tick_rejected(self):
        v=Vocabulary();monitor=Collector(frame(v,0),'D1',v)
        with self.assertRaisesRegex(ValueError,'consecutive'):monitor.observe(frame(v,2))
    def test_integer_clock_checkpoints_and_server_client_mismatch(self):
        previous=None
        for n in range(1,241):
            c=dict(server_tick=n,client_request_tick=n,patch_version=PATCH_VERSION,server_dt=float(np.float32(1/24)))
            result=dict(native_clock=c,lua_tick=n,sim_time=n/24)
            info=dict(native_client_tick=n,native_client_dt=c['server_dt'])
            verify_clock(result,info,previous);previous=n
            if n in(24,48,120,240):self.assertEqual(result['sim_time'],{24:1,48:2,120:5,240:10}[n])
        with self.assertRaisesRegex(ValueError,'clock mismatch'):verify_clock(result,dict(info,native_client_tick=239))
    def test_bulk_combined_tokens_restore_original_names_and_param2(self):
        v=Vocabulary();names={0:'air',4:'mcl_core:water_flowing'};encoder=RegionEncoder(v,names)
        raw=np.zeros((27,16,16,16),RAW);raw['node'][13]=4;raw['param2'][13]=13;raw['light'][:]=123
        tokens=encoder.encode(raw)
        for i in range(27):
            decoded,params=v.decode(tokens[i]);np.testing.assert_array_equal(params,raw['param2'][i]);self.assertTrue(np.all(decoded==names[int(raw['node'][i,0,0,0])]))
        np.testing.assert_array_equal(tokens,encoder.encode(raw));self.assertEqual(len(v.entries),2)
    def test_real_unchanged_tick_records_survive_sparse_delta(self):
        # A native flag is set only to exercise the stream contract, not to label evidence.
        v=Vocabulary()
        with tempfile.TemporaryDirectory()as tmp:
            trace=Trace(Path(tmp)/'trace',v,'fixture')
            for tick in range(4):
                f=frame(v,tick);f.meta['data_source']='native';trace.observe(f,'fixture_tick')
            trace.finish();reader=TraceReader(Path(tmp)/'trace');states=list(reader.frames())
            self.assertEqual(len(states),4);self.assertEqual([s.meta['frame_index']for s in states],list(range(4)))
            self.assertEqual([r['changed_cells']for r in reader.rows][1:],[0,0,0])
            self.assertTrue(all(np.array_equal(states[0].tokens,s.tokens)for s in states))

if __name__=='__main__':unittest.main()
