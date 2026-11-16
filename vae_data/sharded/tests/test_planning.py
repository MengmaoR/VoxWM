"""Small CPU fixtures: geometric predictions are never native success evidence."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from vae_data.sharded.planning import Planner,bernoulli_sources
from vae_data.sharded.planning.planner import four_stages,phase_ticks
from vae_data.sharded.planning.reference import reference
from vae_data.sharded.planning.fixtures import fixture
from vae_data.sharded.protocol import Frame,target_changes
from vae_data.sharded.sampling import Registry,Collector
from vae_data.sharded.trace import Trace,TraceReader

class PlanningTests(unittest.TestCase):
    def test_all_legal_points_no_presampling_and_protection(self):
        f,v=fixture(large=True);f.meta['protected_source_indices']=[int(np.ravel_multi_index((20,20,20),f.tokens.shape))]
        p=Planner(f,v);r=p.evaluate('water')
        self.assertEqual(len(r['sources']),71423);self.assertEqual(r['metrics']['evaluated_source_count'],71423)
        self.assertEqual(len(np.unique(r['sources'])),71423)
        self.assertEqual(r['metrics']['batch_size'],64);self.assertLess(r['metrics']['kernel_working_bytes'],40*1024**2)
    def test_optimized_equals_independent_reference_all_four_modes(self):
        for kind in('water','lava','fire','pair'):
            f,v=fixture(forest=True);p=Planner(f,v);k='lava'if kind=='pair'else kind
            legal=p.geometry.legal[k];ids=np.linspace(0,len(legal)-1,4,dtype=int)
            partner=None
            if kind=='pair':partner=np.full(f.tokens.shape,255,np.uint8);partner[22:25,22:25,19]=1
            r=p.evaluate(k,partner,source_ids=ids,detail=True)
            contact=p.geometry.contacts(k,partner)if partner is not None else None
            for i,source in enumerate(r['sources']):
                stats,arrival=reference(p.geometry,k,int(source),contact)
                np.testing.assert_array_equal(stats,r['stats'][i]);np.testing.assert_array_equal(arrival.ravel(),r['arrivals'][i])
    def test_time_horizon_and_center_separation(self):
        f,v=fixture(large=True);p=Planner(f,v);point=np.ravel_multi_index((0,0,47),f.tokens.shape)
        sid=int(np.searchsorted(p.geometry.legal['water'],point));r=p.evaluate('water',source_ids=[sid],detail=True)
        grid=r['arrivals'][0];self.assertFalse(np.any((grid>10)&(grid<255)))
        self.assertEqual(r['stats'][0,1],0);self.assertFalse(r['qualified'][0])
    def test_four_distinct_time_stages_not_four_voxels(self):
        self.assertFalse(four_stages(1<<1));self.assertFalse(four_stages(sum(1<<t for t in(3,4,5,6))))
        self.assertFalse(four_stages(sum(1<<t for t in(1,2,5,6))));self.assertTrue(four_stages(sum(1<<t for t in(1,2,3,4))))
    def test_bernoulli_independent_one_over_n_uniform_fallback_and_cap(self):
        class R:
            def random(self,n):self.n=n;return np.ones(n)
            def integers(self,n):return n-1
        r=R();ids,stats=bernoulli_sources(np.arange(19),r)
        self.assertEqual(r.n,19);self.assertEqual(stats['p'],1/19);self.assertTrue(stats['fallback']);self.assertEqual(ids.tolist(),[18])
        class All:
            def random(self,n):return np.zeros(n)
            def choice(self,a,n,replace):assert not replace;return a[:n]
        ids,_=bernoulli_sources(np.arange(19),All());self.assertEqual(len(ids),4)
    def test_pair_independent_probabilities_and_real_product_positions(self):
        f,v=fixture();p=Planner(f,v);actions,s=p.select('D4',np.random.default_rng(42),'B')
        self.assertEqual(s['water']['p'],1/s['water']['N']);self.assertEqual(s['lava']['p'],1/s['lava']['N'])
        if actions:self.assertIn('selected_compatible_edges',s)
        # Lava in center above water OUTSIDE center must not claim a center stone.
        partner=np.full(f.tokens.shape,255,np.uint8);partner[20,20,15]=0
        ptr,products,times=p.geometry.contacts('lava',partner)
        lava=np.ravel_multi_index((20,20,16),f.tokens.shape)
        self.assertEqual(ptr[lava],ptr[lava+1])
        partner[:]=255;partner[20,20,16]=0;ptr,products,times=p.geometry.contacts('lava',partner)
        lava=np.ravel_multi_index((20,20,17),f.tokens.shape)
        self.assertIn(np.ravel_multi_index((20,20,16),f.tokens.shape),products[ptr[lava]:ptr[lava+1]])
    def test_joint_pair_needs_real_center_product_and_distinct_time_phases(self):
        f,v=fixture(large=True);p=Planner(f,v);g=p.geometry
        wi=int(np.searchsorted(g.legal['water'],np.ravel_multi_index((20,20,17),g.shape)))
        li=int(np.searchsorted(g.legal['lava'],np.ravel_multi_index((24,20,17),g.shape)))
        wd=p.evaluate('water',source_ids=[wi],detail=True)
        joint=p.evaluate('lava',wd['arrivals'][0].reshape(g.shape),source_ids=[li])
        self.assertGreater(joint['stats'][0,4],0)
        self.assertTrue(four_stages(int(wd['stats'][0,2])|int(joint['stats'][0,2])))
    def test_isolated_natural_sources_do_not_trigger_pool_guard(self):
        f,v=fixture();water=v.add('mcl_core:water_source',0)
        for point in((18,19,19),(18,25,19),(27,19,19),(27,25,19)):f.tokens[point]=water
        p=Planner(f,v);_,selection=p.select('D4',np.random.default_rng(42),'B')
        self.assertNotEqual(selection.get('reason'),'existing_pool_in_D4_B')
        for kind in('water','lava'):
            self.assertEqual(p.results[kind]['metrics']['evaluated_source_count'],len(p.geometry.legal[kind]))
    def test_fire_age_alone_never_counts(self):
        f,v=fixture();fire0=v.add('mcl_fire:fire',0);fire1=v.add('mcl_fire:fire',1)
        f.tokens[20,20,20]=fire0;g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=1,sim_time=.2))
        g.tokens[20,20,20]=fire1;self.assertEqual(target_changes(f,g,v,'D3'),0)
    def test_reuse_equal_role_state_does_not_forge_physical_time(self):
        f,v=fixture();r=Registry();r.add(f)
        g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=9,sim_time=9.))
        key=r.add(g,'second_event','pre');self.assertEqual(key,f.key);self.assertEqual(r.by_key[key].time,0.)
        self.assertEqual(len(r.frames),1);anchor=Registry(f);self.assertIsNone(anchor.add(f,'e','pre'))
    def test_native_trace_lossless_and_synthetic_rejected(self):
        f,v=fixture()
        with tempfile.TemporaryDirectory()as tmp:
            target=Path(tmp)/'trace';trace=Trace(target,v,'D1')
            with self.assertRaises(ValueError):trace.observe(f,'original')
            # Native source flag is set only to test the storage contract; no fixture
            # from this unit test is published as native evidence.
            f.meta['data_source']='native';trace.observe(f,'original')
            g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=3,sim_time=1.7));g.tokens[20,20,20]=v.add('mcl_core:water_source',0);g.light[20,20,20]=241
            trace.observe(g,'native_observation');trace.finish(status={'status':'fixture_storage_test'})
            states=list(TraceReader(target).frames());self.assertEqual(len(states),2)
            for actual,expected in zip(states,(f,g)):
                np.testing.assert_array_equal(actual.tokens,expected.tokens);np.testing.assert_array_equal(actual.light,expected.light)
                self.assertEqual(actual.time,expected.time);self.assertEqual(actual.key,expected.key)
    def test_six_type_caps(self):
        from vae_data.sharded.cli import parser
        from vae_data.sharded.native_six import TYPES
        self.assertEqual(len(set(TYPES)),6)
        with contextlib.redirect_stderr(io.StringIO()):
            for tail in(('--wall-seconds','901'),('--wall-seconds','900','--attempts-per-type','4')):
                with self.assertRaises(SystemExit):parser(['native-six','--dataset','unused',*tail])

if __name__=='__main__':unittest.main()
