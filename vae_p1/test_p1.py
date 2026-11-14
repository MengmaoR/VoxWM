"""Small no-game unit checks for P1's scientific data boundaries."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from .candidates import d2_schedule,choose,fire_draw,next_plant_window,plan,budget
from .tracking import Vocabulary,Event,Reactions,FireMerges
from .storage import Store,Reader
from .run import parse_args,run,collect
from .dry_run import DryWorld
from vae_p0.sampling import State

class P1Tests(unittest.TestCase):
    def setUp(self):
        self.v=Vocabulary();self.air=self.v.add('air',0);self.light=np.zeros((16,16,16),np.uint8)
    def state(self,tokens=None,t=0,origin=(0,0,0)):
        a=np.full((16,16,16),self.air,np.uint16) if tokens is None else tokens.copy()
        return State(a,self.light.copy(),dict(world_seed=42,chunk_origin=list(origin),timestamp=float(t),simulation_step=t+1,split='train',data_source='synthetic_dry_run'))
    def scene(self):
        return dict(scene_id='fixture',instance_id='independent1',pair_id='pair',world_seed=42,index=0,category='D1',subtype='',setting='C',interaction_context='ordinary',delay_seconds=0,synthetic=True,site=dict(x=0,z=0,depth=0))
    def test_scene_attempt_ratios(self):
        a=d2_schedule(10,np.random.default_rng(42));b=d2_schedule(10,np.random.default_rng(42))
        self.assertEqual(a,b);self.assertEqual([a.count(s) for s in ('ordinary','path_2_4','immediate')],[7,2,1])
    def test_stratified_selection(self):
        candidates=[dict(position=[i*3,0,0],stratum_key=str(i%3)) for i in range(30)]
        a=choose(candidates,np.random.default_rng(42),6,3);b=choose(candidates,np.random.default_rng(42),6,3)
        self.assertEqual(a,b);self.assertEqual(len({c['stratum_key'] for c in a}),3)
    def test_fire_draw_and_empty_fallback(self):
        class EmptyRNG:
            def random(self,n):return np.full(n,0.5)
            def integers(self,n):return 7
        draw=fire_draw([{}]*20,EmptyRNG());self.assertEqual(draw['selected_indices'],[7]);self.assertEqual(len(draw['draws']),20)
        with self.assertRaises(ValueError):fire_draw([{}]*19,np.random.default_rng(42))
    def test_intervention_is_not_native_change(self):
        water=self.v.add('mcl_core:water_source',0);post=self.state();post.tokens[0,0,0]=water
        e=Event(post,'D1',self.v);self.assertEqual(e.updates,0)
        evolved=self.state(post.tokens,1);evolved.tokens[1,0,0]=water;e.append(evolved)
        self.assertEqual(e.updates,1);self.assertTrue(e.has_native_evolution())
    def test_distinct_progress_and_light(self):
        water=self.v.add('mcl_core:water_flowing',0);start=self.state();e=Event(start,'D1',self.v)
        light=self.state(t=1);light.light[:]=99;e.append(light);self.assertEqual(len(e.records),1)
        a=start.tokens.copy()
        for i in range(9):
            a[0,0,i]=water;e.append(self.state(a,i+2))
        selected=e.select();self.assertEqual(len(selected),8);self.assertEqual(len({s.tokens.tobytes() for s in selected}),8)
        self.assertTrue(all(b.progress>a.progress for a,b in zip(selected,selected[1:])))
    def test_growth_not_decay_and_reeds(self):
        vine=self.v.add('mcl_core:vine',3);reeds=self.v.add('mcl_core:reeds',0)
        a=self.state();a.tokens[0,0,:4]=vine;e=Event(a,'D4',self.v,'vine')
        for i in range(4):
            b=self.state(a.tokens,i+1);b.tokens[0,0,:i+1]=self.air;e.append(b)
        self.assertEqual(len(e.select()),5);self.assertFalse(e.has_native_evolution())
        b=self.state();b.tokens[0,1,0]=reeds
        growth=Event(self.state(),'D4',self.v,'reeds');growth.append(b);self.assertTrue(growth.has_native_evolution())
    def test_reaction_requires_contact_and_change(self):
        lava=self.v.add('mcl_core:lava_flowing',0);water=self.v.add('mcl_core:water_source',0);cobble=self.v.add('mcl_core:cobble',0)
        left=self.state(origin=(-16,0,0));right=self.state(origin=(0,0,0))
        left.tokens[15,2,3]=lava;right.tokens[0,2,3]=water
        new=self.state(left.tokens,1,(-16,0,0));new.tokens[15,2,3]=cobble
        r=Reactions(self.v);m=r.observe({(-16,0,0):left,(0,0,0):right},{(-16,0,0):new,(0,0,0):self.state(right.tokens,1)})
        self.assertEqual(len(r.products),1);self.assertTrue(m[(-16,0,0)][15,2,3])
        r2=Reactions(self.v);r2.observe({(-16,0,0):new},{(-16,0,0):new});self.assertEqual(r2.products,[])
    def test_preexisting_stone_is_not_reaction(self):
        stone=self.v.add('mcl_core:stone',0);a=self.state();a.tokens[1,1,1]=stone
        r=Reactions(self.v);r.observe({(0,0,0):a},{(0,0,0):self.state(a.tokens,1)})
        self.assertEqual(r.products,[])
    def test_fire_merge_components(self):
        fire=self.v.add('mcl_fire:fire',0);a=self.state();a.tokens[1,1,1]=fire;a.tokens[3,1,1]=fire
        tracker=FireMerges(self.v,[[1,1,1],[3,1,1]]);tracker.observe({(0,0,0):a})
        self.assertEqual(len(tracker.merges),0);b=self.state(a.tokens,1);b.tokens[2,1,1]=fire
        tracker.observe({(0,0,0):b});self.assertEqual(len(tracker.merges),1)
    def test_adaptive_windows(self):
        self.assertEqual(next_plant_window(600,True),1200);self.assertEqual(next_plant_window(1200,True),1800)
        self.assertIsNone(next_plant_window(1800,True));self.assertIsNone(next_plant_window(600,False))
    def test_raw_roundtrip_and_short_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'data',self.v);scene=self.scene();a=self.state(origin=(-16,0,0));b=self.state(a.tokens,1,(-16,0,0))
            water=self.v.add('mcl_core:water_source',0);b.tokens[0,0,0]=water;b.light[2,2,2]=127
            store.raw.append(a,scene,'intervention_before');idx=store.raw.append(b,scene,'intervention_after')
            e=Event(a,'D1',self.v);e.append(b);row,_=store.save_event(e,scene)
            self.assertEqual(row['standard_states'],0);self.assertEqual(row['aux_short_states'],2)
            store.close();rebuilt=Reader(Path(tmp)/'data').frame(idx)
            np.testing.assert_array_equal(rebuilt.tokens,b.tokens);np.testing.assert_array_equal(rebuilt.light,b.light)
    def test_native_requires_manual_budget(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(['--output','unused','--native'])
    def test_plan_uses_independent_paired_instances(self):
        scenes=plan([42],['D4'],list('ABC'),1,120);self.assertEqual(len(scenes),9)
        for subtype in ('bamboo','reeds','vine'):
            matching=[s for s in scenes if s['subtype']==subtype]
            self.assertEqual(matching[0]['site'],matching[1]['site']);self.assertEqual(matching[1]['delay_seconds'],120)
        self.assertIsNone(budget(scenes)['wall_seconds_estimate'])
    def test_all_failed_ignitions_get_one_retry(self):
        class FailThenSucceed(DryWorld):
            calls=0
            def intervene(self,candidates,fallback=False):
                self.calls+=1;frames,clock=super().intervene(candidates,fallback)
                for r in clock['interventions']:r['success']=fallback
                return frames,clock
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'data',self.v);scene=self.scene();scene.update(category='D3')
            world=FailThenSucceed(scene,self.v);collect(world,scene,store,{})
            self.assertEqual(world.calls,2);self.assertEqual(scene['intervention_successes'],1)
            self.assertEqual(len(scene['fire_randomization']['draws']),20);store.close()

if __name__=='__main__':unittest.main()
