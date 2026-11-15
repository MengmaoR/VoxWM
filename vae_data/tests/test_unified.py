"""Small unit/integration tests; fixtures never enter the native pool."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vae_data.core import Vocabulary,Event,State,state_hash,source_actions,fire_actions,trim_actions,CONFIG,split_for_seed
from vae_data.storage import Dataset
from vae_data.engine import overlay

class UnifiedTests(unittest.TestCase):
    def setUp(self):self.v=Vocabulary();self.air=self.v.add('air',0)
    def state(self):return State(np.full((16,16,16),self.air,np.uint16),np.zeros((16,16,16),np.uint8),dict(world_seed=42,chunk_origin=[-16,0,0],timestamp=0,simulation_step=1,data_source='synthetic_fixture',biome_group='Plains'))
    def test_roundtrip_hash_independent_of_token_ids(self):
        s=self.state();names,p=self.v.decode(s.tokens);self.assertTrue(np.all(names=='air'))
        v2=Vocabulary();v2.add('other',99);a=v2.add('air',0)
        t=self.state();t.tokens[:]=a;self.assertEqual(state_hash(s,self.v),state_hash(t,v2))
    def test_suspended_and_attached_multisource(self):
        info=dict(attached=[dict(position=[i*5,0,0]) for i in range(10)],suspended=[dict(position=[i*5,10,0]) for i in range(10)])
        config=dict(CONFIG,source_count_range=[3,3])
        for mode,y in(('attached',0),('suspended',10)):
            a,m=source_actions(info,'D1',np.random.default_rng(42),config,mode)
            self.assertEqual(len(a),3);self.assertTrue(all(x['position'][1]==y for x in a))
    def test_fire_variable_count_probability(self):
        for count in(1,7,40):
            points=[dict(position=[i*5,0,0],under=[i*5,0,-1]) for i in range(count)]
            a,m=fire_actions(dict(fire=points,fire_legal_count=count),np.random.default_rng(42))
            self.assertTrue(a);self.assertEqual(len(m['draws']),len(m['sparse_candidates']));self.assertEqual(m['probability'],0.1)
    def test_root_preservation_and_cross_chunk(self):
        plant=dict(id='root',complete=True,subtype='bamboo',height=10,root=[0,0,14],members=[[0,0,14+i] for i in range(10)])
        a,m=trim_actions(dict(plants=dict(bamboo=[plant,plant])),'bamboo',np.random.default_rng(42))
        self.assertEqual(len(a),1);self.assertGreaterEqual(a[0]['retain_height'],1)
        if a[0]['cut_positions']:self.assertGreater(a[0]['cut_positions'][0][2],14)
    def test_vine_anchors_never_cut(self):
        p=dict(id='vine',complete=True,subtype='vine',members=[[0,0,0],[0,0,1],[0,0,2]],anchors=[[0,0,2]])
        for seed in range(10):
            a,_=trim_actions(dict(plants=dict(vine=[p])),'vine',np.random.default_rng(seed))
            self.assertNotIn([0,0,2],a[0]['cut_positions'])
    def test_intervention_progress_and_decay(self):
        s=self.state();vine=self.v.add('mcl_core:vine',3);s.tokens[0,0,0]=vine;e=Event(s,'D4',self.v,'vine')
        self.assertEqual(e.updates,0);t=self.state();e.append(t);self.assertFalse(e.has_native_evolution())
    def test_scene_serialization_and_synthetic_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Dataset(Path(tmp)/'data');scene=dict(scene_id='a',world_seed=42,origin=[-16,0,0],scene_types=['Ordinary'])
            d.add_scene(scene);d.flush();self.assertEqual(d.scenes(),[scene])
            with self.assertRaises(ValueError):d.append(self.state(),'Ordinary',scene,'r')
            d.close();d=Dataset(Path(tmp)/'data',True);self.assertEqual(d.scenes(),[scene]);d.close()
    def test_resume_truncates_uncommitted_array_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';d=Dataset(root);d.arrays['standard'][0].resize((1,16,16,16));d.close()
            d=Dataset(root,True);self.assertEqual(len(d.arrays['standard'][0]),0);d.close()
    def test_completed_job_with_status_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Dataset(Path(tmp)/'data');d.job('job','complete',dict(status='complete',category='D1'))
            self.assertTrue(d.completed('job'));d.close()
    def test_terminal_state_and_progress_selection(self):
        from vae_data.sampling import select_states
        water=self.v.add('mcl_core:water_flowing',2);first=self.state();event=Event(first,'D1',self.v)
        for i in range(1,14):
            state=self.state();state.tokens.ravel()[:i]=water;state.meta['timestamp']=i;event.append(state)
        chosen=select_states(event)
        self.assertEqual(len(chosen),8);self.assertTrue(np.array_equal(chosen[0].tokens,first.tokens))
        self.assertTrue(np.array_equal(chosen[-1].tokens,event.previous.tokens))
        self.assertEqual(len({s.tokens.tobytes() for s in chosen}),8)
    def test_world_seed_split_integrity(self):
        for seed in(0,42,191664964):self.assertEqual(split_for_seed(seed,[80,10,10]),split_for_seed(seed,[80,10,10]))
    def test_overlay_does_not_touch_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';(source/'builtin/game').mkdir(parents=True);(source/'builtin/game/init.lua').write_text('--original\n')
            (source/'bin').mkdir();target=overlay(source,Path(tmp)/'overlay')
            self.assertEqual((source/'builtin/game/init.lua').read_text(),'--original\n');self.assertIn('voxel_only.lua',(target/'builtin/game/init.lua').read_text())
            self.assertFalse((target/'builtin/game/init.lua').is_symlink())

if __name__=='__main__':unittest.main()
