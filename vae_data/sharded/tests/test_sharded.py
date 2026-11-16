"""Tiny unit fixtures, explicitly isolated from all native dataset outputs."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib,io,json,multiprocessing,tempfile,unittest
from unittest.mock import patch
import numpy as np
from vae_data.sharded.protocol import Frame,Chain,Vocabulary,BIOMES,STRATA,split_for_seed,d4_targets,assemble,kind_mask
from vae_data.sharded.sampling import Collector,Registry
from vae_data.sharded.reachability import pool_components
from vae_data.history.reachability_pre_exhaustive import potential,plan
from vae_data.sharded.storage import Writer,initialize,globalize,rows_at
from vae_data.sharded.validate import validate_chain,check_shard,quota_errors


def frame(v,step,tokens=None,seed=42,rollout='fixture'):
    air=v.add('air',0)
    a=np.full((48,)*3,air,np.uint16)if tokens is None else tokens.copy()
    return Frame(a,np.zeros(a.shape,np.uint8),dict(data_source='synthetic_fixture',center_origin=[0,0,0],region_origin=[-16,-16,-16],
        simulation_step=step,sim_time=step/24,simulation_hz=24,frame_index=step,relative_sim_time=step/24,absolute_sim_time=step/24,native_simulation_step=step,engine_us=step*100,rollout_id=rollout,world_seed=seed,
        split=split_for_seed(seed,[80,10,10]),biome_group='Plains',stratum='surface',event_roles=[]))


def water_chain(seed=42):
    v=Vocabulary();states=[]
    for step in range(1,6):
        f=frame(v,step,seed=seed);f.tokens[20,20,20]=v.add('mcl_core:water_flowing',step);f.meta.update(candidate_index=step-1,candidate_count=5);states.append(f)
    return v,Chain('D1',states)


def overlap_fixture():
    v=Vocabulary();anchor=frame(v,0);water=v.add('mcl_core:water_source',0);lava=v.add('mcl_core:lava_source',0);stone=v.add('mcl_core:stone',0)
    for i in range(6):anchor.tokens[20+i,20,20]=water;anchor.tokens[20+i,20,21]=lava
    frames=[];current=anchor.tokens.copy();events={}
    for step in range(1,7):
        if step==1:current[25,20,20]=v.add('mcl_core:water_flowing',1)
        if step in(2,3,5):
            index={2:0,3:1,5:2}[step];p=[4+index,4,4];current[20+index,20,20]=stone
            events[step]=[dict(position=p,before='mcl_core:water_flowing'if step==5 else'mcl_core:water_source',after='mcl_core:stone',confirmed=True,
                contact=dict(water=p,lava=[p[0],p[1],p[2]+1],water_node='mcl_core:water_source',lava_node='mcl_core:lava_source'),clock=dict(engine_us=step*100-50))]
        if step in(4,6):current[22 if step==4 else 23,20,20]=v.add('mcl_core:water_flowing',step)
        frames.append(frame(v,step,current))
    return v,anchor,frames,events


def parallel_writer(root,shard):
    v,chain=water_chain(42+shard)
    with Writer(root,shard,fixture=True)as writer:
        assert writer.submit(chain,v)
    return shard

class ShardedTests(unittest.TestCase):
    def test_half_integer_chunk_origin_is_not_silently_rounded(self):
        v=Vocabulary();f=frame(v,0);meta=dict(f.meta,center_origin=[.5,0,0])
        with self.assertRaises(ValueError):Frame(f.tokens,f.light,meta)
    def test_neighbor_change_does_not_add_candidate_or_reset_clock(self):
        v=Vocabulary();start=frame(v,0);monitor=Collector(start,'D1',v)
        water=v.add('mcl_core:water_flowing',2)
        for step in range(1,49):
            f=frame(v,step);f.tokens[2,2,2]=water
            stopped=monitor.observe(f)
        self.assertTrue(stopped);self.assertEqual(len(monitor.registry.frames),1)
        self.assertEqual(monitor.last_change,0);self.assertEqual(monitor.reason,'center_no_change_2s')
    def test_exact_five_random_contiguous_window_and_ten_cap(self):
        v=Vocabulary();monitor=Collector(frame(v,0),'D2',v)
        for step in range(1,10):
            f=frame(v,step);f.tokens[20,20,20]=v.add('mcl_core:lava_flowing',step)
            stopped=monitor.observe(f)
        self.assertTrue(stopped);self.assertEqual(len(monitor.registry.frames),10)
        chain=monitor.select(np.random.default_rng(4));self.assertEqual(len(chain.frames),5)
        steps=[f.step for f in chain.frames];self.assertEqual(steps,list(range(steps[0],steps[0]+5)))
        self.assertTrue(validate_chain(chain,v,False))
    def test_gapped_candidate_window_is_rejected(self):
        v,chain=water_chain();chain.frames[-1].meta['candidate_index']=9
        with self.assertRaises(ValueError):validate_chain(chain,v,False)
    def test_repeat_center_states_cannot_pad_candidates(self):
        v=Vocabulary();initial=frame(v,0);monitor=Collector(initial,'D1',v);water=v.add('mcl_core:water_flowing',1)
        for step in range(1,8):
            f=frame(v,step)
            if step%2:f.tokens[20,20,20]=water
            monitor.observe(f)
        self.assertEqual(len(monitor.registry.frames),2);self.assertIsNone(monitor.select(np.random.default_rng(4)))
    def test_chain_cannot_mix_chunks(self):
        v,chain=water_chain();chain.frames[-1].meta['center_origin']=[16,0,0]
        with self.assertRaises(ValueError):validate_chain(chain,v,False)
    def test_overlapping_reactions_share_physical_frames(self):
        v,anchor,frames,events=overlap_fixture();monitor=Collector(anchor,'D4',v)
        for f in frames:monitor.observe(f,events.get(f.step,[]))
        shared=monitor.registry.by_key['fixture:2'];self.assertEqual(len(shared.meta['event_roles']),2)
        roles={x['role']for x in monitor.registry.by_key['fixture:3'].meta['event_roles']}
        self.assertEqual(roles,{'post','reaction'})
        self.assertEqual({x['role']for x in monitor.registry.by_key['fixture:4'].meta['event_roles']},{'post','pre'})
        chain=monitor.select(np.random.default_rng(0));self.assertTrue(validate_chain(chain,v,False))
        self.assertEqual(len({f.key for f in chain.frames}),5);self.assertNotIn(anchor.key,[f.key for f in chain.frames])
    def test_d4_old_stone_and_neighbor_reaction_do_not_qualify(self):
        v=Vocabulary();anchor=frame(v,0);monitor=Collector(anchor,'D4',v);stone=v.add('mcl_core:stone',0)
        for step in range(1,8):
            f=frame(v,step);f.tokens[2,2,2]=stone
            monitor.observe(f,[dict(confirmed=True,position=[-14,-14,-14])])
        self.assertFalse(monitor.events);self.assertIsNone(monitor.select(np.random.default_rng(1)))
    def test_incomplete_d4_event_is_rejected(self):
        v,anchor,frames,events=overlap_fixture();monitor=Collector(anchor,'D4',v)
        for f in frames:monitor.observe(f,events.get(f.step,[]))
        chain=monitor.select(np.random.default_rng(0));chain.events=[]
        with self.assertRaises(ValueError):validate_chain(chain,v,False)
    def test_one_physical_frame_cannot_have_conflicting_arrays(self):
        v=Vocabulary();registry=Registry();f=frame(v,1);registry.add(f);other=frame(v,1);other.tokens[20,20,20]=v.add('mcl_core:stone',0)
        with self.assertRaises(ValueError):registry.add(other)
    def test_48_region_assembly_uses_exact_world_neighbors(self):
        from vae_p0.sampling import State
        chunks={};n=0
        for x in(-16,0,16):
            for y in(-16,0,16):
                for z in(-16,0,16):chunks[x,y,z]=State(np.full((16,)*3,n,np.uint16),np.zeros((16,)*3,np.uint8),{});n+=1
        meta=frame(Vocabulary(),0).meta;r=assemble(chunks,[0,0,0],meta)
        self.assertTrue(np.all(r.center==13));self.assertTrue(np.all(r.tokens[:16,:16,:16]==0))
        chunks.pop((-16,-16,-16))
        with self.assertRaises(ValueError):assemble(chunks,[0,0,0],meta)
    def test_gravity_ranges_and_obstacles(self):
        space=np.ones((24,24,24),bool);space[:,:,0]=False;space[12,:,:]=False
        reached=potential(space,(8,12,1),3)
        self.assertTrue(reached[5,12,1]);self.assertFalse(reached[4,12,1]);self.assertFalse(reached[13].any())
        downward=potential(space,(8,12,15),3);self.assertTrue(downward[8,12,1])
    def test_isolated_springs_are_not_a_static_pool(self):
        mask=np.zeros((48,)*3,bool)
        for i in range(4):mask[4+i*5,5,20]=True
        self.assertFalse(pool_components(mask).any())
        mask[20:22,20:22,20]=True;self.assertEqual(int(pool_components(mask).sum()),4)
    def test_d4_three_geometry_subtypes(self):
        rng=np.random.default_rng(42)
        for subtype,kind in(('A','water'),('C','lava'),('B',None)):
            v=Vocabulary();f=frame(v,0);f.tokens[:,:,:17]=v.add('mcl_core:stone',0)
            if kind:f.tokens[20:25,20:25,17]=v.add('mcl_core:'+kind+'_source',0)
            proposal=plan(f,v,subtype,rng,budget=16);self.assertIsNotNone(proposal,subtype)
            self.assertGreater(proposal['predicted_center_contact_cells'],0)
            if subtype=='C':self.assertTrue(all(a['type']=='water'for a in proposal['actions']))
    def test_strict_quota_recipe_and_subtype_rotation(self):
        rows=[dict(category='Static',biome_group=b,stratum=s)for b in BIOMES for s,n in STRATA.items()for _ in range(n)]
        rows.extend(dict(category=k)for k in('D1','D2','D3','D4')for _ in range(125));chains=[]
        for k in('D1','D2','D3'):chains.extend(dict(category=k,subtype='')for _ in range(25))
        chains.extend(dict(category='D4',subtype=s)for s,n in d4_targets(0).items()for _ in range(n))
        self.assertEqual(len(rows),1000);self.assertFalse(quota_errors(rows,chains,0));rows[0]['stratum']='deep_cave'
        self.assertTrue(quota_errors(rows,chains,0));self.assertEqual(sum(d4_targets(1).values()),25)
    def test_atomic_chain_commit_resume_and_independent_check(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'data';v,chain=water_chain()
            with Writer(root,0,fixture=True)as writer:
                self.assertTrue(writer.submit(chain,v));self.assertFalse(writer.submit(chain,v));writer.export();path=writer.path
                writer.arrays['snapshots']['tokens'].resize((6,48,48,48))
            with Writer(root,0,True,fixture=True)as writer:
                self.assertEqual(writer.counts()['D1'],5);self.assertEqual(len(writer.arrays['snapshots']['tokens']),5);writer.export()
                check=check_shard(root,writer.path,False,False);self.assertTrue(check['valid'],check)
                self.assertFalse(check_shard(root,writer.path,True,False)['valid'])
            self.assertFalse((path/'READY').exists())
    def test_atomic_publication_and_ready_immutability(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'data';v,chain=water_chain()
            with Writer(root,0,fixture=True)as writer:
                writer.submit(chain,v)
                # QA logic has separate quota/event tests. This tests atomic publication only.
                with patch('vae_data.sharded.validate.check_shard',return_value=dict(valid=True)):
                    writer.publish()
            final=root/'shards/shard_0000';ready=(final/'READY').read_text()
            self.assertFalse((root/'shards/.shard_0000.partial').exists())
            with self.assertRaises(FileExistsError):Writer(root,0,True,fixture=True)
            self.assertEqual((final/'READY').read_text(),ready)
    def test_native_generator_rejects_fixture_pool(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'data';initialize(root,True)
            with self.assertRaises(ValueError):initialize(root)
    def test_parallel_shards_share_one_append_only_vocabulary(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'data';initialize(root,True)
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))as pool:
                self.assertEqual(sorted(pool.map(parallel_writer,[str(root)]*2,[0,1])),[0,1])
            entries=Vocabulary(root/'vocabulary.json').entries
            self.assertEqual(len(entries),len(set(entries)))
            for shard in(0,1):
                check=check_shard(root,root/'shards'/f'.shard_{shard:04d}.partial',False,False);self.assertTrue(check['valid'],check)
    def test_keyboard_interrupt_stops_owned_workers_for_resume(self):
        from vae_data.__main__ import main
        with tempfile.TemporaryDirectory()as tmp:
            args=['vae_data','generate','--dataset',tmp,'--wall-seconds=1']
            with patch('sys.argv',args),patch('subprocess.Popen')as popen,patch('subprocess.check_output',return_value='700001 1\n700002 700001\n'),patch('os.kill')as kill,patch('time.sleep'):
                popen.return_value.pid=700001;popen.return_value.wait.side_effect=[KeyboardInterrupt(),0]
                self.assertEqual(main(),130);self.assertEqual(kill.call_count,4)
            self.assertTrue((Path(tmp)/'interrupted.json').exists())
    def test_cli_native_caps_and_budgets(self):
        from vae_data.sharded.cli import parser
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parser(['native-smoke','--dataset','unused','--wall-seconds','1','--samples-per-category','3'])
        args=parser(['generate','--dataset','unused','--wall-seconds','1','--shard','4','--count','2','--workers','2','--resume'])
        self.assertEqual(args.shard,4);self.assertTrue(args.resume)

if __name__=='__main__':unittest.main()
