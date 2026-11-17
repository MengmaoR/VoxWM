"""Small synthetic coordinate/height checks; fixtures are never Native evidence."""
import tempfile,unittest,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vae_p0.sampling import State
from vae_data.sharded.protocol import Frame,Chain,Vocabulary,assemble
from vae_data.sharded.static_height import height_bounds,HeightSearch,rejection,sample,ratios,validate_static
from vae_data.sharded.storage import Writer,static_position_key
from vae_data.sharded.tests.test_sharded import frame,water_chain
from vae_data.sharded.native import Native
from vae_data.sharded.validate import validate_chain


class FixtureWorld:
    def __init__(self,name='mcl_core:stone',air=None):
        self.v=Vocabulary();self.name=name;self.air=air;self.origins=[]
    def prepare_at(self,origin,surface,static_window=False):
        self.origins.append(origin);c=origin[2]+8
        tokens=np.full((48,)*3,self.v.add(self.name,0),np.uint16)
        if self.air is not None:
            center=tokens[16:32,16:32,16:32].copy();center.ravel()[:int(self.air(c)*4096)]=self.v.add('air',0)
            tokens[16:32,16:32,16:32]=center
        meta=frame(self.v,len(self.origins)).meta
        meta.update(sampling_window_kind='static'if static_window else'dynamic',center_origin=list(origin),
            region_origin=(np.array(origin)-16).tolist(),sample_surface_y=surface,rollout_id=str(len(self.origins)))
        return Frame(tokens,np.zeros(tokens.shape,np.uint8),meta),dict(chunks=[dict(stratum='native_fixture')]*27)


def location(surface):return dict(status='located',surface_y=surface,heights=[surface]*5)


class StaticHeightTests(unittest.TestCase):
    def test_complete_region_boundaries_and_all_requested_integer_heights(self):
        for c in(-39,-38,-37,0,8,9,10,15,16,17,23,24,169,170,171):
            with self.subTest(c=c):
                w=FixtureWorld()
                if c in(-39,171):
                    with self.assertRaisesRegex(ValueError,'overworld'):w.prepare_at([0,0,c-8],32,True)
                else:
                    f,_=w.prepare_at([0,0,c-8],32,True)
                    self.assertEqual(f.meta['region_origin'][2],c-24)
                    self.assertGreaterEqual(c-24,-62);self.assertLessEqual(c+23,193)

    def test_intervals_and_uniform_without_replacement_include_every_integer(self):
        self.assertEqual(height_bounds(32,'surface'),(25,40))
        self.assertEqual(height_bounds(32,'elevated_complex'),(41,170))
        self.assertEqual(height_bounds(32,'shallow_underground'),(8,24))
        self.assertEqual(height_bounds(32,'deep_cave'),(-38,0))
        search=HeightSearch(32,'shallow_underground',np.random.default_rng(42))
        self.assertEqual({search.draw()for _ in range(17)},set(range(8,25)))
        self.assertIsNone(search.draw())

    def test_offset_assembly_has_no_gap_overlap_or_coordinate_rounding(self):
        for c in(-38,9,10,17,23,170):
            origin=np.array([0,16,c-8]);low=origin-16;chunks={}
            for x in range(3):
                for n in range(3):
                    for u in range(3):
                        o=low+np.array([x,n,u])*16
                        values=np.broadcast_to(np.arange(o[2],o[2]+16)[None,None,:]+100,(16,16,16)).astype(np.uint16).copy()
                        chunks[tuple(o)]=State(values,np.zeros((16,)*3,np.uint8),{})
            meta=frame(Vocabulary(),0).meta;meta['sampling_window_kind']='static'
            f=assemble(chunks,origin,meta)
            np.testing.assert_array_equal(f.tokens[0,0],np.arange(c-24,c+24)+100)
            np.testing.assert_array_equal(f.center[0,0],np.arange(c-8,c+8)+100)
            with self.assertRaises(ValueError):assemble(chunks,[0,16,c-8+.5],meta)

    def test_dynamic_grid_stays_aligned_and_fractional_origins_fail(self):
        w=FixtureWorld();f,_=w.prepare_at([0,0,1],32,True)
        with self.assertRaises(ValueError):Frame(f.tokens,f.light,dict(f.meta,sampling_window_kind='dynamic'))
        with self.assertRaises(ValueError):Frame(f.tokens,f.light,dict(f.meta,center_origin=[0,0,1.5]))
        with self.assertRaises(ValueError):Frame(f.tokens,f.light,dict(f.meta,region_origin=[-16.,-16.,-15.]))
        v,chain=water_chain()
        self.assertTrue(validate_chain(chain,v,False))
        for item in chain.frames:item.meta.update(center_origin=[0,0,1],region_origin=[-16,-16,-15],sampling_window_kind='static')
        with self.assertRaisesRegex(ValueError,'Dynamic center'):validate_chain(chain,v,False)
        native=object.__new__(Native);native.probe=Mock()
        for c in(-39,171):
            with self.assertRaises(ValueError):native.prepare_at([0,0,c-8],32,True)
        native.probe.command.assert_not_called()

    def test_elevated_air_boundaries_and_strict_surface_separation(self):
        self.assertEqual(rejection('elevated_complex',.099,0),'air_too_low')
        self.assertIsNone(rejection('elevated_complex',.10,0))
        self.assertIsNone(rejection('elevated_complex',.979,0))
        self.assertEqual(rejection('elevated_complex',.98,0),'air_too_high')
        self.assertEqual(height_bounds(32,'elevated_complex')[0]-8,33)

    def test_elevated_empty_height_forces_only_lower_retries(self):
        w=FixtureWorld(air=lambda c:1. if c>41 else .5)
        rng=Mock();rng.integers.side_effect=[170,39,0]
        f,_=sample(w,[0,0],'elevated_complex',rng,location(32))
        self.assertEqual([o[2]+8 for o in w.origins],[170,80,41])
        self.assertEqual(f.meta['height_attempt_count'],3);self.assertTrue(validate_static(f,w.v))

    def test_underground_solid_air_void_and_ignore_are_distinct(self):
        for stratum in('shallow_underground','deep_cave'):
            for name in('mcl_core:stone','mcl_core:dirt','mcl_core:bedrock','air'):
                w=FixtureWorld(name);f,_=sample(w,[0,0],stratum,np.random.default_rng(42),location(32))
                self.assertTrue(validate_static(f,w.v));self.assertEqual(f.meta['void_fraction'],0)
        w=FixtureWorld('mcl_core:void');attempt={}
        with self.assertRaisesRegex(ValueError,'height_candidates_exhausted'):
            sample(w,[0,0],'shallow_underground',np.random.default_rng(42),location(-29),attempt)
        self.assertEqual(len(w.origins),2);self.assertEqual(len({tuple(o)for o in w.origins}),2)
        w=FixtureWorld('ignore')
        with self.assertRaisesRegex(ValueError,'Data integrity'):sample(w,[0,0],'deep_cave',np.random.default_rng(42),location(32))

    def test_validator_recomputes_ratios_ranges_and_retry_history(self):
        w=FixtureWorld();f,_=sample(w,[0,0],'shallow_underground',np.random.default_rng(42),location(32))
        self.assertTrue(validate_static(f,w.v))
        for key,value in(('air_fraction',.2),('void_fraction',1.),('region_height_max',999),('height_candidate_lower',9),('selected_chunk_center_y',99)):
            original=f.meta[key];f.meta[key]=value
            with self.assertRaises(ValueError):validate_static(f,w.v)
            f.meta[key]=original
        f.tokens[0,0,0]=w.v.add('ignore',0)
        with self.assertRaisesRegex(ValueError,'Data integrity'):validate_static(f,w.v)

    def test_static_position_is_unique_across_shards_even_if_state_changes(self):
        with tempfile.TemporaryDirectory()as tmp:
            w=FixtureWorld();rng=Mock();rng.integers.return_value=9
            a,_=sample(w,[0,0],'shallow_underground',rng,location(32))
            with Writer(tmp,0,fixture=True)as writer:self.assertTrue(writer.submit(Chain('Static',[a]),w.v))
            w.name='mcl_core:dirt';b,_=sample(w,[0,0],'shallow_underground',rng,location(32))
            self.assertEqual(static_position_key(a),static_position_key(b))
            with Writer(tmp,1,fixture=True)as writer:
                self.assertFalse(writer.submit(Chain('Static',[b]),w.v))
                rng.integers.return_value=10;c,_=sample(w,[0,0],'shallow_underground',rng,location(32))
                self.assertTrue(writer.submit(Chain('Static',[c]),w.v))
                self.assertEqual(len(writer.arrays['snapshots']['tokens']),1)

    def test_static_position_claim_rolls_back_with_failed_sample_commit(self):
        with tempfile.TemporaryDirectory()as tmp:
            import sqlite3
            w=FixtureWorld();f,_=sample(w,[0,0],'shallow_underground',np.random.default_rng(42),location(32))
            with Writer(tmp,0,fixture=True)as writer:
                writer.db.executescript("CREATE TRIGGER reject_fixture BEFORE INSERT ON samples BEGIN SELECT RAISE(ABORT,'fixture rollback');END;")
                with self.assertRaises(sqlite3.IntegrityError):writer.submit(Chain('Static',[f]),w.v)
                self.assertEqual(writer.db.execute('SELECT COUNT(*) FROM static_registry.positions').fetchone()[0],0)
                self.assertEqual(len(writer.arrays['snapshots']['tokens']),0)
                writer.db.execute('DROP TRIGGER reject_fixture')
                self.assertTrue(writer.submit(Chain('Static',[f]),w.v))

    def test_exhausted_site_restarts_native_world_with_new_seed(self):
        from vae_data.sharded.generate import generate_shard
        with tempfile.TemporaryDirectory()as tmp:
            writer=Mock();writer.__enter__=Mock(return_value=writer);writer.__exit__=Mock(return_value=False)
            writer.path=Path(tmp);writer.missing.return_value=dict(categories=dict(Static=1));writer.export.return_value={}
            missing=dict(static=[dict(biome='Plains',stratum='deep_cave',remaining=1)])
            w1,w2=Mock(),Mock();w1.__enter__=Mock(return_value=w1);w2.__enter__=Mock(return_value=w2);w1.__exit__=Mock();w2.__exit__=Mock()
            state=frame(Vocabulary(),0)
            with patch('vae_data.sharded.generate.Writer',return_value=writer),patch('vae_data.sharded.generate.Native',side_effect=[w1,w2])as native,patch('vae_data.sharded.generate.choose_category',return_value=('Static',missing)),patch('vae_data.sharded.generate.prepare_static',side_effect=[ValueError('height_candidates_exhausted'),state]):
                import time
                result=generate_shard(tmp,0,False,time.monotonic()+60,'unused','auto',42,2,8)
            self.assertEqual(result['status'],'incomplete');self.assertEqual(native.call_count,2)
            self.assertNotEqual(native.call_args_list[0].args[0],native.call_args_list[1].args[0])
            w1.__exit__.assert_called_once();self.assertEqual(writer.submit.call_count,1)

    def test_forceload_covers_every_intersecting_engine_mapblock(self):
        source=Path(__file__).parents[1]/'scene_geometry.lua'
        code="""local g=dofile(arg[1]);for _,c in ipairs({-38,-37,0,8,9,10,15,16,17,23,24,169,170})do
local low={x=-16,z=-16,y=c-24};local high={x=31,z=31,y=c+23};local blocks=g.mapblocks(low,high);local seen={}
for _,p in ipairs(blocks)do seen[p.x..','..p.y..','..p.z]=true end
for x=low.x,high.x do for z=low.z,high.z do for y=low.y,high.y do
local key=(math.floor(x/16)*16)..','..(math.floor(y/16)*16)..','..(math.floor(z/16)*16);assert(seen[key])end end end
assert(#blocks==((c-24)%16==0 and 27 or 36))end"""
        with tempfile.TemporaryDirectory()as tmp:
            script=Path(tmp)/'blocks.lua';script.write_text(code)
            subprocess.run(['luajit',str(script),str(source)],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
