"""Aligned center-only geometry fixtures; never native evidence."""
from pathlib import Path
import subprocess,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from vae_data.sharded.protocol import Frame,Vocabulary
from vae_data.sharded.centers import components,choose_height,pool_kind,prepare_dynamic,spatial_gate,stable_pool,height_statistics
from vae_data.sharded.planning.fixtures import fixture


def surface():
    f,v=fixture();stone=v.add('mcl_core:stone',0);air=v.add('air',0);f.tokens[:]=air;f.tokens[:,:,:21]=stone
    f.meta.update(height_type='surface',terrain_names=['mcl_core:stone'],scene_selection=dict(height_type='surface',solid_surface_heights=np.full((16,16),4).tolist()))
    return f,v

class CenterTests(unittest.TestCase):
    def test_air_65_not_sum_of_disconnected_or_neighbor_voxels(self):
        mask=np.zeros((16,)*3,bool);mask[0:4,0:4,0:4]=True
        self.assertEqual(components(mask)[0],64);mask[4,0,0]=True;self.assertEqual(components(mask)[0],65)
        split=np.zeros_like(mask);split[:4,:4,:4]=True;split[10:14,10:14,10:14]=True
        self.assertEqual(components(split)[0],64)
        f,v=surface();f.tokens[:]=v.add('mcl_core:stone',0);f.tokens[:16,:,:]=v.add('air',0)
        ok,info=spatial_gate(f,v);self.assertFalse(ok);self.assertEqual(info['largest_connected_air_component'],0)
    def test_surface_requires_actual_solid_air_interface_not_lake(self):
        f,v=surface();self.assertTrue(spatial_gate(f,v)[0]);water=v.add('mcl_core:water_source',0)
        f.tokens[:,:,21:25]=water
        ok,info=spatial_gate(f,v);self.assertFalse(ok);self.assertEqual(info['surface_boundary_columns'],0)
        f.tokens[:]=water;self.assertFalse(spatial_gate(f,v)[0])
    def test_cave_is_real_connected_underground_air_not_fixed_depth_stone(self):
        f,v=surface();f.meta.update(height_type='cave',scene_selection=dict(height_type='cave',solid_surface_heights=np.full((16,16),32).tolist()))
        f.tokens[:]=v.add('mcl_core:stone',0);self.assertFalse(spatial_gate(f,v)[0])
        f.tokens[20:25,20:25,20:25]=v.add('air',0);ok,info=spatial_gate(f,v);self.assertTrue(ok);self.assertEqual(info['largest_underground_air_component'],125)
        f.meta['scene_selection']['solid_surface_heights']=np.full((16,16),-20).tolist();self.assertFalse(spatial_gate(f,v)[0])
    def test_pool_19_in_center_stable_flowing_allowed_no_2d_or_volume_rule(self):
        for kind in('water','lava'):
            f,v=surface();flow=v.add('mcl_core:'+kind+'_flowing',5);f.tokens[16:19,16:19,22:25]=flow
            g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=1,sim_time=1.1))
            n,mask=stable_pool(f,g,v,kind);self.assertEqual(n,27);self.assertEqual(mask.sum(),27)
            g.tokens[16:19,16:19,22:25]=v.add('mcl_core:'+kind+'_flowing',6);self.assertEqual(stable_pool(f,g,v,kind)[0],0)
            f.tokens[:]=v.add('air',0);g.tokens[:]=v.add('air',0);f.tokens[:16,:,:]=flow;g.tokens[:16,:,:]=flow
            self.assertEqual(stable_pool(f,g,v,kind)[0],0)
    def test_pool_connectivity_never_joins_through_neighbor_chunks(self):
        f,v=surface();water=v.add('mcl_core:water_source',0);f.tokens[:]=v.add('air',0)
        f.tokens[16:18,16:19,20:23]=water #18
        f.tokens[30:32,16:19,20:23]=water #18; neighbor bridge must not join them
        f.tokens[16:32,:16,20:23]=water
        g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=1,sim_time=1.1))
        largest,mask=stable_pool(f,g,v,'water');self.assertEqual(largest,18);self.assertFalse(mask.any())
    def test_no_target_voxel_hint_only_ac_request_natural_pool(self):
        class World:
            def find_centers(self,site,height,pool):
                self.request=(height,pool);return dict(candidates=[dict(center=[160,-32,0],surface_y=4,height_type=height)],probe_domain={},rejected={})
            def prepare_at(self,origin,height):
                f,v=surface();f=Frame(f.tokens,f.light,dict(f.meta,center_origin=origin,region_origin=(np.array(origin)-16).tolist()));return f,{}
        for category,sub,kind in(('D1','',''),('D2','',''),('D3','',''),('D4','B',''),('D4','A','water'),('D4','C','lava')):
            w=World();f,_=prepare_dynamic(w,[160,-32],category,np.random.default_rng(4),sub,'surface');self.assertEqual(w.request,('surface',kind));self.assertEqual(f.meta['center_origin'],[160,-32,0]);self.assertTrue(np.all(np.array(f.meta['center_origin'])%16==0))
    def test_rejected_center_debug_trace_does_not_reach_planner(self):
        from vae_data.sharded.generate import collect
        f,v=surface();f.meta['scene_selection'].update(probe_qualified=False,diagnostic_rejection_reason='no_pool')
        w=SimpleNamespace(v=v,seed=42,center=[0,0,0]);trace=SimpleNamespace(observe=lambda *a:None,finish=lambda **kw:None)
        with patch('vae_data.sharded.generate.Planner')as planner:
            chain,result=collect(w,f,'D4',np.random.default_rng(4),'A',trace=trace)
            self.assertIsNone(chain);planner.assert_not_called();self.assertEqual(result['reason'],'center_probe_no_pool')
    def test_exact_pool_threshold_accepts_source_and_flowing(self):
        f,v=surface();source=v.add('mcl_core:water_source',0);flow=v.add('mcl_core:water_flowing',5)
        f.tokens[:]=v.add('air',0);f.tokens[16:19,16:19,16:18]=source;f.tokens[19,16,16]=flow
        g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=1,sim_time=1.1))
        largest,mask=stable_pool(f,g,v,'water');self.assertEqual(largest,19);self.assertEqual(mask.sum(),19)
    def test_filter_precedes_all_point_planning(self):
        from vae_data.sharded.generate import collect
        f,v=surface();f.tokens[:]=v.add('mcl_core:stone',0);w=SimpleNamespace(v=v,seed=42,center=[0,0,0])
        with patch('vae_data.sharded.generate.Planner')as planner:
            chain,result=collect(w,f,'D1',np.random.default_rng(4));self.assertIsNone(chain);planner.assert_not_called();self.assertEqual(result['reason'],'center_connected_air_below_65')
    def test_height_opportunities_and_qualified_accepted_counts(self):
        rng=np.random.default_rng(42)
        self.assertEqual({choose_height('D1',rng)for _ in range(10)},{'surface'});self.assertEqual({choose_height('D3',rng)for _ in range(10)},{'surface'})
        self.assertEqual({choose_height('D2',rng)for _ in range(32)},{'surface','cave'})
        stats=height_statistics([dict(category='D2',height_type='cave',scene_spatial_qualified=True,planning_qualified=True,status='success'),dict(category='D2',height_type='surface',status='rejected')])
        self.assertEqual(stats['D2']['cave'],dict(attempts=1,spatial_qualified=1,qualified=1,accepted=1));self.assertEqual(stats['D2']['surface']['attempts'],1)
    def test_lua_native_geometry_uses_same_six_neighbor_counts(self):
        source=Path(__file__).parents[1]/'scene_geometry.lua'
        code="""local g=dofile(arg[1]);local m={};for i=1,64 do m[i]=true end;assert(g.largest(m)==64);m[65]=true;assert(g.largest(m)==65);assert(not g.terrain({walkable=false,liquidtype='source',groups={water=3}}));assert(g.terrain({walkable=true,liquidtype='none',groups={stone=1}}));assert(not g.terrain({walkable=true,liquidtype='none',groups={leaves=1}}))"""
        with tempfile.TemporaryDirectory()as temp:
            script=Path(temp)/'geometry.lua';script.write_text(code);subprocess.run(['luajit',str(script),str(source)],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
