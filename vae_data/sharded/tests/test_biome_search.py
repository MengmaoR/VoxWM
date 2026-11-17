"""Quota-first lookup fixtures; no native world generation or physics run."""
import subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from vae_data.sharded.generate import choose_static_target,prepare_static
from vae_data.sharded.native import Native
from vae_data.sharded.protocol import BIOMES


def world_fixture():
    frame=SimpleNamespace(meta=dict(biome_group='MushroomIsland'))
    world=Mock()
    world.find_biome.return_value=dict(status='found',site=[192,-64],biome='MushroomIsland',query_y=8)
    world.locate.return_value=dict(status='located',surface_y=21,heights=[21]*5,biome='MushroomIslandShore')
    world.prepare_at.return_value=(frame,dict(chunks=[dict(stratum='shallow_underground')]*27))
    return world,frame


class BiomeSearchTests(unittest.TestCase):
    def test_selects_pending_joint_quota_before_looking_at_a_site(self):
        missing=[dict(biome='Forest',stratum='surface',remaining=8),dict(biome='MushroomIsland',stratum='deep_cave',remaining=2)]
        rng=Mock();rng.choice.return_value=1
        self.assertEqual(choose_static_target(missing,rng),missing[1])
        np.testing.assert_allclose(rng.choice.call_args.kwargs['p'],[.8,.2])

    def test_found_group_loads_once_then_checks_and_keeps_original_height_rule(self):
        world,frame=world_fixture();attempt={}
        target=dict(biome='MushroomIsland',stratum='shallow_underground',remaining=1)
        result=prepare_static(world,target,[0,0],np.random.default_rng(42),attempt)
        world.find_biome.assert_called_once_with('MushroomIsland',[0,0])
        world.locate.assert_called_once_with(192,-64)
        world.prepare_at.assert_called_once_with([192,-64,0],21)
        self.assertIs(result,frame);self.assertEqual(frame.meta['stratum'],'shallow_underground')
        self.assertEqual(frame.meta['static_selection']['actual_surface_biome'],'MushroomIslandShore')
        self.assertEqual(attempt['target_biome'],'MushroomIsland')

    def test_miss_and_loaded_biome_mismatch_never_capture_or_relabel(self):
        target=dict(biome='MushroomIsland',stratum='surface',remaining=1)
        for status in('biome_not_found','biome_unregistered','wrong_surface'):
            with self.subTest(status=status):
                world,_=world_fixture();attempt={}
                if status=='wrong_surface':world.locate.return_value['biome']='Plains'
                else:world.find_biome.return_value=dict(status=status)
                with self.assertRaises(ValueError):prepare_static(world,target,[0,0],np.random.default_rng(42),attempt)
                world.prepare_at.assert_not_called()
                if status!='wrong_surface':world.locate.assert_not_called()
                self.assertIn('biome_search',attempt)
        world,frame=world_fixture();frame.meta['biome_group']='Plains'
        with self.assertRaisesRegex(ValueError,'captured surface'):prepare_static(world,target,[0,0],np.random.default_rng(42),{})
        self.assertEqual(frame.meta['biome_group'],'Plains')

    def test_native_command_is_bounded_and_carries_the_existing_group_mapping(self):
        native=object.__new__(Native);native.checked=lambda result:result
        native.probe=SimpleNamespace(command=Mock(return_value=dict(status='biome_not_found',audit={},native_clock={})))
        self.assertEqual(native.find_biome('MushroomIsland',[16,-32]),dict(status='biome_not_found'))
        native.probe.command.assert_called_once_with('find_biome',biome_group='MushroomIsland',biome_groups=BIOMES,
            x=16,z=-32,resolution=64,max_checks=16384)

    def test_lua_search_finds_aliases_without_emergence_and_stops_on_miss(self):
        source=Path(__file__).parents[1]/'biome_search.lua'
        code="""
local search=dofile(arg[1]);assert(loadfile(arg[2]))
local names={[1]='MushroomIsland',[2]='MushroomIslandShore',[3]='Plains'}
local ids={MushroomIsland=1,MushroomIslandShore=2,Plains=3}
local mode='shore';local calls=0
local mt={registered_biomes={MushroomIsland={y_min=4,y_max=20},MushroomIslandShore={y_min=1,y_max=3},Plains={y_min=1,y_max=200}}}
mt.get_mapgen_setting=function(k)return ({water_level='0',mapgen_limit='31000',chunksize='5'})[k]end
mt.get_biome_id=function(name)return ids[name]end;mt.get_biome_name=function(id)return names[id]end
mt.emerge_area=function()error('noise search must not generate terrain')end
mt.get_biome_data=function(p)
    calls=calls+1;assert(p.x%16==8 and p.z%16==8)
    assert(p.y==8 or p.y==3)
    if p.x==8 and p.z==72 then
        if mode=='shore'and p.y==3 then return {biome=2}end
        if mode=='island'and p.y==8 then return {biome=1}end
    end
    return {biome=3}
end
local r={x=0,z=0,resolution=64,max_checks=16,biome_group='MushroomIsland',
    biome_groups={MushroomIsland={'MushroomIsland','MushroomIslandShore'},Plains={'Plains'}}}
for _,value in ipairs({'shore','island'})do
    mode=value;local result=search.find(mt,r)
    assert(result.status=='found'and result.checked_points==2)
    assert(result.site[1]==0 and result.site[2]==64)
    assert(result.biome==(value=='shore'and'MushroomIslandShore'or'MushroomIsland'))
end
mode='absent';calls=0;local result=search.find(mt,r)
assert(result.status=='biome_not_found'and result.checked_points==16 and calls==32)
r.biome_group='unregistered';calls=0;result=search.find(mt,r)
assert(result.status=='biome_unregistered'and calls==0)
"""
        with tempfile.TemporaryDirectory()as tmp:
            script=Path(tmp)/'search.lua';script.write_text(code)
            subprocess.run(['luajit',str(script),str(source),str(source.with_name('native.lua'))],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
