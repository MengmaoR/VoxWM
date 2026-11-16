"""Small state/transition fixtures; no engine, no generation and no model."""
from pathlib import Path
import json,subprocess,tempfile,unittest
import numpy as np
from vae_data.sharded.protocol import Frame,Vocabulary,target_changes,target_change_mask
from vae_data.sharded.planning.fixtures import fixture
from vae_data.sharded.transitions import classify,definition

class NativeStateTests(unittest.TestCase):
    def test_leaves_fire_and_age_have_different_meanings(self):
        self.assertEqual(classify(('air',0),('mcl_fire:fire',0)),'air_to_fire')
        self.assertEqual(classify(('mcl_core:leaves',1),('mcl_fire:fire',0)),'leaves_to_fire')
        self.assertEqual(classify(('mcl_core:leaves',1),('air',0)),'leaves_to_air')
        self.assertEqual(classify(('mcl_fire:fire',0),('air',0)),'fire_to_air')
        self.assertEqual(classify(('mcl_fire:fire',0),('mcl_fire:fire',8)),'fire_age_update')
    def test_source_loss_and_dirt_lava_are_not_ordinary_flow(self):
        self.assertEqual(classify(('mcl_core:lava_source',0),('air',0)),'source_loss_requires_write_evidence')
        self.assertEqual(classify(('mcl_core:dirt',0),('mcl_core:lava_flowing',0)),'solid_to_liquid_requires_write_evidence')
        self.assertEqual(classify(('mcl_core:lava_flowing',2),('air',0)),'flowing_liquid_recession_observed')
        self.assertEqual(classify(('mcl_mushrooms:mushroom_red',0),('mcl_core:lava_flowing',0)),'lava_replaced_lava_destructible_node_observed')
    def test_d3_age_and_neighbor_changes_cannot_fill_five_states(self):
        v=Vocabulary();f,_=fixture();f.tokens[:]=v.add('air',0);f.meta['flammable_names']=['mcl_core:leaves']
        f.tokens[20,20,20]=v.add('mcl_fire:fire',0);f.tokens[21,20,20]=v.add('mcl_core:leaves',0)
        g=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=1,sim_time=.2))
        g.tokens[20,20,20]=v.add('mcl_fire:fire',8);self.assertEqual(target_changes(f,g,v,'D3'),0)
        g.tokens[21,20,20]=v.add('mcl_fire:fire',0);self.assertEqual(target_changes(f,g,v,'D3'),1)
        self.assertTrue(target_change_mask(f,g,v,'D3')[21,20,20])
        g.tokens[2,2,2]=v.add('mcl_fire:fire',0);self.assertEqual(target_changes(f,g,v,'D3'),1)
    def test_browser_state_decoding_and_world_alignment(self):
        module=Path(__file__).parents[1]/'viewer/node_state.js';defs=module.with_name('native_nodes.json')
        code=r'''
import fs from 'node:fs';import assert from 'node:assert/strict';
const m=await import('file://'+process.argv[1]);const defs=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const pairs=[['mcl_core:water_source',15],['mcl_core:water_flowing',15],['mcl_core:lava_source',0],['mcl_core:lava_flowing',11],['mcl_fire:fire',0],['mcl_fire:fire',8],['mcl_core:leaves',0],['air',0],['mcl_core:dirt',0],['mcl_mushrooms:mushroom_red',0]];
const table=m.tokenTable({tokens:pairs.map(([node_name,param2],id)=>({id,node_name,param2}))},defs);
assert.equal(table[0].liquidType,'source');assert.equal(table[1].liquidType,'flowing');
assert.deepEqual(m.liquidState(table[0]),{type:'source',level:null,flowDown:null,height:1,paramtype2:'color'});
assert.equal(m.liquidState(table[1]).level,7);assert.equal(m.liquidState(table[1]).flowDown,true);
assert.equal(m.liquidState(table[3]).level,3);assert.equal(m.liquidState(table[3]).flowDown,true);
assert.equal(m.liquidState(table[4]),null);assert.equal(m.transitionType(table[4],table[5]),'fire_age_update');
assert.equal(m.transitionType(table[6],table[4]),'leaves_to_fire');assert.equal(m.transitionType(table[8],table[3]),'solid_to_liquid_requires_write_evidence');
const meta={region_origin:[-288,1424,-32],center_origin:[-272,1440,-16]},world=[-258,1440,-2];
assert.deepEqual(m.coordinates(world,meta).regionLocal,[30,16,30]);assert.deepEqual(m.coordinates(world,meta).centerChunkLocal,[14,0,14]);
assert.equal(m.coordinates([-256,1440,-2],meta).centerChunkLocal,null);assert.equal(m.worldIndex([-289,1440,-2],meta.region_origin),null);
const before=new Uint16Array(48**3).fill(7),after=before.slice(),id=m.worldIndex(world,meta.region_origin);before[id]=3;after[id]=7;
assert.equal(table[before[m.worldIndex(world,meta.region_origin)]].name,'mcl_core:lava_flowing');assert.equal(table[after[m.worldIndex(world,meta.region_origin)]].name,'air');
const a=new Uint16Array(27).fill(7),b=a.slice();a[13]=4;b[13]=5;a[4]=6;b[4]=6;
assert.deepEqual(m.fireAttachments(a,13,table,[3,3,3]),m.fireAttachments(b,13,table,[3,3,3]));
assert.equal(m.relevantChange('D3',13,a,b,table,[3,3,3]),false);b[4]=4;assert.equal(m.relevantChange('D3',4,a,b,table,[3,3,3]),true);
// Range-aware liquid slope, with a complete surrounding liquid layer.
const t=m.tokenTable({tokens:[{id:0,node_name:'air',param2:0},{id:1,node_name:'mcl_core:water_flowing',param2:7},{id:2,node_name:'mcl_core:water_flowing',param2:1},{id:3,node_name:'mcl_core:water_source',param2:0}]},defs);
const liquid=new Uint16Array(27);for(let x=0;x<3;x++)for(let y=0;y<3;y++)liquid[(x*3+y)*3+1]=1;
const flat=m.liquidCorners(liquid,13,t,[3,3,3]);assert(flat.every(h=>Math.abs(h-6.5/7)<1e-8));
liquid[1]=2;const slope=m.liquidCorners(liquid,13,t,[3,3,3]);assert(slope[0]<slope[2]);liquid[1]=3;assert.equal(m.liquidCorners(liquid,13,t,[3,3,3])[0],1);
'''
        subprocess.run(['node','--input-type=module','-e',code,str(module.resolve()),str(defs.resolve())],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
