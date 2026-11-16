"""Observed node-state classifications, kept separate from claims about write causality."""
import json
from pathlib import Path

DEFINITIONS=json.loads((Path(__file__).parent/'viewer/native_nodes.json').read_text())

def definition(name):return DEFINITIONS['nodes'].get(name,{})
def liquid(name):return definition(name).get('liquidtype','none')
def fire(name):return definition(name).get('drawtype')=='firelike'
def leaves(name):return definition(name).get('groups',{}).get('leaves',0)>0

def classify(before,after,cooling=False):
    a,pa=before;b,pb=after
    if before==after:return 'unchanged'
    if fire(a)and a==b:return 'fire_age_update'if definition(a).get('param2_semantics')=='age'else'fire_param2_update'
    if a=='air'and fire(b):return 'air_to_fire'
    if leaves(a)and fire(b):return 'leaves_to_fire'
    if leaves(a)and b=='air':return 'leaves_to_air'
    if fire(a)and b=='air':return 'fire_to_air'
    if cooling:return 'confirmed_cooling_product'
    if liquid(a)=='source'and b=='air':return 'source_loss_requires_write_evidence'
    if liquid(a)=='flowing'and b=='air':return 'flowing_liquid_recession_observed'
    if liquid(a)=='flowing'and a==b:return 'liquid_level_or_flow_down_update'
    if liquid(a)=='source'and a==b:return 'source_param2_update'
    if liquid(b)!='none'and a!='air'and liquid(a)=='none':
        if definition(a).get('groups',{}).get('destroy_by_lava_flow',0)>0 and definition(b).get('groups',{}).get('lava',0)>0:
            return 'lava_replaced_lava_destructible_node_observed'
        return 'solid_to_liquid_requires_write_evidence'
    if a=='air'and liquid(b)=='flowing':return 'flowing_liquid_appearance_observed'
    if fire(b):return 'node_to_fire'
    if b=='air':return 'node_removed_cause_unrecorded'
    return 'node_state_change'
