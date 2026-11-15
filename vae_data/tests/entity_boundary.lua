-- Real Lua guard code under a minimal C-API stub; no physics result is inferred.
core={registered_entities={},luaentities={}}
function core.register_entity(name,def)core.registered_entities[name]=def end
function core.add_entity()error('original entity API was reached')end
function core.add_item()error('original dropped item API was reached')end
function core.set_node()return true end
dofile('vae_data/native/voxel_only.lua')
assert(core.add_entity({},'mobs_mc:pig')==nil);assert(core.add_item({},'dirt')==nil)
local a=voxel_only.audit();assert(a.nontechnical_entities==0 and a.dropped_items==0 and a.compatible)
core.register_entity('bad',{on_step=function()core.set_node({}, {})end})
assert(not pcall(core.registered_entities.bad.on_step));assert(voxel_only.audit().entity_induced_changes==1)
core.add_entity({},'__builtin:falling_node');assert(not voxel_only.audit().compatible)
print('PASS: entity/item factories blocked; entity mutations fail; falling transitions explicitly excluded')
