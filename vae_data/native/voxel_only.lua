-- Installed after builtin/game/init.lua and BEFORE game mods. No upstream file is modified.
-- Falling-node transitions are explicitly unsupported: the world is quarantined if requested.
local c=core
voxel_only={blocked_entities={},blocked_items=0,falling_attempts=0,entity_mutations=0,entity_callbacks=0}
local v=voxel_only;local depth=0
c.add_entity=function(pos,name,...)
    v.blocked_entities[name]=(v.blocked_entities[name]or 0)+1
    if name=='__builtin:falling_node'then v.falling_attempts=v.falling_attempts+1 end
    return nil
end
c.add_item=function(...)v.blocked_items=v.blocked_items+1;return nil end
if c.spawn_item then c.spawn_item=c.add_item end
local original_register=c.register_entity
local function protect(def)
    local result=setmetatable({},getmetatable(def));for k,value in pairs(def)do result[k]=value end
    for _,key in ipairs({'on_activate','on_step','on_punch','on_rightclick','on_deactivate','on_death'})do
        local fn=def[key]
        if fn then result[key]=function(...)
            depth=depth+1;v.entity_callbacks=v.entity_callbacks+1
            local args={...};local ok,value=pcall(function()return fn(unpack(args))end)
            depth=depth-1;if not ok then error(value)end;return value
        end end
    end
    return result
end
c.register_entity=function(name,def)return original_register(name,protect(def))end
for _,def in pairs(c.registered_entities or{})do
    local safe=protect(def);for k,value in pairs(safe)do def[k]=value end
end
for _,key in ipairs({'set_node','add_node','remove_node','swap_node','bulk_set_node'})do
    local fn=c[key]
    if fn then c[key]=function(...)
        if depth>0 then v.entity_mutations=v.entity_mutations+1;error('Voxel-only: entity attempted a node mutation')end
        return fn(...)
    end end
end
function v.audit()
    local entities,dropped=0,0
    for _,entity in pairs(c.luaentities or{})do
        if entity.object and entity.object:get_pos()then
            entities=entities+1;if entity.name=='__builtin:item'then dropped=dropped+1 end
        end
    end
    return {nontechnical_entities=entities,dropped_items=dropped,entity_induced_changes=v.entity_mutations,
        entity_callbacks=v.entity_callbacks,blocked_entities=v.blocked_entities,blocked_item_attempts=v.blocked_items,
        unsupported_falling_attempts=v.falling_attempts,
        compatible=v.falling_attempts==0 and entities==0 and v.entity_mutations==0}
end
-- VoxeLibre spawner setup dereferences its visual doll without checking add_entity failure.
-- Disable ONLY its non-voxel setup/render/timer path; preserve the voxel cage and metadata.
if c.register_on_mods_loaded then c.register_on_mods_loaded(function()
    if mcl_mobspawners then
        mcl_mobspawners.setup_spawner=function(pos,mob,minlight,maxlight,maxmobs,distance,offset)
            local meta=c.get_meta(pos);meta:set_string('Mob',mob or'mobs_mc:pig')
            meta:set_int('MinLight',minlight or 0);meta:set_int('MaxLight',maxlight or 15)
            meta:set_int('MaxMobsInArea',0);meta:set_int('PlayerDistance',distance or 15);meta:set_int('YOffset',offset or 0)
            c.get_node_timer(pos):stop()
        end
        c.override_item('mcl_mobspawners:spawner',{on_timer=function()return false end,on_punch=function()end})
        for _,lbm in ipairs(c.registered_lbms or{})do
            if lbm.name and lbm.name:find('mcl_mobspawners:',1,true)then lbm.action=function()end end
        end
    end
end)end
