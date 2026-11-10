-- Seeded native map generation, or a user-supplied voxel palette. No fluid solver.
local mt=minetest
local world=mt.get_worldpath()
local seed=tonumber(mt.settings:get("fixed_map_seed")) or 42
math.randomseed(seed)
mt.set_mapgen_setting("seed",tostring(seed),true)
local file=io.open(world.."/custom_world.json","r")
local custom=nil
if file then custom=assert(mt.parse_json(file:read("*a")));file:close() end
local ready,armed,started=false,false,false
local radius={x=24,y=24,z=24}
local function marker(name)
    local f=assert(io.open(world.."/"..name,"w"));f:write("ready");f:close()
end
local function present(name)
    local f=io.open(world.."/"..name,"r");if f then f:close();return true end
    return false
end
local function place_custom(player)
    player:set_pos({x=custom.spawn[1],y=custom.spawn[2],z=custom.spawn[3]})
    player:set_look_horizontal(math.rad(custom.look[1]));player:set_look_vertical(math.rad(custom.look[2]))
end
local function build_custom()
    local o,s=custom.origin,custom.shape
    local vm=VoxelManip()
    local low,high=vm:read_from_map({x=o[1],y=o[2],z=o[3]}, {x=o[1]+s[1]-1,y=o[2]+s[2]-1,z=o[3]+s[3]-1})
    local area=VoxelArea:new({MinEdge=low,MaxEdge=high})
    local data,param=vm:get_data(),vm:get_param2_data()
    local palette={}
    for i,name in ipairs(custom.node_names) do
        assert(mt.registered_nodes[name],"Unknown native node: "..name)
        palette[i]=mt.get_content_id(name)
    end
    local i=1
    for x=0,s[1]-1 do for y=0,s[2]-1 do for z=0,s[3]-1 do
        local at=area:index(o[1]+x,o[2]+y,o[3]+z)
        local class=custom.voxels[i]+1
        data[at]=palette[class];param[at]=custom.node_param2[class];i=i+1
    end end end
    vm:set_data(data);vm:set_param2_data(param);vm:write_to_map(true)
    -- Queue native liquid updates only for nodes supplied as water/liquid by the user.
    i=1
    for x=0,s[1]-1 do for y=0,s[2]-1 do for z=0,s[3]-1 do
        local def=mt.registered_nodes[custom.node_names[custom.voxels[i]+1]]
        if def.liquidtype and def.liquidtype~="none" then mt.transforming_liquid_add({x=o[1]+x,y=o[2]+y,z=o[3]+z}) end
        i=i+1
    end end end
end
mt.register_on_joinplayer(function(player)
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    local p=player:get_pos()
    local low,high={x=p.x-32,y=p.y-32,z=p.z-32},{x=p.x+32,y=p.y+32,z=p.z+32}
    if custom then
        local o,s=custom.origin,custom.shape
        low={x=o[1]-24,y=o[2]-24,z=o[3]-24}
        high={x=o[1]+s[1]+24,y=o[2]+s[2]+24,z=o[3]+s[3]+24}
    end
    mt.emerge_area(low,high,function(_,_,remaining)
        if remaining==0 and not ready then
            if custom then build_custom();place_custom(player) end
            ready=true;marker("gt_ready")
        end
    end)
    local inv=player:get_inventory()
    for item in (mt.settings:get("starting_inventory_creative") or ""):gmatch("[^,]+") do inv:add_item("main",item) end
end)
mt.register_on_dieplayer(function() set_termination() end)
mt.register_globalstep(function()
    local player=mt.get_connected_players()[1];if not player then return end
    if ready and not armed and present("gt_arm") then
        if custom then place_custom(player)
        else player:set_look_horizontal(0);player:set_look_vertical(0) end
        armed=true
    end
    if armed and not started and present("gt_start") then
        mt.set_timeofday(0.5)
        local rows,seen={},{}
        for name,def in pairs(mt.registered_nodes) do
            local id=mt.get_content_id(name)
            if not seen[id] then
                seen[id]=true
                local canonical=mt.get_name_from_content_id(id)
                local d=mt.registered_nodes[canonical] or def
                rows[#rows+1]={id,canonical,d.walkable~=false,(d.groups and (d.groups.water or 0)>0) or false,d.liquidtype or "none"}
            end
        end
        table.sort(rows,function(a,b)return a[1]<b[1]end)
        local f=assert(io.open(world.."/gt_nodes.json","w"));f:write(assert(mt.write_json(rows)));f:close()
        started=true
    end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if mt.settings:get_bool("voxel_obs") then get_voxel_data_cpp(player:get_pos(),radius) end
end)
