-- Custom scenery only. Water flow, collisions, ticks and rendering remain native VoxeLibre.
local mt = minetest
local seed = tonumber(mt.settings:get("fixed_map_seed")) or 42
math.randomseed(seed)
local world = mt.get_worldpath()
local built, armed, started = false, false, false
local radius = {x=24, y=24, z=24}

local function build_scene()
    local vm = VoxelManip()
    local low, high = vm:read_from_map({x=-32,y=-32,z=-32}, {x=32,y=32,z=32})
    local area = VoxelArea:new({MinEdge=low, MaxEdge=high})
    local data = vm:get_data()
    local param2 = vm:get_param2_data()
    local air = mt.get_content_id("air")
    local stone = mt.get_content_id("mcl_core:stone")
    local dirt = mt.get_content_id("mcl_core:dirt_with_grass")
    for z=-32,32 do for y=-32,32 do for x=-32,32 do
        local i=area:index(x,y,z)
        data[i] = y < 0 and stone or (y == 0 and dirt or air)
        param2[i] = 0
    end end end
    -- Dry descending channel: native liquid can reach a lower pool without a hand-coded fluid solver.
    for z=-7,5 do
        local bed = -1 - math.floor((z+7)/4)
        for x=6,8 do for y=bed+1,0 do data[area:index(x,y,z)]=air end end
    end
    for z=6,12 do for x=4,12 do for y=-3,0 do data[area:index(x,y,z)]=air end end end
    -- One air layer below the complete 3x3 horizontal source footprint.
    for z=-7,-5 do for x=6,8 do data[area:index(x,-1,z)]=air end end
    vm:set_data(data); vm:set_param2_data(param2); vm:write_to_map(true)
    local trees = {{0,6},{-8,5},{-7,-8},{11,-10}}
    for _, p in ipairs(trees) do
        for y=1,5 do mt.set_node({x=p[1],y=y,z=p[2]}, {name="mcl_core:tree"}) end
        for x=-2,2 do for z=-2,2 do for y=4,6 do
            if not (x==0 and z==0 and y<=5) and math.abs(x)+math.abs(z)<4 then
                mt.set_node({x=p[1]+x,y=y,z=p[2]+z}, {name="mcl_core:leaves"})
            end
        end end end
    end
end

-- Hide chat/debug from join. Craftium enables mcl_logging_mobs_spawn and
-- those Totals lines land in the chat HUD; keep recorded RGB free of diagnostic overlays.
local function hide_overlay(player)
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
end

mt.register_on_joinplayer(function(player)
    hide_overlay(player)
    player:set_pos({x=0,y=0.5,z=0}) -- spawn only, never a trajectory
    player:set_look_horizontal(0); player:set_look_vertical(0)
    mt.emerge_area({x=-32,y=-32,z=-32}, {x=32,y=32,z=32}, function(_,_,remaining)
        if remaining == 0 and not built then
            build_scene(); built = true
            local f = assert(io.open(world .. "/gt_ready", "w")); f:write("ready"); f:close()
        end
    end)
end)

mt.register_globalstep(function(dtime)
    local player = mt.get_connected_players()[1]
    if not player then return end
    if built and not armed then
        local arm = io.open(world .. "/gt_arm", "r")
        if arm then
            arm:close()
            player:set_pos({x=0,y=0.5,z=0})
            local v = player:get_velocity()
            player:add_velocity({x=-v.x,y=-v.y,z=-v.z})
            player:set_look_horizontal(0); player:set_look_vertical(0)
            armed = true
        end
    end
    if armed and not started then
        local f = io.open(world .. "/gt_start", "r")
        if f then
            f:close()
            mt.set_timeofday(0.5)
            build_scene()
            for z=-7,-5 do for x=6,8 do
                mt.set_node({x=x,y=0,z=z}, {name="mcl_core:water_source"})
            end end
            -- JSON array of [id,name,walkable,water,liquidtype]. Object maps with
            -- numeric-looking keys are not stable under Luanti write_json.
            local rows, seen = {}, {}
            for name, def in pairs(mt.registered_nodes) do
                local id = mt.get_content_id(name)
                if not seen[id] then
                    seen[id] = true
                    local canonical = mt.get_name_from_content_id(id)
                    local d = mt.registered_nodes[canonical] or def
                    rows[#rows+1] = {id, canonical, d.walkable ~= false,
                        (d.groups and (d.groups.water or 0) > 0) or false,
                        d.liquidtype or "none"}
                end
            end
            table.sort(rows, function(a, b) return a[1] < b[1] end)
            local json, err = mt.write_json(rows)
            if not json then error("gt_nodes json: " .. tostring(err)) end
            local out = assert(io.open(world .. "/gt_nodes.json", "w"))
            out:write(json); out:close()
            started = true
        end
    end
    hide_overlay(player)
    -- Official Craftium path publishes nodes, param2 and their matching voxel center.
    if mt.settings:get_bool("voxel_obs") then
        get_voxel_data_cpp(player:get_pos(), radius)
    end
end)
