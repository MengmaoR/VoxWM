-- Vox2Pix uses an isolated native world. Only the observer's pose is scripted.
local mt = minetest
local mod = mt.get_modpath("craftium_env")
local world = mt.get_worldpath()
local search = dofile(mod .. "/biome_search.lua")
local geometry = dofile(mod .. "/scene_geometry.lua")
local pending, started, last = nil, false, 0
local box_names, box_types = {}, {}
assert(voxwm_clock and voxwm_region and mt.get_node_boxes, "Build the VoxWM fixed24 engine")
assert(voxwm_clock().region_capture_version == "voxwm-region-integer-v1", "Build integer-origin region capture")

local function read(name)
    local f = io.open(world .. "/" .. name, "rb")
    if not f then return nil end
    local value = f:read("*a")
    f:close()
    return value
end

local function write(name, value)
    local path = world .. "/" .. name
    local f = assert(io.open(path .. ".tmp", "wb"))
    assert(f:write(type(value) == "string" and value or assert(mt.write_json(value))))
    assert(f:close())
    assert(os.rename(path .. ".tmp", path))
end

local function reply(request, data)
    data.request_id = request.request_id
    data.native_clock = voxwm_clock()
    data.lua_tick = data.native_clock.server_tick
    data.sim_time = data.native_clock.absolute_sim_time
    write("vox2pix_response.json", data)
end

local function xyz(v)
    return {x=v[1], y=v[2], z=v[3]}
end

local function clean_hud(player)
    player:hud_set_flags({hotbar=false, healthbar=false, crosshair=false, wielditem=false,
        breathbar=false, minimap=false, minimap_radar=false, basic_debug=false, chat=false})
    for id in pairs(player:hud_get_all()) do player:hud_remove(id) end
end

local function publish_nodes()
    local rows, definitions, seen = {}, {}, {}
    local selection_types = {
        mesh=true, signlike=true, firelike=true, torchlike=true, fencelike=true,
        raillike=true, glasslike_framed=true, glasslike_framed_optional=true,
    }
    for name in pairs(mt.registered_nodes) do
        local id = mt.get_content_id(name)
        if not seen[id] then
            seen[id] = true
            local canonical = mt.get_name_from_content_id(id)
            local d = mt.registered_nodes[canonical]
            local drawtype = d.drawtype or "normal"
            local groups = d.groups or {}
            local water = (groups.water or 0) > 0
            local liquidtype = d.liquidtype or "none"
            local support = d.walkable ~= false and liquidtype == "none"
            rows[#rows + 1] = {id, canonical, support, water, liquidtype}
            definitions[#definitions + 1] = {
                id=id, name=canonical, drawtype=drawtype, walkable=support,
                water=water, glass=drawtype:sub(1, 9) == "glasslike" or (groups.glass or 0) > 0,
                terrain=geometry.terrain(d), liquidtype=liquidtype,
                visual_scale=d.visual_scale or 1, paramtype2=d.paramtype2 or "none",
                liquid_range=d.liquid_range or 8, liquid_alternative_source=d.liquid_alternative_source,
                liquid_alternative_flowing=d.liquid_alternative_flowing,
            }
            local box_type = drawtype == "nodebox" and "node_box" or
                (selection_types[drawtype] and "selection_box" or nil)
            if box_type then
                box_types[canonical] = box_type
                box_names[#box_names + 1] = canonical
            end
        end
    end
    write("vox2pix_engine.json", {version=mt.get_version(), definitions=definitions,
        mapgen=mt.get_mapgen_setting("mg_name"), simulation_hz=24})
    write("gt_nodes.json", rows)
end

local function survey(r)
    local low, high = xyz(r.low_xyz), xyz(r.high_xyz)
    pending = r
    mt.emerge_area(vector.subtract(low, 1), vector.add(high, 1), function(_, action, remaining)
        if action == mt.EMERGE_ERRORED or action == mt.EMERGE_CANCELLED then r.failed = true end
        if remaining ~= 0 then return end
        mt.after(0, function()
            if r.failed then
                reply(r, {status="emergence_failed"})
            else
                local vm = VoxelManip()
                local vlow, vhigh = vm:read_from_map(low, high)
                local area = VoxelArea:new({MinEdge=vlow, MaxEdge=vhigh})
                local data = vm:get_data()
                local file = assert(io.open(world .. "/vox2pix_survey.bin", "wb"))
                -- One ENU column per write keeps the temporary byte buffer small.
                local column = {}
                for x=low.x,high.x do for z=low.z,high.z do
                    local at = 1
                    for y=low.y,high.y do
                        local id = data[area:index(x,y,z)]
                        column[at], column[at+1] = id % 256, math.floor(id / 256)
                        at = at + 2
                    end
                    assert(file:write(string.char(unpack(column))))
                end end
                assert(file:close())
                reply(r, {status="surveyed"})
            end
            pending = nil
        end)
    end)
end

local function anchor_valid(foot, kind)
    -- The standing collision box occupies these cells under the native
    -- integer-centered convention. Eye clearance is checked from its packet.
    for dy=0,2 do
        if mt.get_node(vector.offset(foot, 0, dy, 0)).name ~= "air" then return false end
    end
    local below = vector.offset(foot, 0, -1, 0)
    if kind == "surface_water" then
        return mt.get_item_group(mt.get_node(below).name, "water") > 0
    end
    local def = mt.registered_nodes[mt.get_node(below).name]
    if not def or def.walkable == false or (def.liquidtype and def.liquidtype ~= "none") then return false end
    for _, box in ipairs(mt.get_node_boxes("collision_box", below)) do
        if box[1] < 0 and box[3] < 0 and box[4] > 0 and box[6] > 0 then return true end
    end
    return false
end

local function check_anchor(r)
    local foot = xyz(r.foot_xyz)
    local data = mt.get_biome_data({x=foot.x, y=r.surface_y, z=foot.z})
    reply(r, {status="anchor", biome=data and mt.get_biome_name(data.biome) or "unresolved",
        valid=anchor_valid(foot, r.anchor_kind)})
end

local function find_site(r)
    if not r.origins then return search.find(mt, r) end
    local best, best_score, last_result = nil, -math.huge, nil
    local probes = {{8,8},{-8,8},{24,8},{8,-8},{8,24}}
    for _, origin in ipairs(r.origins) do
        r.x, r.z = origin[1], origin[2]
        local hit = search.find(mt, r)
        last_result = hit
        if hit.status == "found" then
            local score = 0
            for _, offset in ipairs(probes) do
                local x, z = hit.site[1] + offset[1], hit.site[2] + offset[2]
                local spawn = mt.get_spawn_level(x, z)
                if spawn then
                    local data = mt.get_biome_data({x=x, y=spawn-1, z=z})
                    local name = data and mt.get_biome_name(data.biome) or ""
                    for _, alias in ipairs(r.biome_groups[r.biome_group]) do
                        if name == alias or name:sub(1, #alias+1) == alias .. "_" then
                            score = score + math.max(1, spawn + 8)
                            break
                        end
                    end
                end
            end
            if score > best_score then best, best_score = hit, score end
        end
    end
    return best or last_result
end

local function pose(r, player)
    if r.relocate then
        local foot = xyz(r.foot_xyz)
        player:set_properties({is_visible=false})
        player:set_physics_override({gravity=0, speed=0, jump=0})
        player:set_pos(foot)
    end
    clean_hud(player)
    player:set_look_horizontal(math.rad(r.yaw))
    player:set_look_vertical(math.rad(r.pitch))
    reply(r, {status="posed"})
end

local function snapshot(r)
    local low = xyz(r.low_xyz)
    local foot = vector.add(low, 24)
    local ok, payload = pcall(voxwm_region, low)
    if not ok then reply(r, {status="snapshot_failed", reason=tostring(payload)}); return end
    if not anchor_valid(foot, r.anchor_kind) then
        reply(r, {status="snapshot_failed", reason="Native anchor changed"}); return
    end
    write("vox2pix_region.bin", payload)
    if r.boxes then
        local boxes = {}
        local high = vector.add(low, 47)
        if #box_names > 0 then
            for _, pos in ipairs(mt.find_nodes_in_area(low, high, box_names)) do
                local node = mt.get_node(pos)
                local offset = vector.subtract(pos, low)
                local index = (offset.x * 48 + offset.z) * 48 + offset.y
                boxes[#boxes + 1] = {index, mt.get_node_boxes(box_types[node.name], pos)}
            end
        end
        write("vox2pix_boxes.json", {boxes=boxes})
    end
    reply(r, {status="captured", world_time=mt.get_timeofday(),
        weather=mcl_weather.get_weather(), gamma=tonumber(mt.settings:get("display_gamma")),
        rgb_enabled=not mt.settings:get_bool("voxwm_skip_rgb"),
        engine_us=mt.get_us_time()})
end

mt.register_on_joinplayer(function(player)
    clean_hud(player)
    mt.set_timeofday(0.5)
    if mcl_weather.get_weather() ~= "none" then mcl_weather.change_weather("none", nil, "Vox2Pix") end
    write("gt_ready", "ready\n")
end)

mt.registered_voxwm_poststeps = {function()
    local player = mt.get_connected_players()[1]
    if not player then return end
    if not started and read("gt_start") and read("gt_arm") then
        publish_nodes()
        started = true
    end
    if not started or pending then return end
    local contents = read("vox2pix_command.json")
    if not contents then return end
    local r = assert(mt.parse_json(contents))
    if r.request_id <= last then return end
    last = r.request_id
    -- Native scene searches advance physics without drawing RGB. Rendering is
    -- restored before the usual camera settle ticks and every saved view.
    if r.kind ~= "snapshot" then mt.settings:set_bool("voxwm_skip_rgb", r.kind ~= "pose") end
    if r.kind == "search" then
        reply(r, find_site(r))
    elseif r.kind == "survey" then
        survey(r)
    elseif r.kind == "anchor" then
        check_anchor(r)
    elseif r.kind == "pose" then
        pose(r, player)
    elseif r.kind == "snapshot" then
        snapshot(r)
    else
        error("Unknown Vox2Pix native request: " .. tostring(r.kind))
    end
end}
