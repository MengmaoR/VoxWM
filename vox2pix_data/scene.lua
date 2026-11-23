-- Vox2Pix uses an isolated native world. Only the observer's pose is scripted.
local mt = minetest
local mod = mt.get_modpath("craftium_env")
local world = mt.get_worldpath()
local search = dofile(mod .. "/biome_search.lua")
local geometry = dofile(mod .. "/scene_geometry.lua")
local pending, started, last = nil, false, 0
local box_names, box_types = {}, {}
local cave_events, v3_cave_events = {}, {}
local cave_hints_enabled = mt.settings:get_bool("vox2pix_cave_hints", false)
local v3_scene = mt.settings:get_bool("vox2pix_v3", false)
local replay_restore = v3_scene and mt.settings:get_bool("vox2pix_replay_restore", false)
assert(voxwm_clock and voxwm_region and mt.get_node_boxes, "Build the VoxWM fixed24 engine")
assert(voxwm_clock().region_capture_version == "voxwm-region-integer-v1", "Build integer-origin region capture")

if cave_hints_enabled then
    local event_names = {"large_cave_begin", "large_cave_end"}
    if v3_scene then
        -- Both ordinary and large cave endpoints are registered in mapgen.cpp.
        event_names[#event_names + 1] = "cave_begin"
        event_names[#event_names + 1] = "cave_end"
    end
    local notification_flags = {}
    for _, kind in ipairs(event_names) do notification_flags[kind] = true end
    mt.set_gen_notify(notification_flags)
    mt.register_on_generated(function()
        local events = mt.get_mapgen_object("gennotify") or {}
        for _, kind in ipairs(event_names) do
            for _, p in ipairs(events[kind] or {}) do
                if v3_scene then
                    if pending then
                        local key = math.floor(p.x / 8) .. ":" .. math.floor(p.y / 8) .. ":" .. math.floor(p.z / 8)
                        v3_cave_events[key] = p
                    end
                elseif p.y >= -62 and p.y <= -15 then
                    if #cave_events == 192 then table.remove(cave_events, 1) end
                    cave_events[#cave_events + 1] = p
                end
            end
        end
    end)
end

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

local function site_score(site, aliases)
    local score = 0
    for _, offset in ipairs({{8,8},{-8,8},{24,8},{8,-8},{8,24}}) do
        local x, z = site[1] + offset[1], site[2] + offset[2]
        local spawn = mt.get_spawn_level(x, z)
        if spawn then
            local data = mt.get_biome_data({x=x, y=spawn-1, z=z})
            local name = data and mt.get_biome_name(data.biome) or ""
            for _, alias in ipairs(aliases) do
                if name == alias or name:sub(1, #alias+1) == alias .. "_" then
                    score = score + math.max(1, spawn + 8)
                    break
                end
            end
        end
    end
    return score
end

local function nearby_cave_hints(r, low, high)
    local aliases = r.biome_groups and r.biome_groups[r.biome_group]
    if not cave_hints_enabled or not aliases then return nil end
    local center_x, center_z = (low.x + high.x) / 2, (low.z + high.z) / 2
    local current_x, current_z = math.floor(center_x / 16) * 16, math.floor(center_z / 16) * 16
    local playable = (tonumber(mt.get_mapgen_setting("mapgen_limit")) or 31000)
        - ((tonumber(mt.get_mapgen_setting("chunksize")) or 5) + 1) * 16 - 64
    local seen, ranked, hints = {}, {}, {}
    for _, p in ipairs(cave_events) do
        local x, z = math.floor(p.x / 16) * 16, math.floor(p.z / 16) * 16
        local span = math.max(math.abs(p.x - center_x), math.abs(p.z - center_z))
        local key = x .. ":" .. z
        if span <= 128 and math.abs(x) <= playable and math.abs(z) <= playable
            and (x ~= current_x or z ~= current_z) then
            local outside = p.x < low.x + 24 or p.x > high.x - 23
                or p.z < low.z + 24 or p.z > high.z - 23
            local score = (outside and 20 or 0) - math.abs(p.y + 38) - span / 32
            local row = seen[key]
            if row then
                row.score = math.max(row.score, score)
            else
                row = {site={x,z}, score=score}
                seen[key] = row
                ranked[#ranked + 1] = row
            end
        end
    end
    for _, row in ipairs(ranked) do
        row.score = row.score + 2 * site_score(row.site, aliases)
    end
    table.sort(ranked, function(a, b)
        if a.score ~= b.score then return a.score > b.score end
        if a.site[1] ~= b.site[1] then return a.site[1] < b.site[1] end
        return a.site[2] < b.site[2]
    end)
    for i=1,math.min(3, #ranked) do hints[i] = ranked[i].site end
    if #hints > 0 then return hints end
end

local function survey(r)
    local low, high = xyz(r.low_xyz), xyz(r.high_xyz)
    local limit = tonumber(mt.get_mapgen_setting("mapgen_limit")) or 31000
    local margin = (tonumber(mt.get_mapgen_setting("chunksize")) or 5) * 16 + 16
    if v3_scene then
        for _, axis in ipairs({"x", "y", "z"}) do
            if low[axis] <= -limit + margin or high[axis] >= limit - margin then
                reply(r, {status="survey_outside_mapgen_limit"})
                return
            end
        end
        -- Events from this emergence alone; no cross-site/world accumulation.
        v3_cave_events = {}
    end
    pending = r
    mt.emerge_area(vector.subtract(low, 1), vector.add(high, 1), function(_, action, remaining)
        if action == mt.EMERGE_ERRORED or action == mt.EMERGE_CANCELLED then r.failed = true end
        if remaining ~= 0 then return end
        mt.after(0, function()
            if r.failed then
                reply(r, {status="emergence_failed"})
            elseif r.kind == "emerge_capture" then
                reply(r, {status="capture_emerged"})
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
                local hints
                if v3_scene then
                    hints = {}
                    for _, pos in pairs(v3_cave_events) do
                        if pos.x >= low.x and pos.x <= high.x and pos.y >= low.y and pos.y <= high.y
                            and pos.z >= low.z and pos.z <= high.z then
                            hints[#hints+1] = {pos.x, pos.y, pos.z}
                        end
                    end
                    table.sort(hints, function(a, b)
                        if a[1] ~= b[1] then return a[1] < b[1] end
                        if a[2] ~= b[2] then return a[2] < b[2] end
                        return a[3] < b[3]
                    end)
                else
                    hints = nearby_cave_hints(r, low, high)
                end
                reply(r, {status="surveyed", cave_hints=hints})
            end
            pending = nil
        end)
    end)
end

local function anchor_valid(foot, kind)
    -- The standing collision box occupies these cells under the native
    -- integer-centered convention. Eye clearance is checked from its packet.
    for dy=0,2 do
        local node = mt.get_node(vector.offset(foot, 0, dy, 0))
        if kind == "underwater_floor" then
            if mt.get_item_group(node.name, "water") <= 0 then return false end
        elseif node.name ~= "air" then
            return false
        end
    end
    local below = vector.offset(foot, 0, -1, 0)
    local name = mt.get_node(below).name
    local def = mt.registered_nodes[name]
    if kind == "surface_water" then
        return mt.get_item_group(name, "water") > 0 and
            (not v3_scene or (def and (def.liquidtype == "source" or def.liquidtype == "flowing")))
    end
    if not def or def.walkable == false or (def.liquidtype and def.liquidtype ~= "none") then return false end
    if v3_scene and not geometry.terrain(def) then return false end
    for _, box in ipairs(mt.get_node_boxes("collision_box", below)) do
        if box[1] < 0 and box[3] < 0 and box[4] > 0 and box[6] > 0 then return true end
    end
    return false
end

local function surface_biome(pos)
    local data = mt.get_biome_data(pos)
    return data and mt.get_biome_name(data.biome) or "unresolved"
end

local function check_anchor(r)
    local foot = xyz(r.foot_xyz)
    local biome = surface_biome({x=foot.x, y=r.surface_y, z=foot.z})
    local result = {status="anchor", biome=biome, valid=anchor_valid(foot, r.anchor_kind)}
    if v3_scene then
        result.surface_biome = biome
        result.foot_biome = surface_biome(foot)
    end
    reply(r, result)
end

local function find_water_site(r)
    -- Prefer low or submerged spawn hints near the waterline. The v7 spawn
    -- query can also return nil above high land, so the survey verifies water.
    local water = tonumber(mt.get_mapgen_setting("water_level")) or 1
    local best, best_score = {r.x, r.z}, 0
    for _, offset in ipairs({{0,0}, {64,0}, {-64,0}, {0,64}, {0,-64}, {128,128},
            {-128,128}, {128,-128}, {-128,-128}, {256,0}, {-256,0}, {0,256}, {0,-256}}) do
        local x, z = r.x + offset[1], r.z + offset[2]
        if not mt.get_spawn_level(x, z) then
            local score = 1
            for _, side in ipairs({{48,0}, {-48,0}, {0,48}, {0,-48}}) do
                local shore = mt.get_spawn_level(x + side[1], z + side[2])
                if shore and shore <= water + 8 then score = score + 2 end
            end
            if score > best_score then
                best, best_score = {x, z}, score
            end
        end
    end
    return {status="found", site=best}
end

local function find_site(r)
    if not r.origins then return search.find(mt, r) end
    local best, best_score, last_result = nil, -math.huge, nil
    for _, origin in ipairs(r.origins) do
        r.x, r.z = origin[1], origin[2]
        local hit = search.find(mt, r)
        last_result = hit
        if hit.status == "found" then
            local score = site_score(hit.site, r.biome_groups[r.biome_group])
            if score > best_score then best, best_score = hit, score end
        end
    end
    return best or last_result
end

local function resting_support(player)
    local pos = player:get_pos()
    local box = player:get_properties().collisionbox
    local bottom = pos.y + box[2]
    local min_x, max_x = pos.x + box[1], pos.x + box[4]
    local min_z, max_z = pos.z + box[3], pos.z + box[6]
    local support, best_gap
    for x=math.floor(min_x+.5),math.floor(max_x+.5) do
        for z=math.floor(min_z+.5),math.floor(max_z+.5) do
            for y=math.floor(bottom)-1,math.floor(bottom+.5) do
                local at = {x=x, y=y, z=z}
                local def = mt.registered_nodes[mt.get_node(at).name]
                if geometry.terrain(def) then
                    for _, shape in ipairs(mt.get_node_boxes("collision_box", at)) do
                        local top = y + shape[5]
                        local gap = bottom - top
                        if gap >= -.005 and gap <= .18 and (not best_gap or gap < best_gap)
                            and min_x < x+shape[4] and max_x > x+shape[1]
                            and min_z < z+shape[6] and max_z > z+shape[3] then
                            support, best_gap = {support_xyz={x,y,z}, support_top_y=top}, gap
                        end
                    end
                end
            end
        end
    end
    return support
end

local function pose(r, player)
    if r.relocate then
        local foot = xyz(r.foot_xyz)
        player:set_properties({is_visible=false})
        local natural_sink = v3_scene and r.anchor_kind == "underwater_floor"
        local observer = {gravity=natural_sink and 1 or 0, speed=0, jump=0}
        if v3_scene then observer.liquid_sink = natural_sink and 1 or 0 end
        player:set_physics_override(observer)
        player:set_pos(foot)
    end
    clean_hud(player)
    player:set_look_horizontal(math.rad(r.yaw))
    player:set_look_vertical(math.rad(r.pitch))
    reply(r, {status="posed"})
end

local function snapshot(r, player)
    local low = xyz(r.low_xyz)
    local foot = vector.add(low, 24)
    local ok, payload = pcall(voxwm_region, low)
    if not ok then reply(r, {status="snapshot_failed", reason=tostring(payload)}); return end
    local underwater = v3_scene and r.anchor_kind == "underwater_floor"
    local support = underwater and resting_support(player)
    if (underwater and not support) or (not underwater and not anchor_valid(foot, r.anchor_kind)) then
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
    local state = {status="captured", world_time=mt.get_timeofday(),
        weather=mcl_weather.get_weather(), gamma=tonumber(mt.settings:get("display_gamma")),
        rgb_enabled=not mt.settings:get_bool("voxwm_skip_rgb"),
        engine_us=mt.get_us_time()}
    if v3_scene then
        state.native_render_settings = {
            viewing_range=tonumber(mt.settings:get("viewing_range")),
            enable_fog=mt.settings:get_bool("enable_fog"),
            directional_colored_fog=mt.settings:get_bool("directional_colored_fog"),
            fog_start=tonumber(mt.settings:get("fog_start")),
            display_gamma=tonumber(mt.settings:get("display_gamma")),
            lighting_floor=tonumber(mt.settings:get("lighting_floor")),
        }
        if mt.settings:get_bool("voxwm_material_lighting_probe", false) then
            state.native_render_settings.solid_ambient_floor = tonumber(mt.settings:get("solid_ambient_floor"))
        end
        if mt.settings:get_bool("voxwm_fog_probe", false) then
            state.native_render_settings.voxwm_exact_draw_range = mt.settings:get_bool("voxwm_exact_draw_range")
            local fog_end = tonumber(mt.settings:get("fog_end_distance"))
            if fog_end and fog_end > 0 then
                state.native_render_settings.fog_start = nil
                state.native_render_settings.fog_start_distance = tonumber(mt.settings:get("fog_start_distance"))
                state.native_render_settings.fog_end_distance = fog_end
            end
        end
        state.hud_wielditem_hidden = player:hud_get_flags().wielditem == false
        state.wielded_item = player:get_wielded_item():get_name()
        state.underwater_support = support or nil
    end
    reply(r, state)
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
    if mt.settings:get_bool("voxwm_exact_draw_range", false) then
        local sky = player:get_sky(true)
        local fog = sky.fog or {}
        if fog.fog_distance ~= -1 or fog.fog_start ~= -1 then
            sky.fog = {fog_distance=-1, fog_start=-1, fog_color=fog.fog_color}
            player:set_sky(sky)
        end
    end
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
    if r.kind ~= "snapshot" and r.kind ~= "emerge_capture" and r.kind ~= "restore_replay_nodes" then
        mt.settings:set_bool("voxwm_skip_rgb", r.kind ~= "pose")
    end
    if r.kind == "search" then
        reply(r, find_site(r))
    elseif v3_scene and r.kind == "search_v3" then
        reply(r, r.scene_kind == "underwater" and find_water_site(r) or {status="found", site={r.x,r.z}})
    elseif r.kind == "survey" or (v3_scene and r.kind == "emerge_capture") then
        survey(r)
    elseif r.kind == "surface_biomes" then
        local names = {}
        for i, cell in ipairs(r.positions) do names[i] = surface_biome(xyz(cell)) end
        reply(r, {status="surface_biomes", names=names})
    elseif r.kind == "anchor" then
        check_anchor(r)
    elseif replay_restore and r.kind == "restore_replay_nodes" then
        -- Reconstruct saved Node ID and param2 only in the disposable RGB replay.
        for _, node in ipairs(r.nodes) do
            mt.swap_node(xyz(node), {name=node[4], param2=node[5]})
        end
        reply(r, {status="saved_nodes_restored"})
    elseif r.kind == "pose" then
        pose(r, player)
    elseif r.kind == "snapshot" then
        snapshot(r, player)
    else
        error("Unknown Vox2Pix native request: " .. tostring(r.kind))
    end
end}
