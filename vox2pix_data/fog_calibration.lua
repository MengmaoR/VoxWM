-- Disposable fog calibration only. Native wool nodes provide a textured panel.
local mt = minetest
local world = mt.get_worldpath()
local started, last = false, 0

local function read(name)
    local file = io.open(world .. "/" .. name, "rb")
    if not file then return nil end
    local value = file:read("*a")
    file:close()
    return value
end

local function write(name, value)
    local path = world .. "/" .. name
    local file = assert(io.open(path .. ".tmp", "wb"))
    assert(file:write(type(value) == "string" and value or assert(mt.write_json(value))))
    assert(file:close())
    assert(os.rename(path .. ".tmp", path))
end

local function hide_observer(player)
    player:set_properties({is_visible=false})
    player:set_physics_override({gravity=0, speed=0, jump=0, liquid_sink=0})
    player:hud_set_flags({hotbar=false, healthbar=false, crosshair=false, wielditem=false,
        breathbar=false, minimap=false, minimap_radar=false, basic_debug=false, chat=false})
    for id in pairs(player:hud_get_all()) do player:hud_remove(id) end
end

local function native_sky(player)
    local sky = player:get_sky(true)
    sky.fog = sky.fog or {}
    if sky.fog.fog_distance ~= -1 or sky.fog.fog_start ~= -1 then
        sky.fog.fog_distance = -1
        sky.fog.fog_start = -1
        player:set_sky(sky)
    end
end

mt.register_on_joinplayer(function(player)
    hide_observer(player)
    player:set_pos({x=0, y=.4, z=0})
    player:set_look_horizontal(0)
    player:set_look_vertical(0)
    mt.set_timeofday(.5)
    if mcl_weather.get_weather() ~= "none" then
        mcl_weather.change_weather("none", nil, "Vox2Pix fog calibration")
    end
    mt.emerge_area({x=-16,y=-16,z=-16}, {x=16,y=16,z=64}, function(_, action, remaining)
        assert(action ~= mt.EMERGE_CANCELLED and action ~= mt.EMERGE_ERRORED, "Calibration emergence failed")
        if remaining ~= 0 then return end
        local colors = {"red", "yellow", "blue", "green"}
        for x=-3,3 do for y=-1,5 do
            local color = colors[(x + y + 8) % #colors + 1]
            if math.abs(x) <= 1 and math.abs(y - 2) <= 1 then color = "red" end
            local name = "mcl_wool:" .. color
            assert(mt.registered_nodes[name], "Missing native marker " .. name)
            mt.set_node({x=x,y=y,z=33}, {name=name})
        end end
        write("gt_ready", "ready\n")
    end)
end)

mt.registered_voxwm_poststeps = {function()
    local player = mt.get_connected_players()[1]
    if not player then return end
    hide_observer(player)
    native_sky(player)
    if not started and read("gt_start") and read("gt_arm") then
        local rows, seen = {}, {}
        for name in pairs(mt.registered_nodes) do
            local id = mt.get_content_id(name)
            if not seen[id] then
                seen[id] = true
                local canonical = mt.get_name_from_content_id(id)
                local def = mt.registered_nodes[canonical]
                rows[#rows + 1] = {id, canonical, def.walkable ~= false,
                    (def.groups or {}).water ~= nil, def.liquidtype or "none"}
            end
        end
        table.sort(rows, function(a,b) return a[1] < b[1] end)
        write("gt_nodes.json", rows)
        started = true
    end
    if not started then return end
    local contents = read("vox2pix_command.json")
    if not contents then return end
    local request = assert(mt.parse_json(contents))
    if request.request_id <= last then return end
    last = request.request_id
    if request.kind == "pose" then
        player:set_pos({x=request.foot_xyz[1],y=request.foot_xyz[2],z=request.foot_xyz[3]})
        player:set_look_horizontal(0)
        player:set_look_vertical(0)
    else
        assert(request.kind == "capture", "Unknown fog calibration request")
    end
    local clock = voxwm_clock()
    write("vox2pix_response.json", {request_id=last, status="captured", native_clock=clock,
        lua_tick=clock.server_tick, sim_time=clock.absolute_sim_time,
        hud_wielditem_hidden=not player:hud_get_flags().wielditem,
        wielded_item=player:get_wielded_item():get_name(),
        weather=mcl_weather.get_weather(), world_time=mt.get_timeofday()})
end}
