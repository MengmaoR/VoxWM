-- An observation probe in an untouched native world. Only explicitly requested fire ignition writes a node.
local mt=minetest
local world=mt.get_worldpath()
local seed=tonumber(mt.settings:get("fixed_map_seed")) or 42
math.randomseed(seed);mt.set_mapgen_setting("seed",tostring(seed),true)
local ready,armed,started=false,false,false
local elapsed,last_request=0,0
local region={surface_y=0}
local radius={x=24,y=24,z=24}
local function read(name)
    local f=io.open(world.."/"..name,"r");if not f then return nil end
    local text=f:read("*a");f:close();return text
end
local function write(name,data)
    local f=assert(io.open(world.."/"..name,"w"));f:write(assert(mt.write_json(data)));f:close()
end
local function marker(name)local f=assert(io.open(world.."/"..name,"w"));f:write("ready");f:close() end
local function biome(p)
    local d=mt.get_biome_data(p)
    return d and mt.get_biome_name(d.biome) or "UNRESOLVED"
end
local function reply(request,data)
    data.request_id=request.request_id;data.sim_time=elapsed;data.game_time=mt.get_gametime()
    data.engine_us=mt.get_us_time();data.wall_time_unix=os.time();write("pilot_response.json",data)
end
local function visit(request,player)
    local x,z=request.x,request.z
    local p1={x=x-32,y=-96,z=z-32};local p2={x=x+64,y=256,z=z+64}
    mt.emerge_area(p1,p2,function(_,_,remaining)
        if remaining~=0 then return end
        local height=nil
        for y=256,-96,-1 do
            local n=mt.get_node({x=x+24,y=y,z=z+24});local d=mt.registered_nodes[n.name]
            if d and n.name~="ignore" and n.name~="air" and mt.get_item_group(n.name,"leaves")==0 and mt.get_item_group(n.name,"tree")==0 and (d.walkable or d.liquidtype~="none" and d.liquidtype~=nil) then height=y;break end
        end
        if not height then reply(request,{status="no_surface"});return end
        local target=request.target
        local offsets={surface=-16,subsurface=-32,underground=-48,canopy=16}
        local offset=request.layer=="underground" and -(request.depth or 48) or (offsets[request.layer] or 0)
        local bottom=math.floor((height+offset)/16)*16
        local focus=nil
        local nodes=target=="water" and {"group:water"} or target=="lava" and {"group:lava"} or target=="fire" and {"group:fire"} or target=="vegetation" and {"group:bamboo","group:vine"} or nil
        if nodes then
            local found=mt.find_nodes_in_area({x=x,y=bottom,z=z},{x=x+47,y=bottom+47,z=z+47},nodes)
            if #found==0 and target=="vegetation" then found=mt.find_nodes_in_area({x=x,y=bottom,z=z},{x=x+47,y=bottom+47,z=z+47},{"group:sapling","group:plant","group:flora"}) end
            if #found>0 then focus=found[math.random(#found)] end
        end
        local center={x=x+24,y=bottom+24,z=z+24}
        if focus then center={x=math.floor(focus.x/16)*16+8,y=math.floor(focus.y/16)*16+8,z=math.floor(focus.z/16)*16+8} end
        -- Probe position may jump between sites; environment liquids/ABMs/timers are never paused.
        player:set_pos(center)
        region={surface_y=height,layer=request.layer,target=target}
        mt.emerge_area(vector.subtract(center,32),vector.add(center,32),function(_,_,left)
            if left==0 then reply(request,{status="ready",surface_y=height,center=center,biome=biome({x=center.x,y=height,z=center.z}),focus=focus}) end
        end)
    end)
end
local function ignite(request,player)
    local p=player:get_pos();local found=mt.find_nodes_in_area(vector.subtract(p,20),vector.add(p,20),{"group:flammable"})
    local ignition=nil
    for _=1,math.min(#found,64) do
        local under=found[math.random(#found)]
        local around={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}
        for _,d in ipairs(around) do
            local above={x=under.x+d[1],y=under.y+d[2],z=under.z+d[3]}
            if mt.get_node(above).name=="air" then
                mcl_fire.set_fire({type="node",under=under,above=above},player)
                if mt.get_item_group(mt.get_node(above).name,"fire")>0 then ignition=above;break end
            end
        end
        if ignition then break end
    end
    reply(request,{status=ignition and "ignited" or "no_combustible_site",ignition=ignition})
end
local function sample(request)
    local chunks={}
    for _,o in ipairs(request.origins) do
        local p1={x=o[1],y=o[3],z=o[2]};local p2=vector.add(p1,15)
        local center=vector.add(p1,8)
        local metadata={}
        for _,p in ipairs(mt.find_nodes_with_meta(p1,p2)) do
            local timer=mt.get_node_timer(p)
            metadata[#metadata+1]={local_enu={p.x-p1.x,p.z-p1.z,p.y-p1.y},data=mt.get_meta(p):to_table(),
                                  timer={timeout=timer:get_timeout(),elapsed=timer:get_elapsed()}}
        end
        local rel=center.y-region.surface_y
        local terrain=rel < -24 and "underground" or rel < -8 and "subsurface" or rel>24 and "aboveground" or "surface"
        chunks[table.concat(o,",")]={biome=biome(center),surface_biome=biome({x=center.x,y=region.surface_y,z=center.z}),
            terrain=terrain,height=center.y,surface_height=region.surface_y,relative_height=rel,
            landform=region.surface_y<=1 and "lowland_or_water" or region.surface_y>=80 and "highland" or "land",
            block_metadata=metadata}
    end
    reply(request,{status="sample",chunks=chunks})
end
mt.register_on_joinplayer(function(player)
    player:set_physics_override({gravity=0,speed=0,jump=0}) -- stationary observer, not a physics change to the world
    local p=player:get_pos()
    mt.emerge_area(vector.subtract(p,32),vector.add(p,32),function(_,_,n)if n==0 and not ready then ready=true;marker("gt_ready") end end)
end)
mt.register_globalstep(function(dtime)
    elapsed=elapsed+dtime
    local player=mt.get_connected_players()[1];if not player then return end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if ready and not armed and read("gt_arm") then player:set_look_horizontal(0);player:set_look_vertical(0);armed=true end
    if armed and not started and read("gt_start") then
        local nodes,records,seen={},{},{}
        for name,def in pairs(mt.registered_nodes) do
            local id=mt.get_content_id(name)
            if not seen[id] then
                seen[id]=true;local canonical=mt.get_name_from_content_id(id);local d=mt.registered_nodes[canonical] or def
                nodes[#nodes+1]={id,canonical,d.walkable~=false,(d.groups and (d.groups.water or 0)>0) or false,d.liquidtype or "none"}
                records[#records+1]={id=id,name=canonical,groups=d.groups or {}}
            end
        end
        table.sort(nodes,function(a,b)return a[1]<b[1]end)
        marker("gt_started");write("gt_nodes.json",nodes)
        local biomes={};for name,_ in pairs(mt.registered_biomes)do biomes[#biomes+1]=name end;table.sort(biomes)
        write("pilot_boot.json",{nodes=records,biomes=biomes,enable_fire=mt.settings:get_bool("enable_fire",true)})
        started=true
    end
    if started then
        local text=read("pilot_command.json")
        if text then
            local request=assert(mt.parse_json(text))
            if request.request_id>last_request then
                last_request=request.request_id
                if request.kind=="visit" then visit(request,player)
                elseif request.kind=="ignite" then ignite(request,player)
                elseif request.kind=="sample" then sample(request)
                else error("Unknown pilot command") end
            end
        end
    end
    if mt.settings:get_bool("voxel_obs")then get_voxel_data_cpp(player:get_pos(),radius)end
end)
