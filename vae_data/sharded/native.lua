-- One fixed 48³ region, no discovery cache or raw trajectory. Original ABMs are untouched.
local mt=minetest;local mod=mt.get_modpath('craftium_env');local p0=dofile(mod..'/native.lua');local geometry=dofile(mod..'/scene_geometry.lua')
local world=mt.get_worldpath();local elapsed,step,last=0,0,0;local ready,armed,started=false,false,false
assert(voxwm_clock and voxwm_region,'Build the VoxWM fixed24 timing patch before collection')
local region=nil;local pending_capture=nil;local held={};local reactions={};local zero_tick=nil;local lua_tick=0
local function clock()
    local c=voxwm_clock();local n=c.server_tick
    return {sim_time=n/24,absolute_sim_time=n/24,simulation_step=n,native_simulation_step=n,simulation_hz=24,
        frame_index=zero_tick and n-zero_tick or nil,relative_sim_time=zero_tick and(n-zero_tick)/24 or nil,
        lua_tick=lua_tick,native_clock=c,engine_us=mt.get_us_time()}
end
local function enu(p)return {p.x,p.z,p.y}end
local function eng(p)return {x=p[1],y=p[3],z=p[2]}end
local function read(name)local f=io.open(world..'/'..name);if not f then return nil end;local s=f:read('*a');f:close();return s end
local function write(name,data)
    local path=world..'/'..name;local temporary=path..'.tmp'
    local f=assert(io.open(temporary,'w'));f:write(type(data)=='string'and data or mt.write_json(data));f:close()
    assert(os.rename(temporary,path)) -- Python must never observe an empty/partial response.
end
local function reply(r,data)
    for k,v in pairs(clock())do data[k]=v end;data.request_id=r.request_id;data.audit=voxel_only.audit();write('pilot_response.json',data)
end
local dirs={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}
-- Passive observer around the original set_node. It forwards every argument/return unchanged.
-- Contact is read BEFORE the native cooling write; existing stone is never evidence.
local original_set=mt.set_node
mt.set_node=function(pos,node,...)
    local before=mt.get_node(pos).name;local contact=nil
    if node.name=='mcl_core:stone'and mt.get_item_group(before,'water')>0 then
        local q=vector.offset(pos,0,1,0);local name=mt.get_node(q).name
        if mt.get_item_group(name,'lava')>0 then contact={water=enu(pos),lava=enu(q),water_node=before,lava_node=name}end
    elseif(node.name=='mcl_core:cobble'or node.name=='mcl_core:obsidian')and mt.get_item_group(before,'lava')>0 then
        local def=mt.registered_nodes[before];local expected=def.liquidtype=='source'and'mcl_core:obsidian'or'mcl_core:cobble'
        if node.name==expected then
            for _,d in ipairs(dirs)do if d[2]>=0 then
                local q=vector.offset(pos,d[1],d[2],d[3]);local name=mt.get_node(q).name
                if mt.get_item_group(name,'water')>0 then contact={water=enu(q),lava=enu(pos),water_node=name,lava_node=before};break end
            end end
        end
    end
    local event=contact and{position=enu(pos),before=before,after=node.name,contact=contact,confirmed=true,clock=clock()}
    local result=original_set(pos,node,...)
    if event and mt.get_node(pos).name==node.name then reactions[#reactions+1]=event end
    return result
end
local function capture(r,extra)
    local function emit()
        local start=mt.get_us_time();local payload=voxwm_region(region.low)
        local out=assert(io.open(world..'/native48.bin','wb'));out:write(payload);out:close()
        local data={status='captured',chunks=region.chunks,capture_us=mt.get_us_time()-start,capture_method='post_server_step_MapBlock',
            reaction_events=reactions,center_origin=region.center};reactions={}
        if extra then for k,v in pairs(extra)do data[k]=v end end
        reply(r,data)
    end
    if not region.chunks then
        -- One setup-only P0 comparison, never 27 VoxelManip rebuilds per tick.
        p0.handle({kind='p0_capture',origins=region.origins,surface_y=region.surface_y,verify=true},nil,function(_,data)
            region.chunks=data.chunks
            assert(voxwm_region(region.low)==read('p0_capture.bin'),'C++ capture differs from the original P0 same-tick state')
            emit()
        end)
    else emit()end
end
local function prepare(r)
    for _,p in ipairs(held)do mt.forceload_free_block(p,true)end;held={};reactions={}
    zero_tick=nil
    local center=eng(r.center);local low=vector.subtract(center,16);local high=vector.add(center,31)
    mt.emerge_area(vector.subtract(low,16),vector.add(high,16),function(_,_,left)
        if left~=0 then return end
        mt.after(0,function()
            local origins={};local active=true
            for x=low.x,high.x,16 do for z=low.z,high.z,16 do for y=low.y,high.y,16 do
                origins[#origins+1]={x,z,y};local p={x=x,y=y,z=z}
                if mt.forceload_block(p,true)then held[#held+1]=p else active=false end
            end end end
            region={center=r.center,low=low,high=high,origins=origins,surface_y=r.surface_y}
            if not active then reply(r,{status='activation_failed'})else
                local protected={};local index=0;local player_name=mt.get_connected_players()[1]:get_player_name()
                for x=low.x,high.x do for z=low.z,high.z do for y=low.y,high.y do
                    if mt.is_protected({x=x,y=y,z=z},player_name)then protected[#protected+1]=index end;index=index+1
                end end end
                pending_capture={request=r,extra={status='prepared',protected_source_indices=protected}}
            end
        end)
    end)
end
local function locate(r)
    local x,z=r.x,r.z
    mt.emerge_area({x=x-24,y=-112,z=z-24},{x=x+39,y=192,z=z+39},function(_,_,left)
        if left~=0 then return end
        mt.after(0,function()
            local heights={};local points={{8,8},{-16,-16},{32,-16},{-16,32},{32,32}}
            for _,d in ipairs(points)do
                local h=nil
                for y=191,-96,-1 do
                    local n=mt.get_node({x=x+d[1],y=y,z=z+d[2]});local def=mt.registered_nodes[n.name]
                    if def and n.name~='air'and n.name~='ignore'and mt.get_item_group(n.name,'tree')==0 and mt.get_item_group(n.name,'leaves')==0 and(def.walkable or def.liquidtype~='none'and def.liquidtype~=nil)then h=y;break end
                end
                if not h then reply(r,{status='no_surface'});return end;heights[#heights+1]=h
            end
            local data=mt.get_biome_data({x=x+8,y=heights[1],z=z+8})
            reply(r,{status='located',surface_y=heights[1],heights=heights,biome=data and mt.get_biome_name(data.biome)or'UNRESOLVED'})
        end)
    end)
end
-- A bounded read-only native terrain probe. Every result is an aligned16³ chunk,
-- sampled uniformly by chunk later, never weighted by a random target fluid voxel.
local function find_centers(r)
    local ylow=math.floor(mcl_vars.mg_overworld_min/16)*16;local yhigh=191
    local low={x=r.x-16,z=r.z-16,y=ylow};local high={x=r.x+31,z=r.z+31,y=yhigh}
    mt.emerge_area(vector.subtract(low,16),vector.add(high,16),function(_,_,left)
        if left~=0 then return end
        mt.after(0,function()
            local vm=VoxelManip();local a,b=vm:read_from_map(low,high);local area=VoxelArea:new({MinEdge=a,MaxEdge=b});local data=vm:get_data()
            local terrain,water,lava={}, {}, {};local airid=mt.get_content_id('air');local ignoreid=mt.get_content_id('ignore')
            for name,d in pairs(mt.registered_nodes)do local id=mt.get_content_id(name);terrain[id]=geometry.terrain(d);water[id]=(d.groups.water or 0)>0;lava[id]=(d.groups.lava or 0)>0 end
            local function node(x,z,y)return data[area:index(x,y,z)]end
            local tops={};local boundaries={}
            for x=low.x,high.x do tops[x]={};boundaries[x]={}
                for z=low.z,high.z do
                    local first=true
                    for y=yhigh,ylow,-1 do local id=node(x,z,y)
                        if id==ignoreid then first=false
                        elseif terrain[id]then
                            tops[x][z]=y
                            if first and node(x,z,y+1)==airid then boundaries[x][z]=y end
                            break
                        elseif water[id]or lava[id]then first=false end
                    end
                end
            end
            local candidates={};local diagnostic=nil;local best_score=-1;local rejected={no_air=0,no_height=0,no_pool=0,unavailable=0};local examined=0
            for x=low.x,high.x,16 do for z=low.z,high.z,16 do for y=ylow,yhigh,16 do
                examined=examined+1;local air,under,pool={},{},{};local count=0;local boundary=0;local missing=false
                for dx=0,15 do for dz=0,15 do
                    local top=tops[x+dx][z+dz];local ground=boundaries[x+dx][z+dz]
                    if ground and ground>=y and ground<y+16 then boundary=boundary+1 end
                    for dy=0,15 do local id=node(x+dx,z+dz,y+dy);count=count+1
                        air[count]=id==airid;under[count]=air[count]and top and y+dy<top or false
                        pool[count]=r.pool_kind=='water'and water[id]or r.pool_kind=='lava'and lava[id]or false
                        if id==ignoreid then missing=true end
                    end
                end end
                local largest=geometry.largest(air);local underground=r.height_type=='cave'and geometry.largest(under)or 0
                local fluid=r.pool_kind~=''and geometry.largest(pool)or 0
                local reason=nil
                local height_ok=(r.height_type=='surface'and boundary>0)or(r.height_type=='cave'and boundary==0 and underground>64)
                if missing then reason='unavailable'
                elseif largest<=64 then reason='no_air'
                elseif not height_ok then reason='no_height'
                elseif r.pool_kind~=''and fluid<=18 then reason='no_pool'end
                if reason then rejected[reason]=rejected[reason]+1 end
                local score=(height_ok and 100000 or 0)+math.min(largest,65)*100+math.min(fluid,19)
                if not reason or(not missing and score>best_score)then
                    local roof={};for dx=0,15 do roof[dx+1]={};for dz=0,15 do roof[dx+1][dz+1]=tops[x+dx][z+dz]or ylow-1 end end
                    local info={center={x,z,y},surface_y=tops[x+8][z+8]or y,solid_surface_heights=roof,
                        surface_boundary_columns=boundary,largest_connected_air_component=largest,largest_underground_air_component=underground,
                        initial_connected_pool=fluid,height_type=r.height_type,pool_kind=r.pool_kind,probe_qualified=not reason,diagnostic_rejection_reason=reason}
                    if not reason then candidates[#candidates+1]=info
                    else diagnostic=info;best_score=score end
                end
            end end end
            reply(r,{status='centers_found',candidates=candidates,diagnostic=diagnostic,examined_chunks=examined,rejected=rejected,
                probe_domain={low=enu(low),high=enu(high)},selection='uniform over qualified aligned chunks; no target-voxel sampling'})
        end)
    end)
end
local function inspect(r)
    local attached,suspended,fire={},{},{};local count=0
    for _,p in ipairs(mt.find_nodes_in_area(region.low,region.high,{'air'}))do
        local support=false;local under=nil;local wet=false
        for _,d in ipairs(dirs)do
            local q=vector.offset(p,d[1],d[2],d[3]);local n=mt.get_node(q);local def=mt.registered_nodes[n.name]
            if def and def.walkable then support=true end
            if mt.get_item_group(n.name,'flammable')>0 then under=under or q end
            if mt.get_item_group(n.name,'puts_out_fire')>0 then wet=true end
        end
        local c=eng(region.center)
        if not mt.is_protected(p,mt.get_connected_players()[1]:get_player_name())then
            local row={position=enu(p),attached=support}
            if support then attached[#attached+1]=row else suspended[#suspended+1]=row end
            if under and not wet then fire[#fire+1]={position=enu(p),under=enu(under)};count=count+1 end
        end
    end
    reply(r,{status='inspected',attached=attached,suspended=suspended,fire=fire,fire_legal_count=count})
end
mt.register_on_joinplayer(function(player)
    player:set_physics_override({gravity=0,speed=0,jump=0});player:set_properties({is_visible=false,pointable=false,collide_with_objects=false})
    mt.emerge_area(vector.subtract(player:get_pos(),16),vector.add(player:get_pos(),16),function(_,_,left)if left==0 then ready=true;write('gt_ready','ready')end end)
end)
mt.register_globalstep(function(dt)
    if dt==0 then return end -- boot initialization is not a physics tick
    assert(math.abs(dt-1/24)<1e-8,'Non-fixed native Lua dt')
    lua_tick=voxwm_clock().server_tick
end)
local function post_step(dt)
    elapsed=voxwm_clock().absolute_sim_time;step=voxwm_clock().server_tick;local player=mt.get_connected_players()[1];if not player then return end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if ready and not armed and read('gt_arm')then armed=true end
    if armed and not started and read('gt_start')then
        local nodes,records,seen={},{},{}
        for name,def in pairs(mt.registered_nodes)do
            local id=mt.get_content_id(name)
            if not seen[id]then seen[id]=true;local n=mt.get_name_from_content_id(id);local d=mt.registered_nodes[n]or def
                nodes[#nodes+1]={id,n,d.walkable~=false,(d.groups and(d.groups.water or 0)>0)or false,d.liquidtype or'none'}
                records[#records+1]={id=id,name=n,groups=d.groups or{},walkable=d.walkable,terrain_solid=geometry.terrain(d),liquid_range=d.liquid_range,liquid_viscosity=d.liquid_viscosity,floodable=d.floodable or false,buildable_to=d.buildable_to or false}
            end
        end
        write('pilot_boot.json',{nodes=records,liquid_update=mt.settings:get('liquid_update')or'1.0'});write('gt_nodes.json',nodes);started=true
    end
    if pending_capture then local job=pending_capture;pending_capture=nil;capture(job.request,job.extra)end
    if started then
        local text=read('pilot_command.json')
        if text then local r=assert(mt.parse_json(text))
            if r.request_id>last then last=r.request_id
                if r.kind=='find_centers'then find_centers(r)
                elseif r.kind=='locate'then locate(r)
                elseif r.kind=='prepare'then prepare(r)
                elseif r.kind=='capture'or r.kind=='advance'then capture(r)
                elseif r.kind=='clock'then reply(r,{status='clock'})
                elseif r.kind=='inspect'then inspect(r)
                elseif r.kind=='intervene'then
                    reactions={};local actions={};local valid=true
                    for _,a in ipairs(r.actions)do for axis=1,3 do
                        local offset=a.position[axis]-region.center[axis]
                        if offset<1 or offset>14 or offset~=math.floor(offset)then valid=false end
                    end end
                    if not valid then reply(r,{status='intervention_outside_center_14',actions={}});return end
                    for _,a in ipairs(r.actions)do
                        local p=eng(a.position);local before=mt.get_node(p).name
                        local def=mt.registered_nodes[before]
                        if not mt.is_protected(p,player:get_player_name())and(before=='air'or a.type~='fire'and def and def.buildable_to and def.liquidtype=='none')then
                            if a.type=='fire'then mcl_fire.set_fire({type='node',under=eng(a.under),above=p},player)
                            else mt.set_node(p,{name=a.type=='water'and'mcl_core:water_source'or'mcl_core:lava_source'})end
                        end
                        actions[#actions+1]={type=a.type,position=a.position,before=before,after=mt.get_node(p).name,clock=clock()}
                    end
                    zero_tick=step
                    capture(r,{actions=actions})
                elseif r.kind=='entity_probe'then
                    mt.add_entity(player:get_pos(),'mobs_mc:pig');mt.add_item(player:get_pos(),'mcl_core:dirt');reply(r,{status='probe'})
                else error('Unknown sharded collection command')end
            end
        end
    end
end
mt.registered_voxwm_poststeps={post_step}
