-- One fixed 48³ region, no discovery cache or raw trajectory. Original ABMs are untouched.
local mt=minetest;local mod=mt.get_modpath('craftium_env');local p0=dofile(mod..'/native.lua')
local world=mt.get_worldpath();local elapsed,step,last=0,0,0;local ready,armed,started=false,false,false
local region=nil;local held={};local reactions={}
local function clock()return {sim_time=elapsed,simulation_step=step,engine_us=mt.get_us_time()}end
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
    p0.handle({kind='p0_capture',origins=region.origins,surface_y=region.surface_y,verify=true},nil,function(_,data)
        if extra then for k,v in pairs(extra)do data[k]=v end end
        data.reaction_events=reactions;reactions={};data.center_origin=region.center;reply(r,data)
    end)
end
local function prepare(r)
    for _,p in ipairs(held)do mt.forceload_free_block(p,true)end;held={};reactions={}
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
            if not active then reply(r,{status='activation_failed'})else capture(r,{status='prepared'})end
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
        if p.x>=c.x and p.x<c.x+16 and p.z>=c.z and p.z<c.z+16 and p.y>=c.y and p.y<c.y+24 then
            local row={position=enu(p),attached=support}
            if support then attached[#attached+1]=row else suspended[#suspended+1]=row end
            if under and not wet then fire[#fire+1]={position=enu(p),under=enu(under)};count=count+1 end
        end
    end
    local rng=PcgRandom(r.seed or 42)
    local function subset(list)
        local out={};for i=1,math.min(#list,256)do local j=rng:next(i,#list);list[i],list[j]=list[j],list[i];out[#out+1]=list[i]end;return out
    end
    reply(r,{status='inspected',attached=subset(attached),suspended=subset(suspended),fire=fire,fire_legal_count=count})
end
mt.register_on_joinplayer(function(player)
    player:set_physics_override({gravity=0,speed=0,jump=0});player:set_properties({is_visible=false,pointable=false,collide_with_objects=false})
    mt.emerge_area(vector.subtract(player:get_pos(),16),vector.add(player:get_pos(),16),function(_,_,left)if left==0 then ready=true;write('gt_ready','ready')end end)
end)
mt.register_globalstep(function(dt)
    elapsed=elapsed+dt;step=step+1;local player=mt.get_connected_players()[1];if not player then return end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if ready and not armed and read('gt_arm')then armed=true end
    if armed and not started and read('gt_start')then
        local nodes,records,seen={},{},{}
        for name,def in pairs(mt.registered_nodes)do
            local id=mt.get_content_id(name)
            if not seen[id]then seen[id]=true;local n=mt.get_name_from_content_id(id);local d=mt.registered_nodes[n]or def
                nodes[#nodes+1]={id,n,d.walkable~=false,(d.groups and(d.groups.water or 0)>0)or false,d.liquidtype or'none'}
                records[#records+1]={id=id,name=n,groups=d.groups or{},walkable=d.walkable,liquid_range=d.liquid_range,liquid_viscosity=d.liquid_viscosity,floodable=d.floodable or false}
            end
        end
        write('pilot_boot.json',{nodes=records,liquid_update=mt.settings:get('liquid_update')or'1.0'});write('gt_nodes.json',nodes);started=true
    end
    if started then
        local text=read('pilot_command.json')
        if text then local r=assert(mt.parse_json(text))
            if r.request_id>last then last=r.request_id
                if r.kind=='locate'then locate(r)
                elseif r.kind=='prepare'then prepare(r)
                elseif r.kind=='capture'then capture(r)
                elseif r.kind=='clock'then reply(r,{status='clock'})
                elseif r.kind=='inspect'then inspect(r)
                elseif r.kind=='intervene'then
                    reactions={};local actions={}
                    for _,a in ipairs(r.actions)do
                        local p=eng(a.position);local before=mt.get_node(p).name
                        if before=='air'then
                            if a.type=='fire'then mcl_fire.set_fire({type='node',under=eng(a.under),above=p},player)
                            else mt.set_node(p,{name=a.type=='water'and'mcl_core:water_source'or'mcl_core:lava_source'})end
                        end
                        actions[#actions+1]={type=a.type,position=a.position,before=before,after=mt.get_node(p).name,clock=clock()}
                    end
                    capture(r,{actions=actions})
                elseif r.kind=='entity_probe'then
                    mt.add_entity(player:get_pos(),'mobs_mc:pig');mt.add_item(player:get_pos(),'mcl_core:dirt');reply(r,{status='probe'})
                else error('Unknown sharded collection command')end
            end
        end
    end
    if mt.settings:get_bool('voxel_obs')then get_voxel_data_cpp(player:get_pos(),{x=24,y=24,z=24})end
end)
