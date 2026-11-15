local mt=minetest;local world=mt.get_worldpath();local mod=mt.get_modpath('craftium_env')
local p0=dofile(mod..'/native.lua');local plants=dofile(mod..'/plants.lua')
local elapsed,step,last=0,0,0;local ready,armed,started=false,false,false
local region=nil;local held={};local early={};local generation={}
local auto_seq,last_auto,auto_enabled=0,-1,true
local function clock()return {sim_time=elapsed,simulation_step=step,engine_us=mt.get_us_time()}end
local boot=clock()
local function read(n)local f=io.open(world..'/'..n,'r');if not f then return nil end;local s=f:read('*a');f:close();return s end
local function write(n,d)local f=assert(io.open(world..'/'..n,'w'));f:write(type(d)=='string'and d or mt.write_json(d));f:close()end
local function enu(p)return {p.x,p.z,p.y}end
local function eng(p)return {x=p[1],y=p[3],z=p[2]}end
local function origins(low,high)
    local out={};for x=low.x,high.x,16 do for z=low.z,high.z,16 do for y=low.y,high.y,16 do out[#out+1]={x,z,y}end end end;return out
end
local function capture(r,reply,extra)
    p0.handle({kind='p0_capture',request_id=r.request_id,origins=r.origins or region.origins,surface_y=region.surface_y},nil,function(_,d)
        if extra then for k,v in pairs(extra)do d[k]=v end end;reply(r,d)
    end)
end
local function reply(r,d)
    for k,v in pairs(clock())do d[k]=v end
    d.request_id=r.request_id;d.boot=boot;d.audit=voxel_only.audit();write('pilot_response.json',d)
end
local function prepare(r,player,done)
    for _,p in ipairs(held)do mt.forceload_free_block(p,true)end;held={};early={}
    local life={emerge_requested=clock()}
    local function activate(low,high,surface)
        region={low=low,high=high,surface_y=surface,origins=origins(low,high)}
        life.activation_requested=clock()
        mt.emerge_area(vector.subtract(low,16),vector.add(high,16),function(_,_,left)
            if left~=0 then return end
            mt.after(0,function()
            local active=true
            for _,o in ipairs(region.origins)do local p=eng(o);if mt.forceload_block(p,true)then held[#held+1]=p else active=false end end
            -- Fixed-world VoxelManip needs no player teleport; forceload activates target blocks.
            life.activation_completed=clock()
            capture({origins=region.origins,request_id=r.request_id},function(_,d)
                d.status=active and'ready'or'activation_failed';d.origins=region.origins;d.surface_y=surface
                d.bounds_enu={enu(low),enu(high)};d.lifecycle=life;d.early=early;d.generation=generation;done(r,d)
            end)
            end)
        end)
    end
    if r.region then
        local low,high=eng(r.region.low),eng(r.region.high)
        region={low=low,high=high,surface_y=r.region.surface_y,origins=origins(low,high)}
        activate(low,high,r.region.surface_y)
    else
        local x,z=r.site.x,r.site.z
        mt.emerge_area({x=x-16,y=-96,z=z-16},{x=x+47,y=192,z=z+47},function(_,_,left)
            if left~=0 then return end;life.emerge_completed=clock();local surface=nil
            for y=192,-96,-1 do
                local n=mt.get_node({x=x+8,y=y,z=z+8});local def=mt.registered_nodes[n.name]
                if def and n.name~='air'and n.name~='ignore'and mt.get_item_group(n.name,'tree')==0 and mt.get_item_group(n.name,'leaves')==0 and(def.walkable or def.liquidtype and def.liquidtype~='none')then surface=y;break end
            end
            if not surface then done(r,{status='no_surface'});return end
            local y=math.floor((surface-(r.site.depth or 0))/16)*16-16
            activate({x=x,y=y,z=z},{x=x+31,y=y+47,z=z+31},surface)
        end)
    end
end
mt.register_on_generated(function(low,high)
    generation[#generation+1]={low=enu(low),high=enu(high),clock=clock()}
    if not region then return end
    -- Use the callback's existing VoxelManip. Re-reading map blocks here can deadlock emerge.
    local vm,a,b=mt.get_mapgen_object('voxelmanip')
    if not vm then return end
    local area=VoxelArea:new({MinEdge=a,MaxEdge=b})
    local nodes,light,param=vm:get_data(),vm:get_light_data(),vm:get_param2_data()
    local biomes=mt.get_mapgen_object('biomemap')
    for _,o in ipairs(region.origins)do
        local p=eng(o);local key=table.concat(o,'_')
        if not early[key]and p.x>=low.x and p.y>=low.y and p.z>=low.z and p.x+15<=high.x and p.y+15<=high.y and p.z+15<=high.z then
            local cells={};local air,missing=0,0
            for x=0,15 do for z=0,15 do for y=0,15 do
                local index=area:index(p.x+x,p.y+y,p.z+z);local id=nodes[index]
                cells[#cells+1]=string.char(id%256,math.floor(id/256),light[index],param[index])
                local name=mt.get_name_from_content_id(id)
                if name=='air'then air=air+1 elseif name=='ignore'then missing=missing+1 end
            end end end
            local file='early_'..key..'.bin';local f=assert(io.open(world..'/'..file,'wb'));f:write(table.concat(cells));f:close()
            local idx=(p.z+8-low.z)*(high.x-low.x+1)+(p.x+8-low.x)+1
            local biome=biomes and mt.get_biome_name(biomes[idx])or'UNRESOLVED'
            local center_y=p.y+8;local rel=center_y-region.surface_y
            local info={origin=o,voxel_count=4096,verified=0,mismatches=0,missing=missing,air_fraction=air/4096,solid_fraction=1-air/4096,
                biome=biome,surface_biome=biome,biome_points={biome,biome,biome,biome,biome},surface_height=region.surface_y,
                height=center_y,stratum=rel< -24 and'deep'or rel< -8 and'shallow_underground'or rel>16 and'air'or'surface'}
            early[key]={file=file,result={chunks={info}},clock=clock(),phase='mapgen_callback'}
        end
    end
end)
local dirs={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}
local function candidates(r)
    local fire,attached,suspended={},{},{};local total_fire=0
    local air=mt.find_nodes_in_area(region.low,region.high,{'air'})
    for _,p in ipairs(air)do
        local support,fuel,wet=false,0,0;local under=nil
        for _,d in ipairs(dirs)do
            local q=vector.offset(p,d[1],d[2],d[3]);local n=mt.get_node(q);local def=mt.registered_nodes[n.name]
            support=support or(def and def.walkable)or false
            if mt.get_item_group(n.name,'flammable')>0 then fuel=fuel+1;under=under or q end
            if mt.get_item_group(n.name,'puts_out_fire')>0 or mt.get_item_group(n.name,'water')>0 then wet=wet+1 end
        end
        if fuel>=1 and wet==0 then total_fire=total_fire+1;fire[#fire+1]={position=enu(p),under=enu(under),fuel=fuel}end
        local c={position=enu(p),attached=support}
        if support then attached[#attached+1]=c else suspended[#suspended+1]=c end
    end
    local rng=PcgRandom(r.seed or 42)
    local function subset(a,n)
        for i=1,math.min(#a,n)do local j=rng:next(i,#a);a[i],a[j]=a[j],a[i]end
        local out={};for i=1,math.min(#a,n)do out[i]=a[i]end;return out
    end
    local attached_subset,suspended_subset=subset(attached,512),subset(suspended,512)
    for _,list in ipairs({attached_subset,suspended_subset})do
        for _,point in ipairs(list)do
            local p=eng(point.position);local w=mt.find_node_near(p,8,{'group:water'})
            if w then point.water_distance=vector.distance(p,w);point.water_position=enu(w)end
        end
    end
    return {fire=fire,fire_legal_count=total_fire,attached=attached_subset,suspended=suspended_subset,
        attached_count=#attached,suspended_count=#suspended,plants={bamboo=plants.scan(region.low,region.high,'bamboo'),
        reeds=plants.scan(region.low,region.high,'reeds'),vine=plants.scan(region.low,region.high,'vine')}}
end
mt.register_on_joinplayer(function(player)
    player:set_physics_override({gravity=0,speed=0,jump=0});player:set_properties({is_visible=false,pointable=false,collide_with_objects=false})
    local text=mt.settings:get('vae_replay_region')
    if text then prepare({region=mt.parse_json(text)},player,function(_,d)for k,v in pairs(clock())do d[k]=v end;d.boot=boot;d.audit=voxel_only.audit();d.file='bootstrap_capture.bin'
        assert(os.rename(world..'/p0_capture.bin',world..'/'..d.file));write('bootstrap_capture.json',d);ready=true;write('gt_ready','ready')end)
    else mt.emerge_area(vector.subtract(player:get_pos(),16),vector.add(player:get_pos(),16),function(_,_,left)if left==0 then ready=true;write('gt_ready','ready')end end)end
end)
mt.register_globalstep(function(dt)
    elapsed=elapsed+dt;step=step+1
    local player=mt.get_connected_players()[1];if not player then return end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if ready and not armed and read('gt_arm')then armed=true end
    if armed and not started and read('gt_start')then
        local nodes,records,seen={},{},{}
        for name,def in pairs(mt.registered_nodes)do
            local id=mt.get_content_id(name)
            if not seen[id]then seen[id]=true;local canonical=mt.get_name_from_content_id(id);local d=mt.registered_nodes[canonical]or def
                nodes[#nodes+1]={id,canonical,d.walkable~=false,(d.groups and(d.groups.water or 0)>0)or false,d.liquidtype or'none'}
                records[#records+1]={id=id,name=canonical,groups=d.groups or{}}
            end
        end
        write('gt_nodes.json',nodes);write('pilot_boot.json',{nodes=records});started=true
    end
    if started then
        local text=read('pilot_command.json')
        if text then local r=assert(mt.parse_json(text))
            if r.request_id>last then last=r.request_id;auto_enabled=false
                if r.kind=='prepare'then prepare(r,player,reply)
                elseif r.kind=='capture'then capture(r,reply)
                elseif r.kind=='inspect'then local d=candidates(r);d.status='inspected';reply(r,d)
                elseif r.kind=='clock'then reply(r,{status='clock'})
                elseif r.kind=='intervene'then
                    local actions={}
                    for _,a in ipairs(r.actions)do
                        if a.type=='trim'then actions[#actions+1]=plants.trim(a,player)
                        else
                            local p=eng(a.position);local before=mt.get_node(p).name
                            if before=='air'then
                                if a.type=='fire'then mcl_fire.set_fire({type='node',under=eng(a.under),above=p},player)
                                else mt.set_node(p,{name=a.type=='water'and'mcl_core:water_source'or'mcl_core:lava_source'})end
                            end
                            actions[#actions+1]={type=a.type,position=a.position,before=before,after=mt.get_node(p).name,clock=clock()}
                        end
                    end
                    capture(r,reply,{actions=actions})
                elseif r.kind=='entity_probe'then
                    mt.add_entity(player:get_pos(),'mobs_mc:pig');mt.add_item(player:get_pos(),'mcl_core:dirt');reply(r,{status='probe'})
                else error('Unknown unified request')end
            end
        end
    end
    if auto_enabled and region and ready and elapsed-last_auto>=0.2 then
        last_auto=elapsed;auto_seq=auto_seq+1
        capture({request_id=-auto_seq},function(_,d)
            for k,v in pairs(clock())do d[k]=v end;d.audit=voxel_only.audit();d.file='trace_'..auto_seq..'.bin'
            assert(os.rename(world..'/p0_capture.bin',world..'/'..d.file))
            local f=assert(io.open(world..'/early_trace.jsonl','a'));f:write(mt.write_json(d)..'\n');f:close()
        end)
    end
    if mt.settings:get_bool('voxel_obs')then get_voxel_data_cpp(player:get_pos(),{x=24,y=24,z=24})end
end)
