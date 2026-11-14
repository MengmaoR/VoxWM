-- Isolated P1 observer. Reuses P0's fixed-world VoxelManip capture unchanged.
local mt=minetest
local world=mt.get_worldpath();local elapsed,step=0,0
local ready,armed,started=false,false,false;local last_request=0
local mod=mt.get_modpath('craftium_env')
local p0=dofile(mod..'/native.lua');local p1=dofile(mod..'/p1_native.lua')
local function write(name,data)local f=assert(io.open(world..'/'..name,'w'));f:write(mt.write_json(data));f:close()end
local function read(name)local f=io.open(world..'/'..name,'r');if not f then return nil end;local s=f:read('*a');f:close();return s end
local function marker(name)local f=assert(io.open(world..'/'..name,'w'));f:write('ready');f:close()end
local function clock()return {sim_time=elapsed,simulation_step=step,engine_us=mt.get_us_time(),wall_time_unix=os.time()}end
local boot={mod_initialization=clock(),time_origin='P1 mod globalstep accumulator; NOT mapgen t=0'}
local generated={}
mt.register_on_generated(function(low,high,seed)
    generated[#generated+1]={low_enu={low.x,low.z,low.y},high_enu={high.x,high.z,high.y},observed_callback=clock()}
end)
local function reply(r,data)
    local t=clock();for k,v in pairs(t)do data[k]=v end
    data.request_id=r.request_id;data.boot=boot;data.generated_region_callbacks=generated;generated={}
    write('pilot_response.json',data)
end
mt.register_on_joinplayer(function(player)
    player:set_physics_override({gravity=0,speed=0,jump=0})
    local p=player:get_pos()
    mt.emerge_area(vector.subtract(p,32),vector.add(p,32),function(_,_,left)if left==0 and not ready then ready=true;marker('gt_ready')end end)
end)
mt.register_globalstep(function(dtime)
    elapsed=elapsed+dtime;step=step+1
    if not boot.first_globalstep then boot.first_globalstep=clock()end
    local player=mt.get_connected_players()[1];if not player then return end
    player:hud_set_flags({crosshair=false,basic_debug=false,chat=false})
    if ready and not armed and read('gt_arm')then armed=true end
    if armed and not started and read('gt_start')then
        local nodes,records,seen={},{},{}
        for name,def in pairs(mt.registered_nodes)do
            local id=mt.get_content_id(name)
            if not seen[id]then
                seen[id]=true;local canonical=mt.get_name_from_content_id(id);local d=mt.registered_nodes[canonical]or def
                nodes[#nodes+1]={id,canonical,d.walkable~=false,(d.groups and(d.groups.water or 0)>0)or false,d.liquidtype or'none'}
                records[#records+1]={id=id,name=canonical,groups=d.groups or{}}
            end
        end
        write('gt_nodes.json',nodes);write('pilot_boot.json',{nodes=records});marker('gt_started');started=true
    end
    if started then
        local s=read('pilot_command.json')
        if s then
            local r=assert(mt.parse_json(s))
            if r.request_id>last_request then
                last_request=r.request_id
                if r.kind:sub(1,3)=='p0_'then p0.handle(r,player,reply)
                else p1.handle(r,player,reply,clock,p0)end
            end
        end
    end
    if mt.settings:get_bool('voxel_obs')then get_voxel_data_cpp(player:get_pos(),{x=24,y=24,z=24})end
end)
