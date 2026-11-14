-- P0 reads fixed integer world boxes using the official VoxelManip API.
-- Native liquids/ABMs/timers are unchanged. Only p0_ignite may write one fire node.
local mt=minetest
local P={}
local held,ignited={},{}
local dirs={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}
local function neighbors(p)
    local air,fuel,wet=0,0,0
    for _,d in ipairs(dirs)do
        local q=vector.offset(p,d[1],d[2],d[3]);local n=mt.get_node(q)
        air=air+(n.name=="air" and 1 or 0)
        fuel=fuel+(mt.get_item_group(n.name,"flammable")>0 and 1 or 0)
        wet=wet+(mt.get_item_group(n.name,"puts_out_fire")>0 and 1 or 0)
    end
    return air,fuel,wet
end
local function discover(r,player,reply)
    local center=player:get_pos()
    local low,high=vector.subtract(center,24),vector.add(center,23)
    local names=nil
    if r.category=="D1" then names={"group:water"}
    elseif r.category=="D2" then names={"group:lava"}
    elseif r.category=="D3" then names={"group:fire"}
    elseif r.category=="D4" then
        names={"mcl_core:vine"}
        if mcl_bamboo and mcl_bamboo.bamboo_index then for _,name in ipairs(mcl_bamboo.bamboo_index)do names[#names+1]=name end end
    end
    local found=mt.find_nodes_in_area(low,high,names)
    local natural_fire=#found>0
    if r.category=="D3" and #found==0 then found=mt.find_nodes_in_area(low,high,{"group:flammable"})end
    local choices={};local examined=0
    -- Randomize the candidate subset without changing any native nodes.
    local used={}
    for _=1,math.min(#found,256)do
        local at=math.random(#found)
        if not used[at] then
            used[at]=true;examined=examined+1
            local p=found[at];local n=mt.get_node(p);local air,fuel,wet=neighbors(p)
            local valid=true;local above=nil;local subtype=nil
            if r.category=="D3" and not natural_fire then
                valid=false
                for _,d in ipairs(dirs)do
                    local q=vector.offset(p,d[1],d[2],d[3])
                    if mt.get_node(q).name=="air" then
                        local _,near_fuel,near_water=neighbors(q)
                        if not r.repair or near_fuel>=2 and near_water==0 then above=q;valid=true;fuel=near_fuel;break end
                    end
                end
            elseif r.category=="D4" then
                subtype=n.name=="mcl_core:vine" and "vine" or "bamboo"
                if subtype=="bamboo" then
                    valid=mt.get_node(vector.offset(p,0,1,0)).name=="air" and (mt.get_node_light(p) or 0)>=8
                    if r.repair then
                        local base=vector.new(p);local count=1
                        for _=1,16 do
                            local below=vector.offset(base,0,-1,0)
                            if not mcl_bamboo.is_bamboo(mt.get_node(below).name)then break end
                            base=below;count=count+1
                        end
                        local cap=mt.get_meta(base):get_int("height")
                        valid=valid and cap>count
                    end
                else
                    valid=mcl_core.check_vines_supported(p,n) and air>0
                    if r.repair then valid=valid and mt.get_node(vector.offset(p,0,-1,0)).name=="air" end
                end
            elseif r.category=="D1" or r.category=="D2" then
                if r.repair then valid=air>0 or n.name:find("flowing",1,true)~=nil end
            end
            if valid then choices[#choices+1]={pos=p,node_name=n.name,above=above,fuel_neighbors=fuel,air_neighbors=air,
                subtype=subtype,ignition_mode=r.category=="D3" and not natural_fire and "controlled_random" or "natural"}end
        end
    end
    if r.repair and r.category=="D3" then table.sort(choices,function(a,b)return a.fuel_neighbors>b.fuel_neighbors end)end
    if r.repair and r.category=="D4" then table.sort(choices,function(a,b)return a.subtype=="vine" and b.subtype~="vine" end)end
    reply(r,{status="discovered",candidate_count=#found,examined=examined,candidates=choices})
end
local function activate(r,player,reply)
    for _,p in ipairs(held)do mt.forceload_free_block(p,true)end;held={}
    local p=r.position
    local origins={}
    -- Two blocks per axis around the existing target, not a halo in any VAE sample.
    local bx=math.floor((p.x-8)/16)*16;local by=math.floor((p.y-8)/16)*16;local bz=math.floor((p.z-8)/16)*16
    for x=0,1 do for z=0,1 do for y=0,1 do origins[#origins+1]={bx+x*16,bz+z*16,by+y*16}end end end
    local low={x=bx,y=by,z=bz};local high=vector.add(low,31)
    mt.emerge_area(low,high,function(_,_,remaining)
        if remaining==0 then
            local active=true
            for _,o in ipairs(origins)do
                local q={x=o[1],y=o[3],z=o[2]}
                if mt.forceload_block(q,true)then held[#held+1]=q else active=false end
            end
            player:set_pos(vector.add(low,16))
            reply(r,{status="activated",origins=origins,all_blocks_active=active})
        end
    end)
end
local function getbiome(p)
    local d=mt.get_biome_data(p);return d and mt.get_biome_name(d.biome) or "UNRESOLVED"
end
local function surface(x,z,hint)
    for y=math.floor(hint)+32,math.floor(hint)-48,-1 do
        local n=mt.get_node({x=x,y=y,z=z});local d=mt.registered_nodes[n.name]
        if d and n.name~="air" and n.name~="ignore" and mt.get_item_group(n.name,"tree")==0 and mt.get_item_group(n.name,"leaves")==0 and (d.walkable or d.liquidtype and d.liquidtype~="none")then return y end
    end
    return nil
end
local function capture(r,reply)
    local path=mt.get_worldpath().."/p0_capture.bin"
    local out=assert(io.open(path,"wb"));local metadata={}
    for _,o in ipairs(r.origins)do
        local low={x=o[1],y=o[3],z=o[2]};local high=vector.add(low,15)
        local vm=VoxelManip();local a,b=vm:read_from_map(low,high)
        local area=VoxelArea:new({MinEdge=a,MaxEdge=b})
        local nodes,light,param=vm:get_data(),vm:get_light_data(),vm:get_param2_data()
        local cells={};local mismatch,air,solid,missing=0,0,0,0
        for x=0,15 do for z=0,15 do for y=0,15 do
            local at=area:index(low.x+x,low.y+y,low.z+z)
            local id,l,p=nodes[at],light[at],param[at]
            cells[#cells+1]=string.char(id%256,math.floor(id/256),l,p)
            local name=mt.get_name_from_content_id(id)
            if name=="air" then air=air+1 elseif name=="ignore" then missing=missing+1 else solid=solid+1 end
            if r.verify then
                local n=mt.get_node({x=low.x+x,y=low.y+y,z=low.z+z})
                if mt.get_content_id(n.name)~=id or n.param1~=l or n.param2~=p then mismatch=mismatch+1 end
            end
        end end end
        out:write(table.concat(cells))
        local center=vector.add(low,8);local h=surface(center.x,center.z,r.surface_y or center.y)
        local corners={}
        for _,d in ipairs({{0,0},{-8,-8},{8,-8},{-8,8},{8,8}})do corners[#corners+1]=getbiome({x=center.x+d[1],y=h or r.surface_y or center.y,z=center.z+d[2]})end
        local stratum="unknown"
        if h then
            if center.y<h-24 then stratum=air>200 and "deep_cave" or "deep"
            elseif center.y<h-8 then stratum="shallow_underground"
            elseif low.y>h+2 then stratum=solid>200 and "elevated" or "air"
            else stratum="surface" end
        end
        metadata[#metadata+1]={origin=o,voxel_count=4096,verified=r.verify and 4096 or 0,mismatches=mismatch,
            missing=missing,air_fraction=air/4096,solid_fraction=solid/4096,
            biome=getbiome(center),surface_biome=corners[1],biome_points=corners,
            surface_height=h,height=center.y,stratum=stratum}
    end
    out:close();reply(r,{status="captured",chunks=metadata})
end
function P.handle(r,player,reply)
    if r.kind=="p0_discover" then discover(r,player,reply)
    elseif r.kind=="p0_activate" then activate(r,player,reply)
    elseif r.kind=="p0_capture" then capture(r,reply)
    elseif r.kind=="p0_ignite" then
        if ignited[r.case_id]then error("Only one ignition is allowed per event")end
        ignited[r.case_id]=true
        mcl_fire.set_fire({type="node",under=r.under,above=r.above},player)
        reply(r,{status=mt.get_item_group(mt.get_node(r.above).name,"fire")>0 and "ignited" or "not_ignited",position=r.above})
    else error("Unknown P0 command")end
end
return P
