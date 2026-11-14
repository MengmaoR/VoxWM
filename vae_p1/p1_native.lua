-- P1 permits ONE initialization phase, followed only by unchanged native simulation.
local mt=minetest
local E=dofile(mt.get_modpath('craftium_env')..'/eligibility.lua')
local P={};local region=nil;local held={};local initialized={}
local function rounded(p)return vector.round(p)end
local function floor16(n)return math.floor(n/16)*16 end
local function scan_positions(request)
    local low,high=region.low,region.high;local names
    if request.setting=='C'then
        if request.category=='D3'then names={'group:flammable'}
        elseif request.category=='D4'and request.subtype=='bamboo'then names=mcl_bamboo.bamboo_dirt_nodes
        elseif request.category=='D4'and request.subtype=='reeds'then names={'group:soil_sugarcane'}
        else names={'group:soil','group:stone','group:tree','group:leaves','group:sand','mcl_core:stone','mcl_core:gravel'}end
        local sources=mt.find_nodes_in_area(low,high,names);local points,seen={},{}
        local dirs={{0,1,0},{1,0,0},{-1,0,0},{0,0,1},{0,0,-1},{0,-1,0}}
        for _,p in ipairs(sources)do
            for k,d in ipairs(dirs)do
                local vertical_plant=request.category=='D4'and request.subtype~='vine'
                if not vertical_plant or k==1 then
                    local q=vector.offset(p,d[1],d[2],d[3]);local key=mt.pos_to_string(q)
                    if not seen[key]and mt.get_node(q).name=='air'then seen[key]=true;points[#points+1]=q end
                end
            end
        end
        return points,#sources
    elseif request.category=='D1'then names={'group:water'}
    elseif request.category=='D2'then names={'group:lava'}
    elseif request.category=='D3'then names={'group:fire'}
    elseif request.subtype=='bamboo'then names=mcl_bamboo.bamboo_index
    elseif request.subtype=='reeds'then names={'mcl_core:reeds'}
    else names={'mcl_core:vine'}end
    local points=mt.find_nodes_in_area(low,high,names);return points,#points
end
local function candidates(request,player,reply)
    local found,total_sources=scan_positions(request);local eligible,rejected,seen={},{},{}
    local rng=PseudoRandom(request.selection_seed);local n=#found;local examined=math.min(n,request.budget or 256)
    for i=1,examined do
        local j=rng:next(i,n);found[i],found[j]=found[j],found[i]
        local candidate,why=E.check(found[i],request,region)
        if candidate then
            local key=table.concat(candidate.position,',')
            if not seen[key]then seen[key]=true;eligible[#eligible+1]=candidate else rejected['same_plant_root']=(rejected['same_plant_root']or 0)+1 end
        else rejected[why]=(rejected[why]or 0)+1 end
    end
    reply(request,{status='discovered',discovery_source_nodes=total_sources,discovered=n,examined=examined,eligible=#eligible,rejected=rejected,candidates=eligible})
end
local function prepare(request,player,reply,clock)
    for _,p in ipairs(held)do mt.forceload_free_block(p,true)end;held={}
    local x,z=request.site.x,request.site.z;local lifecycle={emerge_requested=clock()}
    mt.emerge_area({x=x-16,y=-96,z=z-16},{x=x+47,y=192,z=z+47},function(_,_,remaining)
        if remaining~=0 then return end
        lifecycle.emerge_completed=clock();local surface=nil
        for y=192,-96,-1 do
            local p={x=x+8,y=y,z=z+8};local n=mt.get_node(p);local d=mt.registered_nodes[n.name]
            if d and n.name~='air'and n.name~='ignore'and mt.get_item_group(n.name,'tree')==0 and mt.get_item_group(n.name,'leaves')==0 and (d.walkable or d.liquidtype and d.liquidtype~='none')then surface=y;break end
        end
        if not surface then reply(request,{status='no_surface',lifecycle=lifecycle});return end
        local center_y=surface-(request.site.depth or 0)
        local low={x=x,y=floor16(center_y)-16,z=z};local high={x=x+31,y=low.y+47,z=z+31}
        lifecycle.activation_requested=clock()
        mt.emerge_area(low,high,function(_,_,left)
            if left~=0 then return end
            local origins={};local active=true
            for dx=0,1 do for dz=0,1 do for dy=0,2 do
                local p=vector.offset(low,dx*16,dy*16,dz*16)
                origins[#origins+1]=E.enu(p)
                if mt.forceload_block(p,true)then held[#held+1]=p else active=false end
            end end end
            player:set_pos(vector.add(low,16));region={low=low,high=high,surface_y=surface}
            lifecycle.activation_completed=clock()
            reply(request,{status=active and 'ready'or'activation_failed',origins=origins,surface_y=surface,
                bounds_enu={E.enu(low),E.enu(high)},lifecycle=lifecycle})
        end)
    end)
end
local function recheck(request,player,reply)
    local checked={};local count=0
    for _,candidate in ipairs(request.candidates)do
        local now,why=E.check(E.engine(candidate.position),request,region)
        checked[#checked+1]={position=candidate.position,eligible=now~=nil,reason=why,features=now and now.features or nil}
        if now then count=count+1 end
    end
    reply(request,{status='checked',eligible=count,candidates=checked,can_continue_growth=count>0})
end
local function intervene(request,player,reply,clock,p0)
    local previous=initialized[request.scene_id]or 0
    if previous>0 and not(request.fallback and previous==1 and request.category=='D3'and #request.candidates==1)then error('Repeated initialization is not allowed')end
    initialized[request.scene_id]=previous+1
    local results={}
    for _,candidate in ipairs(request.candidates)do
        local p=E.engine(candidate.position);local checked,why=E.check(p,request,region)
        local row={position=candidate.position,before=mt.get_node(p).name,clock=clock(),eligible=checked~=nil,reason=why}
        if checked then
            if request.category=='D3'then mcl_fire.set_fire({type='node',under=E.engine(checked.under),above=p},player)
            elseif request.category=='D4'then
                local item=ItemStack(checked.placement);local def=mt.registered_items[item:get_name()]
                if def and def.on_place then def.on_place(item,player,{type='node',under=E.engine(checked.under),above=p})end
                -- Native on_place assigns bamboo's random cap; we never set it or call growth.
            else mt.set_node(p,{name=checked.placement})end
            local n=mt.get_node(p);row.after=n.name;row.param2=n.param2
            row.success=request.category=='D4' and (request.subtype=='bamboo' and mcl_bamboo.is_bamboo(n.name)~=false or n.name==checked.placement)or n.name==checked.placement
            if request.subtype=='bamboo'then row.native_assigned_height=mt.get_meta(p):get_int('height')end
        else row.success=false end
        results[#results+1]=row
    end
    -- Capture immediately in this callback: no Python round-trip before post-intervention state.
    p0.handle({kind='p0_capture',request_id=request.request_id,origins=request.origins,surface_y=region.surface_y},player,function(r,data)
        data.interventions=results;data.intervention_completed=clock();reply(request,data)
    end)
end
function P.handle(request,player,reply,clock,p0)
    if request.kind=='p1_prepare'then prepare(request,player,reply,clock)
    elseif request.kind=='p1_candidates'then candidates(request,player,reply)
    elseif request.kind=='p1_recheck'then recheck(request,player,reply)
    elseif request.kind=='p1_intervene'then intervene(request,player,reply,clock,p0)
    elseif request.kind=='p1_clock'then reply(request,{status='clock'})
    else error('Unknown P1 command')end
end
return P
