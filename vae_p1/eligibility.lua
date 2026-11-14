-- Read-only physical eligibility using the pinned VoxeLibre node rules.
local mt=minetest
local E={}
local dirs={{1,0,0},{-1,0,0},{0,0,1},{0,0,-1},{0,-1,0},{0,1,0}}
function E.enu(p)return {p.x,p.z,p.y}end
function E.engine(a)return {x=a[1],y=a[3],z=a[2]}end
local function at(p,d)return vector.offset(p,d[1],d[2],d[3])end
local function group(p,g)return mt.get_item_group(mt.get_node(p).name,g)>0 end
local function air(p)return mt.get_node(p).name=='air'end
local function solid(p)local d=mt.registered_nodes[mt.get_node(p).name];return d and d.walkable end
local function inside(p,low,high)return p.x>=low.x and p.x<=high.x and p.y>=low.y and p.y<=high.y and p.z>=low.z and p.z<=high.z end

function E.water_path(p)
    local queue={{p=p,path={E.enu(p)},distance=0}};local seen={[mt.pos_to_string(p)]=true};local i=1
    while i<=#queue do
        local row=queue[i];i=i+1
        if row.distance<4 then
            for k=1,5 do -- horizontal / down only; liquids cannot climb uphill
                local q=at(row.p,dirs[k]);local key=mt.pos_to_string(q)
                if not seen[key]then
                    seen[key]=true;local path=table.copy(row.path);path[#path+1]=E.enu(q)
                    if group(q,'water')then return row.distance+1,path,q end
                    if air(q)then queue[#queue+1]={p=q,path=path,distance=row.distance+1}end
                end
            end
        end
    end
    return nil,nil,nil
end

local function vine_space(p,node)
    local support=mt.wallmounted_to_dir(node.param2)
    if not support then return 0 end
    if not mcl_core.check_vines_supported(p,node)then return 0 end
    local density=#mt.find_nodes_in_area(vector.offset(p,-4,-1,-4),vector.offset(p,4,1,4),{'mcl_core:vine'})
    local count=air(vector.offset(p,0,-1,0)) and 1 or 0
    if density<5 then
        local top=vector.offset(p,0,1,0)
        -- Native up and horizontal growth use different backing directions.
        if air(top) and mcl_core.supports_vines(mt.get_node(vector.subtract(top,support)).name)then count=count+1 end
        for k=1,4 do
            local d=dirs[k];local q=at(p,d)
            if not(support.x==d[1] and support.y==d[2]) and air(q) and mcl_core.supports_vines(mt.get_node(vector.add(q,support)).name)then count=count+1 end
        end
    end
    return count,density
end

function E.check(p,request,region)
    p=vector.round(p);local node=mt.get_node(p);local controlled=request.setting=='C'
    local subtype=request.subtype or '';local under=nil;local placement=nil;local root=nil
    local height,cap,light,soil,space=0,0,mt.get_node_light(p) or 0,'',0
    if request.category=='D4' then
        if subtype=='vine' then
            if controlled then
                if not air(p)then return nil,'placement_not_air'end
                for k=1,4 do local q=at(p,dirs[k]);if mcl_core.supports_vines(mt.get_node(q).name)then under=q;break end end
                if not under then return nil,'no_vine_support_face'end
                node={name='mcl_core:vine',param2=mt.dir_to_wallmounted(vector.subtract(under,p))}
                -- Test potential extension without placing a vine or invoking growth.
                local below=vector.offset(p,0,-1,0);space=air(below) and 1 or 0
                for k=1,4 do local q=at(p,dirs[k]);if air(q) and mcl_core.supports_vines(mt.get_node(vector.add(q,vector.subtract(under,p))).name)then space=space+1 end end
            else
                if node.name~='mcl_core:vine'then return nil,'not_vine'end
                local direction=mt.wallmounted_to_dir(node.param2)
                if not direction or mt.get_node(vector.add(p,direction)).name=='ignore'then return nil,'unknown_support'end
                space=vine_space(p,node)
            end
            if space==0 then return nil,'vine_unsupported_dense_or_no_extension'end
            placement='mcl_core:vine'
        else
            local is_bamboo=function(name)return mcl_bamboo.is_bamboo(name)~=false end
            local expected=subtype=='bamboo' and is_bamboo or function(name)return name=='mcl_core:reeds'end
            root=vector.new(p)
            if controlled then
                if not air(p)then return nil,'placement_not_air'end
            else
                if not expected(node.name)then return nil,'wrong_plant'end
                for _=1,16 do local q=vector.offset(root,0,-1,0);if expected(mt.get_node(q).name)then root=q else break end end
                local tip=vector.new(root);height=1
                for _=1,16 do local q=vector.offset(tip,0,1,0);if expected(mt.get_node(q).name)then tip=q;height=height+1 else break end end
                p=root;light=mt.get_node_light(tip) or 0
                if not air(vector.offset(tip,0,1,0))then return nil,'blocked_growth_top'end
            end
            under=vector.offset(root,0,-1,0);soil=mt.get_node(under).name
            if subtype=='bamboo'then
                if not mcl_bamboo.is_dirt(soil)then return nil,'invalid_bamboo_soil'end
                cap=controlled and 16 or mt.get_meta(root):get_int('height')
                if not controlled and (cap<1 or height>=cap)then return nil,'mature_or_unknown_native_height'end
                if light<8 then return nil,'insufficient_bamboo_light'end
                placement='mcl_bamboo:bamboo'
                if root.y+16>region.high.y then return nil,'growth_outside_monitored_region'end
            else
                if mt.get_item_group(soil,'soil_sugarcane')==0 then return nil,'invalid_reeds_soil'end
                if not mt.find_node_near(root,1,{'group:water','group:frosted_ice'})then return nil,'no_water_or_frosted_ice'end
                cap=3;if height>=cap then return nil,'mature_reeds'end
                placement='mcl_core:reeds'
            end
            space=cap-height
        end
    elseif request.category=='D3' then
        if controlled then
            if not air(p)then return nil,'ignition_not_air'end
            local fuel,wet=0,0
            for _,d in ipairs(dirs)do
                local q=at(p,d)
                if group(q,'flammable')then fuel=fuel+1;under=under or q end
                if group(q,'water')or group(q,'puts_out_fire')then wet=wet+1 end
            end
            if fuel<2 then return nil,'sparse_fuel'end
            if wet>0 then return nil,'wet_ignition_site'end
            space=fuel;placement='mcl_fire:fire'
        elseif mt.get_item_group(node.name,'fire')==0 then return nil,'not_natural_fire'end
    else
        if controlled and not air(p)then return nil,'fluid_placement_not_air'end
        local native_group=request.category=='D1' and 'water' or 'lava'
        if not controlled and mt.get_item_group(node.name,native_group)==0 then return nil,'wrong_fluid'end
        local support=false
        for k,d in ipairs(dirs)do
            local q=at(p,d)
            if k<=5 and air(q)then space=space+1 end
            if solid(q)then support=true end
        end
        if space==0 then return nil,'no_flowable_air'end
        if controlled and not support then return nil,'no_terrain_attachment'end
        placement=request.category=='D1' and 'mcl_core:water_source' or 'mcl_core:lava_source'
    end
    if not inside(p,region.low,region.high)then return nil,'outside_region'end
    local distance,path,water=E.water_path(p)
    if request.category=='D2' and controlled then
        if request.interaction_context=='path_2_4' and (not distance or distance<2 or distance>4)then return nil,'no_2_to_4_cell_water_path'end
        if request.interaction_context=='immediate' and distance~=1 then return nil,'no_adjacent_native_water'end
        if request.interaction_context~='ordinary' and (not water or not inside(water,region.low,region.high))then return nil,'water_outside_observation'end
    end
    local nearby=mt.find_nodes_in_area(vector.offset(p,-1,-1,-1),vector.offset(p,1,1,1),{'group:water','group:lava'})
    local air_count,obstacles,types=0,0,{}
    for _,d in ipairs(dirs)do local q=at(p,d);local n=mt.get_node(q).name;types[n]=true;if n=='air'then air_count=air_count+1 elseif solid(q)then obstacles=obstacles+1 end end
    local diversity=0;for _ in pairs(types)do diversity=diversity+1 end
    local drop=0;for i=1,6 do if air(vector.offset(p,0,-i,0))then drop=i else break end end
    local biome=mt.get_biome_data(p);local biome_name=biome and mt.get_biome_name(biome.biome)or'UNRESOLVED'
    local boundary=math.min(p.x%16,15-p.x%16,p.y%16,15-p.y%16,p.z%16,15-p.z%16)
    local terrain=p.y<region.surface_y-16 and 'cave' or 'surface'
    local context=boundary<=1 and 'chunk_boundary' or drop>=2 and 'vertical_drop' or obstacles>=4 and 'narrow_channel' or
        obstacles>=2 and 'obstacle' or distance and 'pool_edge' or 'slope_or_open'
    local contexts={}
    if boundary<=1 then contexts[#contexts+1]='chunk_boundary'end
    if drop>=2 then contexts[#contexts+1]='vertical_drop'end
    if obstacles>=4 then contexts[#contexts+1]='narrow_channel'end
    if obstacles>=2 then contexts[#contexts+1]='obstacle'end
    if distance then contexts[#contexts+1]='pool_edge'end
    local feature={contexts=contexts,biome=biome_name,terrain=terrain,height=p.y,air_neighbors=air_count,obstacle_neighbors=obstacles,
        drop=drop,water_path_distance=distance,water_path=path,water_position=water and E.enu(water)or nil,
        nearby_fluid_nodes=#nearby,soil=soil,light=light,plant_height=height,native_height_cap=cap,
        growth_space=space,chunk_boundary_distance=boundary,local_node_diversity=diversity,context=context}
    return {position=E.enu(p),under=under and E.enu(under)or nil,subtype=subtype,placement=placement,
        node_name=mt.get_node(p).name,features=feature,stratum_key=table.concat({biome_name,terrain,context,subtype},'/')}
end
return E
