-- Discover natural plants by world root / connected vine component. Never plant or set metadata.
local mt=minetest;local P={}
local function enu(p)return {p.x,p.z,p.y}end
local function engine(p)return {x=p[1],y=p[3],z=p[2]}end
local function bamboo(name)return name=='mcl_bamboo:bamboo_endcap'or mcl_bamboo.is_bamboo(name)~=false end
local function same(name,subtype)return subtype=='bamboo'and bamboo(name)or subtype=='reeds'and name=='mcl_core:reeds'end
local function key(p)return mt.pos_to_string(p)end
function P.scan(low,high,subtype)
    local result,seen={},{}
    if subtype~='vine'then
        local names=subtype=='bamboo'and table.copy(mcl_bamboo.bamboo_index)or{'mcl_core:reeds'}
        if subtype=='bamboo'then names[#names+1]='mcl_bamboo:bamboo_endcap'end
        for _,p in ipairs(mt.find_nodes_in_area(low,high,names))do
            local root=vector.new(p);local complete=true
            for _=1,128 do
                local q=vector.offset(root,0,-1,0);local n=mt.get_node(q).name
                if n=='ignore'then complete=false;break end
                if same(n,subtype)then root=q else break end
            end
            if same(mt.get_node(vector.offset(root,0,-1,0)).name,subtype)then complete=false end
            if not seen[key(root)]then
                seen[key(root)]=true;local members={};local q=vector.new(root)
                for _=1,128 do
                    local node=mt.get_node(q)
                    if node.name=='ignore'then complete=false;break end
                    if not same(node.name,subtype)then break end
                    members[#members+1]=enu(q);q=vector.offset(q,0,1,0)
                end
                if #members==128 then complete=false end
                result[#result+1]={id=key(root),subtype=subtype,root=enu(root),members=members,height=#members,
                    complete=complete,root_metadata=mt.get_meta(root):to_table()}
            end
        end
    else
        local scan_low,scan_high=vector.subtract(low,16),vector.add(high,16)
        local nodes=mt.find_nodes_in_area(scan_low,scan_high,{'mcl_core:vine'})
        local lookup={};for _,p in ipairs(nodes)do lookup[key(p)]=p end
        local dirs={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}
        for _,p in ipairs(nodes)do
            if not seen[key(p)]then
                local queue={p};seen[key(p)]=true;local i=1;local members,anchors={},{};local intersects=false;local boundary=false
                while i<=#queue do
                    local q=queue[i];i=i+1;members[#members+1]=enu(q)
                    if q.x==scan_low.x or q.x==scan_high.x or q.y==scan_low.y or q.y==scan_high.y or q.z==scan_low.z or q.z==scan_high.z then boundary=true end
                    if q.x>=low.x and q.x<=high.x and q.y>=low.y and q.y<=high.y and q.z>=low.z and q.z<=high.z then intersects=true end
                    local node=mt.get_node(q);local d=mt.wallmounted_to_dir(node.param2%8)
                    if d and mcl_core.supports_vines(mt.get_node(vector.add(q,d)).name)then anchors[#anchors+1]=enu(q)end
                    for _,d2 in ipairs(dirs)do
                        local r=vector.offset(q,d2[1],d2[2],d2[3]);local k=key(r)
                        if lookup[k]and not seen[k]then seen[k]=true;queue[#queue+1]=r end
                    end
                end
                if intersects then result[#result+1]={id=key(p),subtype='vine',members=members,anchors=anchors,complete=#anchors>0 and not boundary,unresolved_boundary=boundary}end
            end
        end
    end
    return result
end
function P.trim(action,player)
    local row={plant_id=action.plant.id,subtype=action.plant.subtype,retain_height=action.retain_height,
        original_height=action.plant.height,root=action.plant.root,cut_positions=action.cut_positions or{}}
    local root=action.plant.root and engine(action.plant.root)
    local before=root and mt.get_meta(root):to_table()or nil
    for _,a in ipairs(row.cut_positions)do
        local p=engine(a);local node=mt.get_node(p)
        if node.name~='air'and node.name~='ignore'then mt.node_dig(p,node,player)end
    end
    local remaining={}
    for _,a in ipairs(action.plant.members)do
        local n=mt.get_node(engine(a)).name
        if same(n,action.plant.subtype)or action.plant.subtype=='vine'and n=='mcl_core:vine'then remaining[#remaining+1]=a end
    end
    row.actual_remaining_members=remaining;row.actual_remaining_count=#remaining
    row.root_preserved=not root or same(mt.get_node(root).name,action.plant.subtype)
    row.root_metadata_before=before;row.root_metadata_after=root and mt.get_meta(root):to_table()or nil
    row.metadata_unchanged=mt.write_json(row.root_metadata_before)==mt.write_json(row.root_metadata_after)
    row.matches_requested_height=action.plant.subtype=='vine'or #remaining==action.retain_height
    row.anchors_remaining=0
    for _,a in ipairs(action.plant.anchors or{})do if mt.get_node(engine(a)).name=='mcl_core:vine'then row.anchors_remaining=row.anchors_remaining+1 end end
    return row
end
return P
