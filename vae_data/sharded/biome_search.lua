-- Seeded native biome lookup without emerging terrain, like VoxeLibre findbiome.
-- A hit is only a candidate; Python verifies the actual surface after loading it.
local M={}
local function group_of(name,groups)
    local best,length=nil,-1
    for group,aliases in pairs(groups)do for _,alias in ipairs(aliases)do
        if name==alias or name:sub(1,#alias+1)==alias..'_'then
            if #alias>length or(#alias==length and group>best)then best,length=group,#alias end
        end
    end end
    return best
end
function M.find(mt,r)
    local resolution,limit=r.resolution,r.max_checks
    local water=tonumber(mt.get_mapgen_setting('water_level'))or 0
    local playable=(tonumber(mt.get_mapgen_setting('mapgen_limit'))or 31000)
        -((tonumber(mt.get_mapgen_setting('chunksize'))or 5)+1)*16-64
    local ids,heights,seen={},{},{}
    for name,def in pairs(mt.registered_biomes)do
        if group_of(name,r.biome_groups)==r.biome_group then
            local lo,hi=math.max(def.y_min or -31000,-96),math.min(def.y_max or 31000,191)
            if lo<=hi then
                ids[mt.get_biome_id(name)]=true
                local y=math.max(lo,math.min(water+8,hi))
                if not seen[y]then heights[#heights+1]=y;seen[y]=true end
            end
        end
    end
    table.sort(heights,function(a,b)
        local da,db=math.abs(a-water-8),math.abs(b-water-8)
        return da<db or(da==db and a<b)
    end)
    local result={status='biome_not_found',method='native_biome_noise_spiral',biome_group=r.biome_group,
        origin={r.x,r.z},resolution=resolution,max_checks=limit,checked_points=0,biome_queries=0}
    if #heights==0 then result.status='biome_unregistered';return result end
    local x,z=math.floor(r.x/16)*16+8,math.floor(r.z/16)*16+8
    local dirs={{0,1},{-1,0},{0,-1},{1,0}}
    local edge,passed,turns,direction=1,0,0,1
    for _=1,limit do
        if math.abs(x)<=playable and math.abs(z)<=playable then
            result.checked_points=result.checked_points+1
            for _,y in ipairs(heights)do
                local data=mt.get_biome_data({x=x,y=y,z=z});result.biome_queries=result.biome_queries+1
                if data and ids[data.biome]then
                    result.status='found';result.site={x-8,z-8};result.query_y=y
                    result.biome=mt.get_biome_name(data.biome);return result
                end
            end
        end
        if passed==edge then
            passed=0;turns=turns+1;direction=direction%4+1;edge=math.floor(turns/2)+1
        end
        x=x+dirs[direction][1]*resolution;z=z+dirs[direction][2]*resolution;passed=passed+1
    end
    return result
end
return M
