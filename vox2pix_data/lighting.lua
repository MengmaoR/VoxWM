-- V3 observes the pinned game's lighting without writing any world nodes.
local mt = minetest
local M = {}
local torch = "mcl_torches:torch"

function M.configure(player, enabled)
    assert(mt.registered_items[torch], "Native torch item is unavailable")
    player:set_wielded_item(ItemStack(enabled and torch or ""))
end

function M.observe(player)
    local pos = vector.round(vector.add(player:get_pos(), {x=0, y=1.6, z=0}))
    local light, param1 = {}, {}
    for _, offset in ipairs({{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}) do
        local at = vector.offset(pos, unpack(offset))
        light[#light+1] = mt.get_node_light(at) or -1
        param1[#param1+1] = mt.get_node(at).param1
    end
    local item = player:get_wielded_item():get_name()
    return {item=item, enabled=item == torch, item_light_source=mt.registered_items[torch].light_source or 0,
        module_available=mt.get_modpath("vl_wieldlight") ~= nil,
        setting_enabled=mt.settings:get_bool("enable_vl_wieldlight", false),
        hidden=player:hud_get_flags().wielditem == false, light=light, param1=param1}
end

function M.settings()
    return {viewing_range=tonumber(mt.settings:get("viewing_range")),
        enable_fog=mt.settings:get_bool("enable_fog"),
        directional_colored_fog=mt.settings:get_bool("directional_colored_fog"),
        fog_start=tonumber(mt.settings:get("fog_start")),
        display_gamma=tonumber(mt.settings:get("display_gamma"))}
end

return M
