-- Read-only scene geometry; no simulation rules or node writes.
local G={}
function G.terrain(d)
    if not d or d.walkable==false or(d.liquidtype and d.liquidtype~='none')then return false end
    local g=d.groups or{}
    return(g.tree or 0)==0 and(g.leaves or 0)==0 and(g.wood or 0)==0 and(g.fence or 0)==0 and(g.plant or 0)==0
end
function G.largest(mask,n)
    n=n or 16;local seen={};local largest=0
    for first=1,n*n*n do if mask[first]and not seen[first]then
        local q={first};seen[first]=true;local at=1
        while at<=#q do
            local p=q[at];at=at+1;local k=p-1;local x=math.floor(k/(n*n));local y=math.floor(k/n)%n;local z=k%n
            local neighbors={x>0 and p-n*n or 0,x<n-1 and p+n*n or 0,y>0 and p-n or 0,y<n-1 and p+n or 0,z>0 and p-1 or 0,z<n-1 and p+1 or 0}
            for _,v in ipairs(neighbors)do if v>0 and mask[v]and not seen[v]then seen[v]=true;q[#q+1]=v end end
        end
        largest=math.max(largest,#q)
    end end
    return largest
end
return G
