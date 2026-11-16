// Pure state decoding. Geometry depends on exact Node Name + its definition, never a color category.
export const LIQUID_LEVEL_MASK=0x07,LIQUID_FLOW_DOWN_MASK=0x08;
export function tokenTable(vocabulary,definitions,colors=[],flammable=[]){
 const fuel=new Set(flammable);
 return vocabulary.tokens.map((entry,i)=>{
  if(entry.id!==i)throw Error('Non-contiguous Combined Token vocabulary');
  const known=definitions.nodes[entry.node_name],d=known||{drawtype:'normal',liquidtype:'none',groups:{}};
  const color=colors[i];if(color&&(color.name!==entry.node_name||color.param2!==entry.param2))throw Error('Viewer mapping differs from original vocabulary');
  return{id:entry.id,name:entry.node_name,param2:entry.param2,definition:d,definitionKnown:!!known,
   air:d.drawtype==='airlike',liquidType:d.liquidtype||'none',fire:d.drawtype==='firelike',
   leaves:(d.groups?.leaves||0)>0,flammable:fuel.has(entry.node_name)||(d.groups?.flammable||0)>0,
   color:d.color||color?.color||'#a29579',category:color?.category||'other'};
 });
}
export function liquidState(node){
 if(!node||node.liquidType==='none')return null;
 if(node.liquidType==='source')return{type:'source',level:null,flowDown:null,height:1,paramtype2:node.definition.paramtype2||'none'};
 const level=node.param2&LIQUID_LEVEL_MASK,flowDown=!!(node.param2&LIQUID_FLOW_DOWN_MASK),range=Math.max(1,Math.min(8,node.definition.liquid_range));
 // Pinned Craftium getLiquidNeighborhood: adjust levels to the node's liquid_range.
 const adjusted=Math.max(0,level-(8-range));
 return{type:'flowing',level,flowDown,height:(adjusted+.5)/range,paramtype2:'flowingliquid'};
}
export function localXYZ(index,shape=[48,48,48]){return[Math.floor(index/(shape[1]*shape[2])),Math.floor(index/shape[2])%shape[1],index%shape[2]];}
export function worldIndex(world,origin,shape=[48,48,48]){const p=world.map((v,i)=>v-origin[i]);return p.every((v,i)=>Number.isInteger(v)&&v>=0&&v<shape[i])?(p[0]*shape[1]+p[1])*shape[2]+p[2]:null;}
export function coordinates(world,metadata,shape=[48,48,48]){const index=worldIndex(world,metadata.region_origin,shape);const chunk=world.map((p,i)=>p-metadata.center_origin[i]);return{world:[...world],regionLocal:index===null?null:world.map((p,i)=>p-metadata.region_origin[i]),centerChunkLocal:chunk.every(v=>v>=0&&v<16)?chunk:null,index};}
function at(state,table,p,shape){return p.every((v,i)=>v>=0&&v<shape[i])?table[state[(p[0]*shape[1]+p[1])*shape[2]+p[2]]]:null;}
function same(a,b){return a&&b&&a.liquidType!=='none'&&b.liquidType!=='none'&&a.definition.liquid_alternative_source===b.definition.liquid_alternative_source;}
export function liquidCorners(state,index,table,shape=[48,48,48]){
 const p=localXYZ(index,shape),node=table[state[index]];
 if(node.liquidType==='source')return[1,1,1,1];
 // ENU corners: (-x,-y),(+x,-y),(+x,+y),(-x,+y). Air and solid differ.
 return[[0,0],[1,0],[1,1],[0,1]].map(([cx,cy])=>{
  let sum=0,count=0,air=0;
  for(const dx of[cx-1,cx])for(const dy of[cy-1,cy]){
   const q=[p[0]+dx,p[1]+dy,p[2]],neighbor=at(state,table,q,shape),above=at(state,table,[q[0],q[1],q[2]+1],shape);
   if(neighbor&&neighbor.name!=='ignore'&&same(node,above))return 1;
   if(same(node,neighbor)&&neighbor.liquidType==='source')return 1;
   if(same(node,neighbor)){sum+=liquidState(neighbor).height;count++;}else if(neighbor?.name==='air')air++;
  }
  if(air>=2)return .02; // engine -0.5 +0.2/BS, BS=10, converted to bottom-relative height
  return count?sum/count:.5;
 });
}
export function fireAttachments(state,index,table,shape=[48,48,48]){
 const p=localXYZ(index,shape),support=q=>{const n=at(state,table,q,shape);return !!n&&!n.air&&!n.fire&&(n.definition.walkable!==false||n.flammable);};
 const bottom=support([p[0],p[1],p[2]-1]);const walls=[];
 for(const [dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]])if(support([p[0]+dx,p[1]+dy,p[2]]))walls.push([dx,dy]);
 return{floor:bottom,walls,ceiling:support([p[0],p[1],p[2]+1]),cross:bottom||!walls.length};
}
export function transitionType(a,b,cooling=false){
 if(!a?.name||!b?.name)return'outside_observed_region';
 if(a.id===b.id)return'unchanged';
 if(a.fire&&a.name===b.name)return a.definition.param2_semantics==='age'?'fire_age_update':'fire_param2_update';
 if(a.name==='air'&&b.fire)return'air_to_fire';
 if(a.leaves&&b.fire)return'leaves_to_fire';
 if(a.leaves&&b.name==='air')return'leaves_to_air';
 if(a.fire&&b.name==='air')return'fire_to_air';
 if(cooling)return'confirmed_cooling_product';
 if(a.liquidType==='source'&&b.name==='air')return'source_loss_requires_write_evidence';
 if(a.liquidType==='flowing'&&b.name==='air')return'flowing_liquid_recession_observed';
 if(a.liquidType==='flowing'&&a.name===b.name)return'liquid_level_or_flow_down_update';
 if(a.liquidType==='source'&&a.name===b.name)return'source_param2_update';
 if(b.liquidType!=='none'&&a.name!=='air'&&a.liquidType==='none')return(a.definition.groups?.destroy_by_lava_flow>0&&b.definition.groups?.lava>0)?'lava_replaced_lava_destructible_node_observed':'solid_to_liquid_requires_write_evidence';
 if(a.name==='air'&&b.liquidType==='flowing')return'flowing_liquid_appearance_observed';
 if(b.fire)return'node_to_fire';if(b.name==='air')return'node_removed_cause_unrecorded';return'node_state_change';
}
export function relevantChange(category,index,before,after,table,shape=[48,48,48]){
 const a=table[before[index]],b=table[after[index]];if(a.id===b.id)return false;
 const liquidKind=(n,k)=>(n.definition.groups?.[k]||0)>0;
 if(category==='D1')return liquidKind(a,'water')||liquidKind(b,'water');
 if(category==='D2')return(liquidKind(a,'lava')||liquidKind(b,'lava'))&&!['mcl_core:stone','mcl_core:cobble','mcl_core:obsidian'].includes(b.name);
 if(category==='D3'){
  if(a.fire&&b.fire&&a.name===b.name)return false;
  if(a.fire||b.fire)return true;
  if(!(a.flammable||b.flammable))return false;
  const p=localXYZ(index,shape);
  return[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]].some(d=>{const q=p.map((v,i)=>v+d[i]);return at(before,table,q,shape)?.fire||at(after,table,q,shape)?.fire;});
 }
 if(category==='D4')return['water','lava'].some(k=>liquidKind(a,k)||liquidKind(b,k))||(['mcl_core:stone','mcl_core:cobble','mcl_core:obsidian'].includes(b.name)&&!['mcl_core:stone','mcl_core:cobble','mcl_core:obsidian'].includes(a.name));
 return false;
}
export function transitionColor(type){if(type.includes('requires_'))return'#ff3862';return({air_to_fire:'#ffd24d',leaves_to_fire:'#ff693b',leaves_to_air:'#b6e1ff',fire_to_air:'#76b6ed',fire_age_update:'#bd8fea'})[type]||'#e47f75';}
