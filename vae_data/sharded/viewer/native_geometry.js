import * as THREE from 'three';
import {liquidCorners,liquidState,localXYZ,fireAttachments} from './node_state.js';

function worldPoint(p){return[p[0]-24,p[2]-24,-(p[1]-24)];}
function triangle(store,owner,a,b,c){store.positions.push(...worldPoint(a),...worldPoint(b),...worldPoint(c));store.owners.push(owner);}
function quad(s,id,a,b,c,d){triangle(s,id,a,b,c);triangle(s,id,a,c,d);}
function disposeMaterial(material){if(material.userData?.temporaryMap)material.map?.dispose();material.dispose();}
export function clearGroup(group,sharedGeometry){for(const mesh of[...group.children]){group.remove(mesh);if(mesh.geometry!==sharedGeometry)mesh.geometry.dispose();disposeMaterial(mesh.material);}}
let flameTexture;
function fireTexture(){
 if(flameTexture)return flameTexture;
 const canvas=document.createElement('canvas');canvas.width=64;canvas.height=96;const c=canvas.getContext('2d');
 const g=c.createLinearGradient(0,96,0,5);g.addColorStop(0,'#fff9ae');g.addColorStop(.42,'#ffc344');g.addColorStop(1,'#fa6429');c.fillStyle=g;
 c.beginPath();c.moveTo(9,95);c.bezierCurveTo(1,70,24,68,18,36);c.bezierCurveTo(25,48,25,49,28,62);c.bezierCurveTo(38,49,25,21,42,2);c.bezierCurveTo(39,29,63,48,51,65);c.bezierCurveTo(65,74,61,95,50,96);c.closePath();c.fill();
 flameTexture=new THREE.CanvasTexture(canvas);return flameTexture;
}
export function addNativeShapes(group,state,table,permitted,highlight='all'){
 const liquids=new Map(),firePlanes=new Map(),plants=new Map();const sources=[],downward=[];
 const get=(map,key,props)=>{if(!map.has(key))map.set(key,{positions:[],owners:[],...props});return map.get(key);};
 for(let id=0;id<state.length;id++){
  const node=table[state[id]],[x,y,z]=localXYZ(id);if(!permitted(x,y,z)||node.air)continue;
  if(node.liquidType!=='none'){
   const family=node.definition.liquid_alternative_source;
   const blocked=(dx,dy,dz)=>{if(!permitted(x+dx,y+dy,z+dz))return false;const q=table[state[((x+dx)*48+y+dy)*48+z+dz]];return !!q&&(q.definition.liquid_alternative_source===family||q.definition.walkable===true&&!q.leaves);};
   const top=liquidCorners(state,id,table),s=get(liquids,node.name,{node}),a=[x,y,z+top[0]],b=[x+1,y,z+top[1]],c=[x+1,y+1,z+top[2]],d=[x,y+1,z+top[3]];
   if(!blocked(0,0,1))quad(s,id,a,b,c,d);
   if(!blocked(0,-1,0))quad(s,id,[x,y,z],[x+1,y,z],b,a);
   if(!blocked(1,0,0))quad(s,id,[x+1,y,z],[x+1,y+1,z],c,b);
   if(!blocked(0,1,0))quad(s,id,[x+1,y+1,z],[x,y+1,z],d,c);
   if(!blocked(-1,0,0))quad(s,id,[x,y+1,z],[x,y,z],a,d);
   if(!blocked(0,0,-1))quad(s,id,[x,y,z],[x,y+1,z],[x+1,y+1,z],[x+1,y,z]);
   if(liquidState(node)?.flowDown){const h=top.reduce((a,b)=>a+b,0)/4;downward.push([x+.5,y+.5,z+Math.max(.04,h*.8)],[x+.5,y+.5,z+.02]);}
   if(node.liquidType==='source'&&!blocked(0,0,1))sources.push({id,p:[x+.5,y+.5,z+1.015],color:node.color});
  }else if(node.fire){
   const s=get(firePlanes,node.name,{node}),support=fireAttachments(state,id,table);
   if(support.cross){quad(s,id,[x+.08,y+.08,z],[x+.92,y+.92,z],[x+.92,y+.92,z+1],[x+.08,y+.08,z+1]);quad(s,id,[x+.08,y+.92,z],[x+.92,y+.08,z],[x+.92,y+.08,z+1],[x+.08,y+.92,z+1]);}
   for(const[dx,dy]of support.walls){const cx=x+.5+dx*.43,cy=y+.5+dy*.43;quad(s,id,[cx-dy*.43,cy+dx*.43,z],[cx+dy*.43,cy-dx*.43,z],[cx+dy*.3-dx*.15,cy-dx*.3-dy*.15,z+1],[cx-dy*.3-dx*.15,cy+dx*.3-dy*.15,z+1]);}
  }else if(node.definition.drawtype==='plantlike'){
   const s=get(plants,node.name,{node});const stem=.07,cap=.23;
   // Small mushroom geometry, not a dirt cube. Exact identity remains in the inspector.
   quad(s,id,[x+.5-stem,y+.5-stem,z],[x+.5+stem,y+.5-stem,z],[x+.5+stem,y+.5-stem,z+.3],[x+.5-stem,y+.5-stem,z+.3]);
   quad(s,id,[x+.5-cap,y+.5-cap,z+.32],[x+.5+cap,y+.5-cap,z+.32],[x+.5+cap,y+.5+cap,z+.32],[x+.5-cap,y+.5+cap,z+.32]);
  }
 }
 function mesh(s,material,uv=false){if(!s.positions.length){material.dispose();return;}if(highlight!=='all'&&s.node.category!==highlight){material.transparent=true;material.opacity*=.16;} const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(s.positions,3));g.computeVertexNormals();if(uv){const coords=[];for(let i=0;i<s.owners.length;i+=2)coords.push(0,0,1,0,1,1,0,0,1,1,0,1);g.setAttribute('uv',new THREE.Float32BufferAttribute(coords,2));}const m=new THREE.Mesh(g,material);m.userData.voxelIndices=s.owners;m.userData.nativeShape=s.node.definition.drawtype;group.add(m);}
 for(const s of liquids.values())mesh(s,new THREE.MeshLambertMaterial({color:s.node.color,side:THREE.DoubleSide,transparent:true,opacity:s.node.definition.groups?.water ? .7 : .96,depthWrite:false}));
 for(const s of firePlanes.values())mesh(s,new THREE.MeshBasicMaterial({map:fireTexture(),side:THREE.DoubleSide,transparent:true,alphaTest:.15,depthWrite:false}),true);
 for(const s of plants.values())mesh(s,new THREE.MeshLambertMaterial({color:s.node.color,side:THREE.DoubleSide}));
 if(sources.length){
  const positions=sources.flatMap(s=>worldPoint(s.p)),g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  const marks=new THREE.Points(g,new THREE.PointsMaterial({color:'#e1f8ff',size:.12,depthTest:false}));marks.userData.voxelIndices=sources.map(s=>s.id);marks.userData.sourceMarks=true;group.add(marks);
 }
 if(downward.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(downward.flatMap(worldPoint),3));const lines=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:'#f7eabd'}));lines.userData.skipInspector=true;group.add(lines);}
 return{liquidTypes:[...liquids.keys()],fireTypes:[...firePlanes.keys()],plantTypes:[...plants.keys()],sourceMarks:sources.length};
}
