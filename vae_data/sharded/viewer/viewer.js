// Standalone actual-GT viewer. No model, no voxel interpolation, no writes to raw data.
import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {VoxelStream,timeIndex,worldToLocal} from './stream.js';
import {tokenTable,liquidState,liquidCorners,localXYZ,worldIndex,coordinates,transitionType,relevantChange,transitionColor} from './node_state.js';
import {addNativeShapes,clearGroup} from './native_geometry.js';
const $=id=>document.getElementById(id),initial=new URLSearchParams(location.search);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('canvas').appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#15212c');
const camera=new THREE.PerspectiveCamera(48,1,.1,500);camera.position.set(65,52,62);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,1.6));const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(20,60,35);scene.add(sun);
const voxels=new THREE.Group(),boxes=new THREE.Group(),markers=new THREE.Group(),overlays=new THREE.Group(),ghosts=new THREE.Group(),selection=new THREE.Group();scene.add(voxels,boxes,markers,overlays,ghosts,selection);
const cube=new THREE.BoxGeometry(.98,.98,.98),matrix=new THREE.Matrix4();
let manifest,data,stream,table,index=0,playing=false,playStart=0,playTime=0,arrivals=[],reference=null,baseTime=0,origin=[0,0,0],loadId=0,firstLoad=true,pinned=null,displayed=null,pair=null;
const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=.18;
const point=p=>new THREE.Vector3(...worldToLocal(p,origin));
function clear(g){clearGroup(g,cube);}
function drawBounds(){clear(boxes);for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
 const center=x===0&&y===0&&z===0,line=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(16,16,16)),new THREE.LineBasicMaterial({color:center?0xf1d66e:0x617787,transparent:true,opacity:center?1:.28,depthTest:!center}));line.position.set(x*16,z*16,-y*16);boxes.add(line);
}}
function permitted(x,y,z){
 const low=$('region').value==='16'?16:0,high=48-low;
 if(!(x>=low&&x<high&&y>=low&&y<high&&z>=low&&z<high&&[x,y,z][Number($('axis').value)]<=Number($('slice').value)))return false;
 return !($('isolate').checked&&pinned)||[x,y,z].every((p,i)=>Math.abs(p+origin[i]-pinned[i])<=3);
}
function addCubes(group,indices,color,opacity=1,wireframe=false){
 if(!indices.length)return;
 const material=new THREE.MeshLambertMaterial({color,transparent:opacity<1,opacity,wireframe,depthWrite:opacity===1});
 const mesh=new THREE.InstancedMesh(cube,material,indices.length);
 indices.forEach((p,i)=>{const[x,y,z]=localXYZ(p);matrix.makeTranslation(x+.5-24,z+.5-24,-(y+.5-24));mesh.setMatrixAt(i,matrix);});
 mesh.instanceMatrix.needsUpdate=true;mesh.userData.voxelIndices=indices;group.add(mesh);
}
function addPoints(indices,color,size,kind='diagnostic',frameIndex=null){
 if(!indices.length)return;const positions=indices.flatMap(id=>{const[x,y,z]=localXYZ(id);return[x+.5-24,z+.5-24,-(y+.5-24)];});
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
 const m=new THREE.Points(g,new THREE.PointsMaterial({color,size,transparent:true,opacity:.8,depthTest:false}));m.userData={voxelIndices:indices,kind,frameIndex};overlays.add(m);
}
function currentPair(){
 const cache=new Map(),at=i=>{i=Math.max(0,Math.min(data.frames.length-1,i));if(!cache.has(i))cache.set(i,{index:i,values:stream.frame(i).slice(),metadata:data.frames[i].metadata});return cache.get(i);};
 if(!$('freeze').checked){$('beforeFrame').value=Math.max(0,index-1);$('afterFrame').value=index;}
 const before=at(Number($('beforeFrame').value)||0),after=at(Number($('afterFrame').value)||0);
 if(before.index>after.index)throw Error('Before frame must precede After frame');
 const display=$('mode').value==='changes'?($('pairView').value==='before'?before:after):at(index);
 return{before,after,display,previous:display.index?at(display.index-1):null};
}
function updateInspector(){
 if(!pinned||!displayed||!pair)return;
 const state=s=>{if(!s)return null;const id=worldIndex(pinned,s.metadata.region_origin);return id===null?{outsideObservedRegion:true}:{...table[s.values[id]],sim_time:s.metadata.sim_time,frame:s.index};};
 const current=state(displayed),before=state(pair.before),after=state(pair.after),previous=state(pair.previous);
 const coord=coordinates(pinned,displayed.metadata),flow=liquidState(current),afterId=worldIndex(pinned,pair.after.metadata.region_origin);
 const target=afterId!==null&&!!coordinates(pinned,pair.after.metadata).centerChunkLocal&&relevantChange(data.unique_type.split('-')[0],afterId,pair.before.values,pair.after.values,table);
 const brief=n=>n&&!n.outsideObservedRegion?`${n.name} [param2=${n.param2}, token=${n.id}] @ ${n.sim_time.toFixed(6)}s (#${n.frame})`:n?'outside observed region':'no previous observation';
 const result={world_coordinate:pinned,region_local:coord.regionLocal,center_chunk_local:coord.centerChunkLocal,
  combined_token_id:current?.id??null,node_name:current?.name??null,raw_param2:current?.param2??null,
  node_type:current?.definitionKnown?current.definition.drawtype:'unknown (semantic cube fallback)',definition_known:current?.definitionKnown??false,
  liquid_type:current?.definitionKnown?current.liquidType:'unknown',liquid_level:flow?.level??null,flow_down_flag:flow?.flowDown??null,
  source_param2_semantics:current?.liquidType==='source'?current.definition.paramtype2:null,
  fire_age:current?.definition?.param2_semantics==='age'?current.param2:null,
  previous_node_state:brief(previous),current_node_state:brief(current),before_node_state:brief(before),after_node_state:brief(after),
  observed_transition:transitionType(before,after),native_simulation_time:displayed.metadata.sim_time,
  center_valid_dynamic_in_pair:target,ghost_is_current_GT:false};
 $('inspector').textContent=JSON.stringify(result,null,2);$('world').value=pinned.join(', ');
 clear(selection);if(coord.index!==null){const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04,1.04,1.04)),new THREE.LineBasicMaterial({color:'#faff87',depthTest:false}));box.position.copy(point(pinned));selection.add(box);}
 window.nativeViewerState.inspector=result;
}
function renderState(){
 if(!stream||!data)return;
 try{
  for(const g of[voxels,markers,overlays,ghosts,selection])clear(g);
  pair=currentPair();displayed=pair.display;origin=displayed.metadata.region_origin;const state=displayed.values,groups=new Map(),native=$('mode').value!=='semantic';
  for(let id=0;id<state.length;id++){
   const c=table[state[id]],[x,y,z]=localXYZ(id);if(c.air||!permitted(x,y,z))continue;
   if(native&&(c.liquidType!=='none'||c.fire||c.definition.drawtype==='plantlike'))continue;
   let exposed=false;
   for(const[dx,dy,dz]of[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]){
    const a=x+dx,b=y+dy,k=z+dz;
    if(!permitted(a,b,k)){exposed=true;break;}
    const n=table[state[(a*48+b)*48+k]];
    if(n.air||native&&(n.liquidType!=='none'||n.fire||n.definition.drawtype==='plantlike')){exposed=true;break;}
   }
   if(!exposed)continue;
   const opacity=$('highlight').value==='all'||$('highlight').value===c.category?1:.16,key=c.color+'/'+opacity;
   if(!groups.has(key))groups.set(key,{color:c.color,opacity,indices:[]});groups.get(key).indices.push(id);
  }
  for(const g of groups.values())addCubes(voxels,g.indices,g.color,g.opacity);
  const shapes=native?addNativeShapes(voxels,state,table,permitted,$('highlight').value):null;
  const expected=[],actual=[],changed=[],burnt=[],b=pair.before.values,a=pair.after.values,elapsed=Math.max(0,displayed.metadata.sim_time-baseTime),changeGroups=new Map();
  for(let id=0;id<state.length;id++){
   if(!permitted(...localXYZ(id)))continue;
   if(['predicted','both'].includes($('overlay').value)&&arrivals.some(g=>g[id]!==255&&g[id]<=Math.min(10,elapsed)))expected.push(id);
   if(displayed.metadata.sim_time>=baseTime&&state[id]!==reference[id]&&['actual','both'].includes($('overlay').value))actual.push(id);
   if(b[id]!==a[id]){
    changed.push(id);const type=transitionType(table[b[id]],table[a[id]]);
    if(!changeGroups.has(type))changeGroups.set(type,[]);changeGroups.get(type).push(id);
    if(table[b[id]].leaves&&(table[a[id]].fire||table[a[id]].name==='air'))burnt.push(id);
   }
  }
  addPoints(expected,'#49a8ff',.17);addPoints(actual,'#ff574f',.25);
  if($('mode').value==='changes'&&$('pairView').value==='difference')for(const[type,ids]of changeGroups)addPoints(ids,transitionColor(type),.34,'transition',pair.after.index);
  if($('ghost').checked){addCubes(ghosts,burnt,'#a8e5ff',.4,true);for(const m of ghosts.children){m.userData.isGhost=true;m.material.depthTest=false;m.renderOrder=20;}}
  $('transition').replaceChildren();const blank=document.createElement('option');blank.value='';blank.textContent=`${changed.length} observed changes in view`; $('transition').append(blank);
  for(const[type,ids]of changeGroups)for(const id of ids){const o=document.createElement('option');o.value=id;const xyz=localXYZ(id).map((p,i)=>p+pair.after.metadata.region_origin[i]);o.textContent=`${xyz.join(',')} · ${table[b[id]].name} → ${table[a[id]].name} · ${type}`;$('transition').append(o);}
  if($('sources').checked)for(const source of data.planning?.selected_sources||[]){const marker=new THREE.Mesh(new THREE.SphereGeometry(.3,10,8),new THREE.MeshBasicMaterial({color:source.kind==='water'?0x50c6ff:source.kind==='lava'?0xff962e:0xffd448,depthTest:false}));marker.position.copy(point(source.position));markers.add(marker);}
  boxes.visible=$('bounds').checked;$('sliceLabel').value=$('slice').value;$('timeline').value=index;
  const row=data.frames[displayed.index];$('time').textContent=`${displayed.metadata.sim_time.toFixed(6)} sim s`;
  $('readout').textContent=`Timeline #${index}; displaying original observation #${displayed.index}/${data.frames.length-1} · native step ${row.metadata.simulation_step}\nCenter raw changes: ${row.center_changes} · neighbor: ${row.neighbor_changes} · roles: ${row.stages.join(', ')}${data.selected_indices.includes(displayed.index)?' · ACCEPTED VAE STATE':''}\nOrigin ENU: ${origin.join(', ')} · ghost previous leaves: ${$('ghost').checked?burnt.length:0} (not current GT)\nPair ${pair.before.index} → ${pair.after.index}: ${[...changeGroups].map(([type,ids])=>`${type}: ${ids.length}`).join(' | ')}`;
  window.nativeViewerState={type:$('type').value,attempt:$('attempt').value,index:displayed.index,timelineIndex:index,time:row.metadata.sim_time,step:row.metadata.simulation_step,frames:data.frames.length,shape:stream.shape,selectedIndices:data.selected_indices,source:'actual_native_GT',mode:$('mode').value,shapes,ghostCount:$('ghost').checked?burnt.length:0,transitionCounts:Object.fromEntries([...changeGroups].map(([k,v])=>[k,v.length]))};
  updateInspector();$('error').textContent='';
 }catch(e){$('error').textContent=e.stack||String(e);}
}
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${url}`);return r.json();}
async function load(){
 const token=++loadId;playing=false;$('play').textContent='Play';$('error').textContent='';stream=null;data=null;pinned=null;
 for(const g of[voxels,markers,overlays,ghosts,selection])clear(g);
 const type=manifest.types.find(t=>t.id===$('type').value),entry=(type?.attempts||[]).find(a=>a.attempt===$('attempt').value)||type;
 if(!entry?.available){$('status').textContent=entry?.reason||'No native snapshot';return;}
 try{
  const path=`/data/viewer/${entry.path}`;
  const[m,v,c,defs,buffer]=await Promise.all([json(path+'/frames.json'),json(path+'/vocabulary.json'),json(path+'/classes.json'),json('/native_nodes.json'),fetch(path+'/voxels.bin').then(r=>{if(!r.ok)throw Error('Missing voxel stream');return r.arrayBuffer();})]);
  const grids=await Promise.all((m.planning?.selected_sources||[]).map(async(_,i)=>{const r=await fetch(`${path}/planning/selected_${i}.bin`);if(!r.ok)throw Error('Missing selected-source estimate');return new Uint8Array(await r.arrayBuffer());}));
  if(token!==loadId)return;
  data=m;stream=new VoxelStream(buffer);table=tokenTable(v,defs,c,m.frames[0].metadata.flammable_names||[]);arrivals=grids;
  if(stream.frames!==m.frames.length||stream.shape.some(v=>v!==48))throw Error('Voxel and metadata shape/index mismatch');
  if(m.frames.some(f=>f.metadata.region_origin.some((v,i)=>v!==m.frames[0].metadata.region_origin[i])))throw Error('Moving native region: this fixed-region comparison cannot assume equal local indices');
  index=0;const ri=Math.max(0,data.frames.findIndex(f=>f.stages.includes('after_intervention')));reference=stream.frame(ri).slice();baseTime=data.frames[ri].metadata.sim_time;
  const status=m.selection.status||{};$('status').textContent=`${type.id} / ${entry.attempt}\n${status.status}: ${status.reason||'accepted native chain'}\nRollout: ${m.frames[0].metadata.rollout_id}${m.planning.native_intervention_executed===false?'\nOffline plan only; proposed sources NOT executed':''}`;
  $('planning').textContent=JSON.stringify(m.planning,null,2);$('timeline').max=m.frames.length-1;for(const id of['beforeFrame','afterFrame'])$(id).max=m.frames.length-1;
  $('stages').replaceChildren();const notes=[];m.frames.forEach((row,i)=>{const roles=row.stages.filter(s=>!['native_observation','original_scene'].includes(s));if(m.selected_indices.includes(i))roles.push('VAE');if(roles.length){const o=document.createElement('option');o.value=i;o.label=roles.join('/');$('stages').append(o);notes.push(`#${i} ${row.metadata.sim_time.toFixed(3)}s ${roles.join('/')}`);}});$('markers').textContent=notes.join(' · ');
  $('contact').href=path+'/contact_sheet.png';$('gif').href=path+'/evolution.gif';$('report').href=path+'/report.md';
  if(firstLoad){
   for(const[id,values]of[['region',['16','48']],['overlay',['actual','predicted','both']],['mode',['semantic','native','changes']],['pairView',['before','after','difference']]])if(values.includes(initial.get(id)))$(id).value=initial.get(id);
   index=Math.max(0,Math.min(m.frames.length-1,Number(initial.get('frame'))||0));
   if(initial.has('before')&&initial.has('after')){$('beforeFrame').value=Number(initial.get('before'));$('afterFrame').value=Number(initial.get('after'));$('freeze').checked=true;}
   if(initial.get('world')){const w=initial.get('world').split(',').map(Number);if(w.length===3&&w.every(Number.isInteger))pinned=w;}
   if(initial.get('isolate')==='1')$('isolate').checked=true;
   firstLoad=false;
  }
  renderState();
 }catch(e){if(token===loadId)$('error').textContent=e.stack||String(e);}
}
function attempts(){const entry=manifest.types.find(t=>t.id===$('type').value);$('attempt').replaceChildren();for(const a of entry?.attempts||[]){const o=document.createElement('option');o.value=a.attempt;o.textContent=a.attempt;$('attempt').append(o);}$('attempt').value=entry?.attempt||'';}
function jump(i){if(!data)return;playing=false;$('play').textContent='Play';index=Math.max(0,Math.min(data.frames.length-1,i));renderState();}
$('play').onclick=()=>{if(!data)return;playing=!playing;$('play').textContent=playing?'Pause':'Play';if(playing&&index===data.frames.length-1){index=0;renderState();}playStart=performance.now();playTime=data.frames[index].metadata.sim_time;};
$('back').onclick=()=>jump(index-1);$('next').onclick=()=>jump(index+1);$('timeline').oninput=()=>jump(Number($('timeline').value));
for(const id of['mode','pairView','region','highlight','axis','slice','overlay','bounds','sources','ghost','isolate','freeze','beforeFrame','afterFrame'])$(id).onchange=renderState;
$('slice').oninput=renderState;
$('transition').onchange=()=>{if($('transition').value==='')return;const i=Number($('transition').value);pinned=localXYZ(i).map((p,k)=>p+pair.after.metadata.region_origin[k]);updateInspector();};
$('pin').onclick=()=>{const p=$('world').value.replace(/[\[\]]/g,'').trim().split(/[,\s]+/).map(Number);if(p.length!==3||!p.every(Number.isInteger)){$('error').textContent='Enter three integer ENU world coordinates';return;}pinned=p;renderState();};
$('focus').onclick=()=>{if(!pinned||!displayed)return;const target=point(pinned),id=worldIndex(pinned,origin),n=id===null?null:table[displayed.values[id]];if(n?.definition.drawtype==='plantlike')target.y-=.34;else if(n?.liquidType==='flowing'){const h=liquidCorners(displayed.values,id,table).reduce((a,b)=>a+b,0)/4;target.y+=h/2-.5;}controls.target.copy(target);camera.position.copy(target.clone().add(new THREE.Vector3(7,5,7)));controls.update();renderState();};
$('reset').onclick=()=>{camera.position.set(65,52,62);controls.target.set(0,0,0);controls.update();};
$('type').onchange=()=>{attempts();$('freeze').checked=false;load();};$('attempt').onchange=()=>{$('freeze').checked=false;load();};
$('speed').onchange=()=>{if(data){playStart=performance.now();playTime=data.frames[index].metadata.sim_time;}};
let down=null;renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
renderer.domElement.addEventListener('pointerup',e=>{
 if(!displayed||!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;
 const r=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);
 const candidates=[...voxels.children,...overlays.children].filter(m=>!m.userData.skipInspector);
 for(const hit of raycaster.intersectObjects(candidates,false)){
  const id=hit.object.userData.voxelIndices?.[hit.instanceId??hit.index??hit.faceIndex];
  if(id===undefined)continue;const fi=hit.object.userData.frameIndex??displayed.index;
  pinned=localXYZ(id).map((p,k)=>p+data.frames[fi].metadata.region_origin[k]);updateInspector();break;
 }
});
new ResizeObserver(()=>{const r=$('canvas').getBoundingClientRect();renderer.setSize(r.width,r.height);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();}).observe($('canvas'));
function tick(now){requestAnimationFrame(tick);if(playing&&data){const t=playTime+(now-playStart)/1000*Number($('speed').value),next=timeIndex(data.frames,t);if(next!==index){index=next;renderState();}if(t>=data.frames.at(-1).metadata.sim_time){playing=false;$('play').textContent='Play';}}controls.update();renderer.render(scene,camera);}requestAnimationFrame(tick);
drawBounds();try{manifest=await json('/data/viewer/manifest.json');for(const entry of manifest.types){const o=document.createElement('option');o.value=entry.id;o.textContent=entry.id+(entry.available?'':' (no observation)');$('type').append(o);}if(manifest.types.some(t=>t.id===initial.get('type')))$('type').value=initial.get('type');attempts();if(initial.get('attempt'))$('attempt').value=initial.get('attempt');await load();}catch(e){$('error').textContent=String(e);}
