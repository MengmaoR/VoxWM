// Adapted from experiments/controlled_ab/viewer/viewer.js (working tree, 2026-11-01).
// ABV1 decoding, exposed-voxel rendering, static masks, and camera transforms are preserved.
// This page adds English presentation and a shared, frame-seeking RGB/voxel/chart clock.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { cameraPose, cropOrigin, enuToViewer } from './camera_pose.js';
const $=id=>document.getElementById(id);
const DATA_VIEWER=new URL('./data/viewer/',import.meta.url);
const getJSON=async path=>{const r=await fetch(new URL(path,DATA_VIEWER),{cache:'no-store'});if(!r.ok)throw Error(`${r.status}: ${path}`);return r.json();};
const PANE_KEYS=['gt','s','xl'];
const SOURCE={
  gt:{label:'Ground Truth',video:'gtVideo',fig:'gtFig',check:'showGt',name:'GT'},
  s:{label:'PERSIST-S',video:'sVideo',fig:'sFig',check:'showS',model:'S',name:'PERSIST-S'},
  xl:{label:'PERSIST-XL',video:'xlVideo',fig:'xlFig',check:'showXl',model:'XL',name:'PERSIST-XL'},
};

class VoxelStream {
  constructor(buffer) {
    const h=new DataView(buffer);
    if(h.getUint32(0,true)!==0x31564241)throw Error('Invalid ABV1 voxel stream');
    this.frames=h.getUint32(4,true);this.shape=[8,12,16].map(o=>h.getUint32(o,true));
    const volume=this.shape.reduce((a,b)=>a*b,1);
    let o=20;this.initial=new Uint16Array(buffer.slice(o,o+volume*2));o+=volume*2;
    this.ptr=new Uint32Array(buffer.slice(o,o+(this.frames+1)*4));o+=(this.frames+1)*4;
    const n=this.ptr[this.frames];
    this.indices=new Uint32Array(buffer.slice(o,o+n*4));o+=n*4;
    this.values=new Uint16Array(buffer.slice(o,o+n*2));
    this.current=this.initial.slice();this.t=0;
  }
  frame(t) {
    if(t<this.t){this.current.set(this.initial);this.t=0;}
    while(this.t<t){this.t++;for(let j=this.ptr[this.t];j<this.ptr[this.t+1];j++)this.current[this.indices[j]]=this.values[j];}
    return this.current;
  }
}

let manifest,classes,gt,predS,predXL,view,record,viewS,recS,viewXL,recXL;
const videoIDs=['gtVideo','sVideo','xlVideo'];
let displayedPoses=[];
let t=0,playing=false,loadCounter=0,renderWidth=0,renderHeight=0;
const world=$('world');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));world.appendChild(renderer.domElement);
renderer.setClearColor(0xeef2eb);
const camera=new THREE.PerspectiveCamera(48,1,.1,400);
camera.position.set(42,32,42);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,-2,0);controls.enableDamping=true;
const geometry=new THREE.BoxGeometry(.98,.98,.98), matrix=new THREE.Matrix4();
const scenes=[new THREE.Scene(),new THREE.Scene(),new THREE.Scene()];
const solidMaterial=new THREE.MeshLambertMaterial({color:0xffffff});
const waterMaterial=new THREE.MeshLambertMaterial({color:0xffffff,transparent:true,opacity:.66,depthWrite:false});
const meshes=[],players=[],cameraRigs=[];
for(const scene of scenes){
  scene.add(new THREE.HemisphereLight(0xe9ffff,0x344136,2.5));
  const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(20,40,10);scene.add(sun);
  const pair=[new THREE.InstancedMesh(geometry,solidMaterial,48**3),new THREE.InstancedMesh(geometry,waterMaterial,48**3)];
  for(const m of pair){m.count=0;m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);m.frustumCulled=false;scene.add(m);}meshes.push(pair);
  const player=new THREE.Mesh(new THREE.SphereGeometry(.22,12,8),new THREE.MeshBasicMaterial({color:0xffda8c}));
  scene.add(player);players.push(player);
  const poseCamera=new THREE.PerspectiveCamera(50,640/360,.15,4),group=new THREE.Group();group.visible=false;
  const frustum=new THREE.CameraHelper(poseCamera),cyan=new THREE.Color(0x56e5ff),green=new THREE.Color(0x91ee9a);
  frustum.setColors(cyan,cyan,green,cyan,cyan);
  const body=new THREE.Mesh(new THREE.SphereGeometry(.18,12,8),new THREE.MeshBasicMaterial({color:cyan}));
  const forward=new THREE.ArrowHelper(new THREE.Vector3(0,0,-1),new THREE.Vector3(),3,cyan,.5,.25);
  const up=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),new THREE.Vector3(),1.4,green,.3,.18);
  group.add(frustum,body,forward,up);
  group.traverse(o=>{o.renderOrder=1000;if(o.material){o.material.depthTest=false;o.material.depthWrite=false;}});
  scene.add(group);cameraRigs.push({group,camera:poseCamera,frustum,body,forward,up});
  const grid=new THREE.GridHelper(48,12,0xb6c5b0,0xd8e1d3);grid.position.y=-24;scene.add(grid);
}

function streamOf(kind){return kind==='gt'?gt:kind==='s'?predS:predXL;}
function viewOf(kind){return kind==='gt'?view:kind==='s'?viewS:viewXL;}
function recOf(kind){return kind==='gt'?record:kind==='s'?recS:recXL;}
function enabledPanes(){
  const panes=PANE_KEYS.filter(kind=>$(SOURCE[kind].check).checked&&streamOf(kind));
  if(panes.length)return panes;
  if(gt){$('showGt').checked=true;syncPaneButton();return ['gt'];}
  return [];
}
function applyAvailability(haveS,haveXl){
  $('showS').disabled=!haveS;if(!haveS)$('showS').checked=false;
  $('showXl').disabled=!haveXl;if(!haveXl)$('showXl').checked=false;
  syncPaneButton();
}
function syncPaneButton(){
  const names=PANE_KEYS.filter(kind=>$(SOURCE[kind].check).checked).map(kind=>SOURCE[kind].model||'GT');
  $('paneSummary').textContent=names.length?names.join(' · '):'Voxel panes';
}
function syncVideos(){
  const panes=enabledPanes();
  for(const kind of PANE_KEYS)$(SOURCE[kind].fig).hidden=!panes.includes(kind);
  $('experiment').style.setProperty('--pane-count',panes.length||1);
}
function clearScene(i){
  for(const m of meshes[i])m.count=0;
  cameraRigs[i].group.visible=false;players[i].visible=false;
}

function poseFor(kind){
  const video=$(SOURCE[kind].video),aspect=video.videoWidth&&video.videoHeight?video.videoWidth/video.videoHeight:640/360;
  const stream=streamOf(kind),v=viewOf(kind),rec=recOf(kind);
  if(kind==='gt')return {pose:cameraPose(view,t,gt.shape,aspect,{groundTruth:true}),label:'GT recorded camera',worldLabel:'World ENU'};
  const name=SOURCE[kind].model;
  if(v){
    const gtConditioned=rec?.use_camera_gt===true;
    const referenceOrigin=gtConditioned?cropOrigin(view,t,gt.shape):null;
    return {pose:cameraPose(v,t,stream.shape,aspect,{referenceOrigin}),
      label:`PERSIST-${name} exported camera`+(gtConditioned?' (GT-conditioned)':''),
      worldLabel:gtConditioned?'GT crop reference ENU':'Unregistered world ENU'};
  }
  return {pose:null,label:`PERSIST-${name} camera export unavailable`};
}

function renderCamera(kind,sceneIndex){
  const row=poseFor(kind),rig=cameraRigs[sceneIndex],p=row.pose;
  displayedPoses.push(row);rig.group.visible=$('showCamera').checked&&!!p;
  players[sceneIndex].visible=!!view.player_position?.[t]&&(kind==='gt'||recOf(kind)?.use_camera_gt===true);
  if(players[sceneIndex].visible){
    const origin=cropOrigin(view,t,gt.shape),local=view.player_position[t].map((x,i)=>x-origin[i]-gt.shape[i]/2);
    players[sceneIndex].position.fromArray(enuToViewer(local));
  }
  if(!p)return;
  rig.camera.position.fromArray(p.position);rig.camera.up.fromArray(p.viewerUp);
  rig.camera.lookAt(new THREE.Vector3().fromArray(p.position).add(new THREE.Vector3().fromArray(p.viewerForward)));
  rig.camera.fov=THREE.MathUtils.radToDeg(p.fovY);rig.camera.aspect=p.aspect;
  rig.camera.updateProjectionMatrix();rig.camera.updateMatrixWorld(true);rig.frustum.update();
  for(const o of [rig.body,rig.forward,rig.up])o.position.copy(rig.camera.position);
  rig.forward.setDirection(new THREE.Vector3().fromArray(p.viewerForward));rig.up.setDirection(new THREE.Vector3().fromArray(p.viewerUp));
}

function renderVoxels(stream,sceneIndex,highlightGT=null){
  const data=stream.frame(t),shape=stream.shape,[sx,sy,sz]=shape;
  const reference=highlightGT?.data,staticMask=highlightGT?.mask,cut=Number($('cut').value);
  const count=[0,0],colors=classes.map(c=>new THREE.Color(c.color));
  const index=(x,y,z)=>(x*sy+y)*sz+z;
  for(let x=0;x<sx;x++)for(let y=0;y<sy;y++)for(let z=0;z<Math.min(sz,cut);z++){
    const i=index(x,y,z),c=classes[data[i]];
    if(!c)throw Error(`Unknown class ${data[i]}`);
    const differs=reference&&staticMask[i]&&data[i]!==reference[i];
    if((c.air||c.ignore)&&!differs)continue;
    const neighbours=[[x-1,y,z],[x+1,y,z],[x,y-1,z],[x,y+1,z],[x,y,z-1],[x,y,z+1]];
    const exposed=neighbours.some(([a,b,d])=>a<0||a>=sx||b<0||b>=sy||d<0||d>=Math.min(sz,cut)||
      classes[data[index(a,b,d)]].air||classes[data[index(a,b,d)]].ignore||classes[data[index(a,b,d)]].water!==c.water);
    if(!exposed&&!differs)continue;
    const kind=c.water?1:0,m=meshes[sceneIndex][kind],n=count[kind]++;
    matrix.makeTranslation(x+.5-sx/2,z+.5-sz/2,-(y+.5-sy/2));m.setMatrixAt(n,matrix);
    // Magenta includes missing solids when the optional diagnostic overlay is enabled.
    m.setColorAt(n,differs?new THREE.Color('#f767bc'):colors[data[i]]);
  }
  for(let k=0;k<2;k++){const m=meshes[sceneIndex][k];m.count=count[k];m.instanceMatrix.needsUpdate=true;if(m.instanceColor)m.instanceColor.needsUpdate=true;}
}

function updateStats(){
  $('frame').textContent=`Frame ${t} / ${gt.frames-1}`;$('time').value=t;
  document.dispatchEvent(new CustomEvent('briefing:frame',{detail:{frame:t,frames:gt.frames,fps:record.fps||24,episode:$('episode').value}}));
}

function drawFrame(){
  if(!gt)return;
  const panes=enabledPanes();
  let reference=null;
  if($('highlight').checked&&panes.some(kind=>kind!=='gt')){
    const target=gt.frame(t).slice(),old=gt.frame(Math.max(t-1,0)),[sx,sy,sz]=gt.shape;
    const previousOrigin=cropOrigin(view,Math.max(t-1,0),gt.shape);
    const mask=new Uint8Array(target.length),d=cropOrigin(view,t,gt.shape).map((x,k)=>Math.round(x-previousOrigin[k]));
    if(t)for(let x=0;x<sx;x++)for(let y=0;y<sy;y++)for(let z=0;z<sz;z++){
      const a=x+d[0],b=y+d[1],c=z+d[2],i=(x*sy+y)*sz+z;
      if(a>=0&&a<sx&&b>=0&&b<sy&&c>=0&&c<sz&&classes[target[i]].solid&&old[(a*sy+b)*sz+c]===target[i])mask[i]=1;
    }
    gt.frame(t);reference={data:target,mask};
  }
  displayedPoses=[];
  panes.forEach((kind,i)=>{renderVoxels(streamOf(kind),i,kind==='gt'?null:reference);renderCamera(kind,i);});
  for(let i=panes.length;i<scenes.length;i++)clearScene(i);
  $('labels').innerHTML=panes.map(kind=>`<span class="pane-label"><b class="series-label ${kind}">${SOURCE[kind].label}</b></span>`).join('');
  syncVideos();
  updateStats();
}

let ready=false,presenting=false,pending=null,clockStart=0,clockFrame=0,visible=true;
const emit=(name,detail)=>document.dispatchEvent(new CustomEvent(`briefing:${name}`,{detail}));
const videos=()=>videoIDs.map($).filter(v=>v.hasAttribute('src'));
function pause(){
  playing=false;$('play').textContent='Play';
  for(const v of videos())v.pause();
}
function setControls(enabled){
  for(const id of ['play','restart','previous','next','time'])$(id).disabled=!enabled;
  $('experiment').setAttribute('aria-busy',String(!enabled));
}
function mediaReady(video,event){
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{video.removeEventListener(event,done);video.removeEventListener('error',fail);video.removeEventListener('emptied',done);};
    const done=()=>{cleanup();resolve();};
    const fail=()=>{cleanup();reject(Error('Unable to read recorded RGB: '+video.getAttribute('src')));};
    video.addEventListener(event,done,{once:true});video.addEventListener('error',fail,{once:true});
    if(event==='seeked')video.addEventListener('emptied',done,{once:true});
  });
}
async function alignVideos(frame){
  const seconds=frame/(record.fps||24);
  await Promise.all(videos().map(async v=>{
    // A single frame clock seeks all three paused videos, preventing native-player drift.
    if(Math.abs(v.currentTime-seconds)<1e-6&&!v.seeking&&v.readyState>=2)return;
    const settled=mediaReady(v,'seeked');v.currentTime=seconds;await settled;
  }));
}
async function presentFrame(value){
  if(!ready)return;
  pending={frame:Math.max(0,Math.min(gt.frames-1,Math.round(value))),token:loadCounter};
  if(presenting)return;
  presenting=true;
  try{
    while(pending){
      const request=pending;pending=null;
      await alignVideos(request.frame);
      if(request.token!==loadCounter||!ready)continue;
      if(pending)continue;
      t=request.frame;drawFrame();
      if(t===gt.frames-1)pause();
    }
  }catch(e){error(e);}finally{presenting=false;}
}
function seek(value){pause();presentFrame(value);}
async function loadEpisode(){
  pause();ready=false;pending=null;setControls(false);
  const token=++loadCounter,episode=$('episode').value;
  $('status').className='';$('status').textContent='Loading recorded experiment…';
  $('voxelLoading').hidden=false;
  for(let i=0;i<scenes.length;i++)clearScene(i);
  const item=manifest.episodes[episode];
  if(!item)throw Error('This episode is unavailable.');
  emit('loading',{episode});
  const fetchStream=async folder=>{const r=await fetch(folder+'/voxels.bin');if(!r.ok)throw Error('Voxel stream unavailable: '+folder);return new VoxelStream(await r.arrayBuffer());};
  const optionalView=folder=>folder?fetch(folder+'/view.json').then(r=>r.ok?r.json():null):null;
  const values=await Promise.all([
    fetchStream(item.gt),item.pred?fetchStream(item.pred):null,item.pred_xl?fetchStream(item.pred_xl):null,
    getJSON(item.gt+'/view.json'),getJSON(item.gt+'/record.json'),
    optionalView(item.pred),item.pred?getJSON(item.pred+'/record.json'):null,
    optionalView(item.pred_xl),item.pred_xl?getJSON(item.pred_xl+'/record.json'):null,
  ]);
  if(token!==loadCounter)return;
  [gt,predS,predXL,view,record,viewS,recS,viewXL,recXL]=values;t=0;
  applyAvailability(predS,predXL);$('time').max=gt.frames-1;
  await Promise.all([['gt','gtVideo'],['pred','sVideo'],['pred_xl','xlVideo']].map(async([key,id])=>{
    const v=$(id);v.pause();
    if(!item[key]){v.removeAttribute('src');v.load();return;}
    v.src=item[key]+'/rgb.mp4';
    if(v.readyState<2)await mediaReady(v,'loadeddata');
  }));
  if(token!==loadCounter)return;
  ready=true;setControls(true);$('voxelLoading').hidden=true;
  $('status').textContent=`${gt.frames} frames · ${record.fps||24} fps`;
  homeCamera();
  emit('ready',{episode,frames:gt.frames,fps:record.fps||24});
  await presentFrame(0);
}
function homeCamera(){camera.position.set(48,34,48);controls.target.set(0,-12,0);controls.update();}
function error(e){pause();ready=false;pending=null;setControls(false);$('status').className='error';$('status').textContent=e.message;$('voxelLoading').hidden=true;console.error(e);}
$('previous').onclick=()=>seek(t-1);$('next').onclick=()=>seek(t+1);
$('restart').onclick=()=>seek(0);
$('time').oninput=e=>seek(Number(e.target.value));
$('play').onclick=()=>{
  if(!ready)return;
  if(playing){pause();return;}
  clockFrame=t>=gt.frames-1?0:t;clockStart=performance.now();playing=true;$('play').textContent='Pause';
  if(clockFrame!==t)presentFrame(clockFrame);
};
for(const kind of PANE_KEYS)$(SOURCE[kind].check).onchange=()=>{syncPaneButton();if(ready)drawFrame();};
for(const id of ['cut','highlight','showCamera'])$(id).addEventListener(id==='cut'?'input':'change',()=>{if(ready)drawFrame();});
document.addEventListener('click',e=>{for(const id of ['paneMenu','viewOptions']){const menu=$(id);if(menu.open&&!menu.contains(e.target))menu.open=false;}});
$('locateCamera').onclick=()=>{const p=displayedPoses.find(row=>row.pose)?.pose;if(!p)return;controls.target.fromArray(p.position);camera.position.copy(controls.target).add(new THREE.Vector3(8,6,8));controls.update();};
$('resetView').onclick=homeCamera;
$('episode').onchange=()=>loadEpisode().catch(error);
$('speed').onchange=()=>{clockFrame=t;clockStart=performance.now();};
const experiment=$('experiment');
function expandViewer(expanded){
  experiment.classList.toggle('is-expanded',expanded);
  document.body.classList.toggle('viewer-expanded',expanded);
  $('expandViewer').setAttribute('aria-pressed',String(expanded));
  $('expandViewer').textContent=expanded?'Exit focus':'Focus view';
  experiment.focus({preventScroll:true});
}
$('expandViewer').onclick=()=>expandViewer(!experiment.classList.contains('is-expanded'));
experiment.addEventListener('pointerdown',e=>{
  if(e.target===renderer.domElement)experiment.focus({preventScroll:true});
});
experiment.addEventListener('keydown',e=>{
  if(e.key==='Escape'){expandViewer(false);return;}
  if(!ready||e.altKey||e.ctrlKey||e.metaKey||e.target.matches('input,select,textarea,button,summary')||e.target.isContentEditable)return;
  const actions={' ':()=>$('play').click(),ArrowLeft:()=>seek(t-1),ArrowRight:()=>seek(t+1),Home:()=>seek(0),End:()=>seek(gt.frames-1)};
  if(actions[e.key]){e.preventDefault();actions[e.key]();}
});
document.addEventListener('briefing:seek',e=>seek(e.detail));
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;},{rootMargin:'200px'}).observe(world);
function animate(now){
  requestAnimationFrame(animate);
  if(playing&&ready&&!presenting){
    const next=Math.min(gt.frames-1,clockFrame+Math.floor((now-clockStart)/1000*(record.fps||24)*Number($('speed').value)));
    if(next!==t)presentFrame(next);
  }
  if(!visible)return;
  controls.update();
  const w=world.clientWidth,h=world.clientHeight;
  if(w!==renderWidth||h!==renderHeight){renderer.setSize(w,h);renderWidth=w;renderHeight=h;}
  const n=Math.max(1,enabledPanes().length);camera.aspect=w/n/h;camera.updateProjectionMatrix();
  renderer.setScissorTest(true);
  for(let i=0;i<n;i++){const width=w/n;renderer.setViewport(i*width,0,width,h);renderer.setScissor(i*width,0,width,h);renderer.render(scenes[i],camera);}
}
requestAnimationFrame(animate);
try{
  [manifest,classes]=await Promise.all([getJSON('./manifest.json'),getJSON('../classes.json')]);
  // Manifest paths belong to the original results Viewer, not this document.
  for(const item of Object.values(manifest.episodes))for(const key of ['gt','pred','pred_xl'])if(item[key])item[key]=new URL(item[key],DATA_VIEWER).href;
  for(const option of $('episode').options)option.disabled=!manifest.episodes[option.value];
  await loadEpisode();
}catch(e){error(e);}
