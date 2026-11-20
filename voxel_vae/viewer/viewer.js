import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

const $=id=>document.getElementById(id),cube=new THREE.BoxGeometry(.94,.94,.94),matrix=new THREE.Matrix4();
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('canvas').appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#14202a');
const camera=new THREE.PerspectiveCamera(46,1,.1,200);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,1.7));const sun=new THREE.DirectionalLight(0xffffff,2.2);sun.position.set(18,30,24);scene.add(sun);
const content=new THREE.Group(),bounds=new THREE.Group(),errors=new THREE.Group();scene.add(content,bounds,errors);
let manifest,current=null,playing=null;

function resetCamera(){camera.position.set(33,25,42);controls.target.set(0,0,0);controls.update();}
function clear(group,sharedGeometry=false){while(group.children.length){const child=group.children[group.children.length-1];group.remove(child);if(!sharedGeometry)child.geometry?.dispose();child.material?.dispose();}}
function xyz(index){const x=Math.floor(index/256),r=index-x*256,y=Math.floor(r/16);return[x,y,r-y*16];}
function position(index,offset){const[x,y,z]=xyz(index);return[offset+x-7.5,z-7.5,-(y-7.5)];}
function visible(index){const axis=$('axis').value;if(axis==='all')return true;return xyz(index)[Number(axis)]<=Number($('slice').value);}
function colorFor(token){const name=manifest.vocabulary[token].node_name.toLowerCase();
 if(name==='air')return null;if(name.includes('water'))return new THREE.Color('#238bd1');if(name.includes('lava'))return new THREE.Color('#f06a1b');
 if(name.includes('fire'))return new THREE.Color('#ffb12b');if(/leaf|leaves|vine|bamboo|grass|flower|sapling/.test(name))return new THREE.Color('#58a94b');
 if(/dirt|soil|mud/.test(name))return new THREE.Color('#8b6842');if(/sand|gravel/.test(name))return new THREE.Color('#c6ad72');
 if(/wood|tree|log|plank/.test(name))return new THREE.Color('#8a5932');if(/ice|snow/.test(name))return new THREE.Color('#bde4eb');
 if(/stone|cobble|ore/.test(name))return new THREE.Color('#82878a');let h=2166136261;for(const c of name)h=Math.imul(h^c.charCodeAt(0),16777619);return new THREE.Color().setHSL(((h>>>0)%360)/360,.45,.55);}
function addVolume(values,offset){const entries=[];for(let i=0;i<values.length;i++){const color=colorFor(values[i]);if(color&&visible(i))entries.push([i,color]);}
 const mesh=new THREE.InstancedMesh(cube,new THREE.MeshLambertMaterial({vertexColors:true}),entries.length);
 entries.forEach(([index,color],i)=>{matrix.makeTranslation(...position(index,offset));mesh.setMatrixAt(i,matrix);mesh.setColorAt(i,color);});
 mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;content.add(mesh);
}
function addBounds(offset){const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(16,16,16)),new THREE.LineBasicMaterial({color:0x8fa5b2}));box.position.x=offset;bounds.add(box);}
function addErrors(gt,pred){if(!$('errors').checked)return;for(const offset of[-10,10]){const ids=[];for(let i=0;i<gt.length;i++)if(gt[i]!==pred[i]&&visible(i))ids.push(i);const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(1.02,1.02,1.02),new THREE.MeshBasicMaterial({color:0xff3f37,wireframe:true,depthTest:false}),ids.length);ids.forEach((id,i)=>{matrix.makeTranslation(...position(id,offset));mesh.setMatrixAt(i,matrix);});mesh.instanceMatrix.needsUpdate=true;errors.add(mesh);}}
function legend(values){const ids=[...new Set(values)].filter(id=>colorFor(id)).sort((a,b)=>a-b);$('legend').innerHTML=ids.slice(0,30).map(id=>{const token=manifest.vocabulary[id],color='#'+colorFor(id).getHexString();return`<div class="legend-row"><span class="swatch" style="background:${color}"></span>${token.node_name} · p2=${token.param2}</div>`}).join('')+(ids.length>30?`<div>… ${ids.length-30} more token states</div>`:'');}
function render(){if(!current)return;clear(content,true);clear(bounds);clear(errors);addVolume(current.gt,-10);addVolume(current.predicted,10);addBounds(-10);addBounds(10);addErrors(current.gt,current.predicted);legend([...current.gt,...current.predicted]);const r=current.record,m=current.metrics;$('summary').textContent=`exact=${(100*m.exact_accuracy).toFixed(5)}% · non-air=${m.non_air_accuracy===null?'n/a':(100*m.non_air_accuracy).toFixed(5)+'%'} · errors=${m.errors}/4096`;$('metadata').textContent=JSON.stringify(r,null,2);}
async function loadSample(index){$('error').textContent='';try{const response=await fetch(`/api/sample?index=${index}`);if(!response.ok)throw Error(await response.text());current=await response.json();render();}catch(error){$('error').textContent=error.message;}}
function group(){return manifest.groups[$('category').value][Number($('example').value)];}
function updateState(){const g=group(),state=Math.min(Number($('state').value),g.indices.length-1);$('state').max=Math.max(0,g.indices.length-1);$('state').value=state;$('state').disabled=g.indices.length===1;$('stateLabel').textContent=`${state+1} / ${g.indices.length}`;loadSample(g.indices[state]);}
function updateExamples(){const groups=manifest.groups[$('category').value];$('example').innerHTML=groups.map((g,i)=>`<option value="${i}">${g.label}</option>`).join('');$('state').value=0;updateState();}
function step(delta){const max=Number($('state').max);$('state').value=(Number($('state').value)+delta+max+1)%(max+1);updateState();}
function stop(){if(playing){clearInterval(playing);playing=null;$('play').textContent='Play';}}
async function init(){manifest=await(await fetch('/api/manifest')).json();const categories=Object.keys(manifest.groups);$('category').innerHTML=categories.map(c=>`<option>${c}</option>`).join('');resetCamera();updateExamples();}
$('category').onchange=()=>{stop();updateExamples();};$('example').onchange=()=>{stop();$('state').value=0;updateState();};$('state').oninput=()=>{stop();updateState();};
$('back').onclick=()=>{stop();step(-1);};$('next').onclick=()=>{stop();step(1);};$('play').onclick=()=>{if(playing){stop();return;}$('play').textContent='Pause';playing=setInterval(()=>step(1),900);};
$('axis').onchange=render;$('slice').oninput=()=>{$('sliceLabel').textContent=$('slice').value;render();};$('errors').onchange=render;$('reset').onclick=resetCamera;
function resize(){const box=$('canvas').getBoundingClientRect();renderer.setSize(box.width,box.height,false);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}
function animate(){requestAnimationFrame(animate);resize();controls.update();renderer.render(scene,camera);}window.addEventListener('resize',resize);init().catch(error=>$('error').textContent=error.message);animate();
