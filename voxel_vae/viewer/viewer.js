import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

const $=id=>document.getElementById(id),cube=new THREE.BoxGeometry(.94,.94,.94),matrix=new THREE.Matrix4();
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;$('canvas').appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#14202a');
const camera=new THREE.PerspectiveCamera(46,1,.1,200);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,1.7));const sun=new THREE.DirectionalLight(0xffffff,2.2);sun.position.set(18,30,24);scene.add(sun);
const content=new THREE.Group(),bounds=new THREE.Group(),errors=new THREE.Group();scene.add(content,bounds,errors);
const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let manifest,current=null,playing=null;

function resetCamera(){camera.position.set(33,25,42);controls.target.set(0,0,0);controls.update();}
function clear(group,sharedGeometry=false){while(group.children.length){const child=group.children[group.children.length-1];group.remove(child);if(!sharedGeometry)child.geometry?.dispose();child.material?.dispose();}}
function xyz(index){const x=Math.floor(index/256),r=index-x*256,y=Math.floor(r/16);return[x,y,r-y*16];}
function position(index,offset){const[x,y,z]=xyz(index);return[offset+x-7.5,z-7.5,-(y-7.5)];}
function visible(index){const axis=$('axis').value;if(axis==='all')return true;return xyz(index)[Number(axis)]<=Number($('slice').value);}
function colorFor(token){const row=manifest.vocabulary[token],name=row.node_name.toLowerCase();let color;
 if(name==='air')return null;if(name.includes('water'))color='#238bd1';else if(name.includes('lava'))color='#f06a1b';else if(name.includes('fire'))color='#ffb12b';
 else if(name.includes('granite'))color='#a36f5d';else if(name.includes('diorite'))color='#d0cdc2';else if(name.includes('andesite'))color='#6f7678';else if(name.includes('deepslate'))color='#3f464d';else if(name.includes('bedrock'))color='#30363a';
 else if(name.includes('stone_with_coal'))color='#4d5357';else if(name.includes('stone_with_iron'))color='#aa876d';else if(name.includes('stone_with_copper'))color='#b66f48';else if(name.includes('stone_with_lapis'))color='#3f62a8';else if(name.includes('stone_with_redstone'))color='#a23e3e';else if(name.includes('stone_with_gold'))color='#c5a33b';else if(name.includes('stone_with_emerald'))color='#3c9c68';else if(name.includes('stone_with_diamond'))color='#55b9bf';
 else if(name.includes('mossycobble'))color='#66755c';else if(name.includes('cobble'))color='#707578';else if(name.includes('gravel'))color='#91877e';else if(name.includes('stone'))color='#8b9094';
 else if(/leaf|leaves|vine|bamboo|grass|flower|sapling/.test(name))color='#58a94b';else if(/dirt|soil|mud/.test(name))color='#8b6842';else if(name.includes('sand'))color='#c6ad72';
 else if(/wood|tree|log|plank/.test(name))color='#8a5932';else if(/ice|snow/.test(name))color='#bde4eb';else{let h=2166136261;for(const c of name)h=Math.imul(h^c.charCodeAt(0),16777619);return new THREE.Color().setHSL(((h>>>0)%360)/360,.45,.55);}
 const result=new THREE.Color(color);if(row.param2)result.offsetHSL(0,0,((row.param2%7)-3)*.018);return result;}
function addVolume(values,offset){const groups=new Map();for(let i=0;i<values.length;i++){const token=values[i],color=colorFor(token);if(!color||!visible(i))continue;if(!groups.has(token))groups.set(token,{color,indices:[]});groups.get(token).indices.push(i);}
 for(const [token,{color,indices}] of groups){const material=new THREE.MeshBasicMaterial({color,vertexColors:false,toneMapped:false});const mesh=new THREE.InstancedMesh(cube,material,indices.length);indices.forEach((index,i)=>{matrix.makeTranslation(...position(index,offset));mesh.setMatrixAt(i,matrix);});mesh.instanceMatrix.needsUpdate=true;mesh.userData={indices,token,side:offset<0?'GT':'Prediction'};content.add(mesh);}
}
function addBounds(offset){const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(16,16,16)),new THREE.LineBasicMaterial({color:0x8fa5b2}));box.position.x=offset;bounds.add(box);}
function addWire(ids,offset,color){const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(1.02,1.02,1.02),new THREE.MeshBasicMaterial({color,wireframe:true,depthTest:false,toneMapped:false}),ids.length);ids.forEach((id,i)=>{matrix.makeTranslation(...position(id,offset));mesh.setMatrixAt(i,matrix);});mesh.instanceMatrix.needsUpdate=true;errors.add(mesh);}
function addErrors(gt,pred){const mode=$('errorMode').value;if(mode==='off')return;if(mode==='comparison'&&current.predictions.Plain&&current.predictions.Masked){const plain=current.predictions.Plain,masked=current.predictions.Masked,improved=[],regressed=[],bothWrong=[],unchangedError=[];for(let i=0;i<gt.length;i++){if(!visible(i))continue;const p=plain[i]===gt[i],m=masked[i]===gt[i];if(current.changed_mask[i]){if(!p&&m)improved.push(i);else if(p&&!m)regressed.push(i);else if(!p&&!m)bothWrong.push(i);}else if(pred[i]!==gt[i])unchangedError.push(i);}addWire(improved,10,0x36df76);addWire(regressed,10,0xd95cff);addWire(bothWrong,10,0xff3f37);addWire(unchangedError,10,0xffa52f);return;}const targets=mode==='both'?[[-10,0xffd84a],[10,0xff3f37]]:[[10,0xff3f37]];for(const[offset,color]of targets){const ids=[];for(let i=0;i<gt.length;i++)if(gt[i]!==pred[i]&&visible(i))ids.push(i);addWire(ids,offset,color);}}
function legend(values){const ids=[...new Set(values)].filter(id=>colorFor(id)).sort((a,b)=>a-b);$('legend').innerHTML=ids.slice(0,30).map(id=>{const token=manifest.vocabulary[id],color='#'+colorFor(id).getHexString();return`<div class="legend-row"><span class="swatch" style="background:${color}"></span>${token.node_name} · p2=${token.param2}</div>`}).join('')+(ids.length>30?`<div>… ${ids.length-30} more token states</div>`:'');}
function selectedPrediction(){return current.predictions[$('variant').value]||current.predicted;}
function render() {
  if (!current) return;
  const predicted = selectedPrediction();
  const correct = predicted.reduce((count, token, i) => count + (token === current.gt[i]), 0);
  clear(content, true);
  clear(bounds);
  clear(errors);
  addVolume(current.gt, -10);
  addVolume(predicted, 10);
  addBounds(-10);
  addBounds(10);
  addErrors(current.gt, predicted);
  legend([...current.gt, ...predicted]);
  const variant = $('variant').value;
  const changed = current.changed_mask.filter(Boolean).length;
  const summary = `${variant} exact=${(100 * correct / 4096).toFixed(5)}% · errors=${4096 - correct}/4096 · changed-mask voxels=${changed}`;
  $('summary').textContent = $('errorMode').value === 'comparison'
    ? `${summary}\nGreen: error fixed in Masked. Magenta: regression. Red: shared error. Orange: unchanged-region error.`
    : summary;
  $('metadata').textContent = JSON.stringify(current.record, null, 2);
  document.querySelector('.side.pred').textContent = `${variant} reconstruction`;
}
function tokenText(id){const row=manifest.vocabulary[id];return`${id}: ${row.node_name}, param2=${row.param2}`;}
function inspect(event){if(!current)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);const hit=raycaster.intersectObjects(content.children,false)[0];if(!hit||hit.instanceId===undefined)return;const id=hit.object.userData.indices[hit.instanceId],[x,y,z]=xyz(id),gt=current.gt[id],lines=Object.entries(current.predictions).map(([name,value])=>`${name}: ${tokenText(value[id])} ${value[id]===gt?'✓':'✗'}`);$('inspector').textContent=`Clicked: ${hit.object.userData.side}\nLocal XYZ: ${x}, ${y}, ${z}\nChanged mask: ${current.changed_mask[id]?'YES':'no'}\nGT: ${tokenText(gt)}\n${lines.join('\n')}`;}
async function loadSample(index){$('error').textContent='';try{const response=await fetch(`/api/sample?index=${index}`);if(!response.ok)throw Error(await response.text());current=await response.json();render();}catch(error){$('error').textContent=error.message;}}
function group(){return manifest.groups[$('category').value][Number($('example').value)];}
function updateState(){const g=group(),state=Math.min(Number($('state').value),g.indices.length-1);$('state').max=Math.max(0,g.indices.length-1);$('state').value=state;$('state').disabled=g.indices.length===1;$('stateLabel').textContent=`${state+1} / ${g.indices.length}`;loadSample(g.indices[state]);}
function updateExamples(){const groups=manifest.groups[$('category').value];$('example').innerHTML=groups.map((g,i)=>`<option value="${i}">${g.label}</option>`).join('');$('state').value=0;updateState();}
function step(delta){const max=Number($('state').max);$('state').value=(Number($('state').value)+delta+max+1)%(max+1);updateState();}
function stop(){if(playing){clearInterval(playing);playing=null;$('play').textContent='Play';}}
async function init() {
  manifest = await (await fetch('/api/manifest')).json();
  const categories = Object.keys(manifest.groups);
  $('category').innerHTML = categories.map(category => `<option>${category}</option>`).join('');
  $('variant').innerHTML = manifest.variants.map(variant => `<option>${variant}</option>`).join('');
  const hasMasked = manifest.variants.includes('Masked');
  $('variant').value = hasMasked ? 'Masked' : manifest.variants.at(-1);
  $('pairedOverlay').hidden = !(hasMasked && manifest.variants.includes('Plain'));
  resetCamera();
  updateExamples();
}
$('category').onchange=()=>{stop();updateExamples();};$('example').onchange=()=>{stop();$('state').value=0;updateState();};$('state').oninput=()=>{stop();updateState();};
$('back').onclick=()=>{stop();step(-1);};$('next').onclick=()=>{stop();step(1);};$('play').onclick=()=>{if(playing){stop();return;}$('play').textContent='Pause';playing=setInterval(()=>step(1),900);};
$('axis').onchange=render;$('slice').oninput=()=>{$('sliceLabel').textContent=$('slice').value;render();};$('variant').onchange=render;$('errorMode').onchange=render;$('reset').onclick=resetCamera;renderer.domElement.addEventListener('click',inspect);
function resize(){const box=$('canvas').getBoundingClientRect();renderer.setSize(box.width,box.height,false);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}
function animate(){requestAnimationFrame(animate);resize();controls.update();renderer.render(scene,camera);}window.addEventListener('resize',resize);init().catch(error=>$('error').textContent=error.message);animate();
