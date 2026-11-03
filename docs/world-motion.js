// Adapted with the owner's permission from https://mengmaor.github.io/VoxEvolve/.
// Original source: docs/research_site/dist/index.html (verified against the live site, 2026-11-03).
// This is the original procedural concept illustration, not an experimental model rollout.
const $=id=>document.getElementById(id);
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let edited=false;
const N=19,TW=28,TH=14,Z=15;
const project=(x,y,z)=>[320+(x-y)*TW/2,105+(x+y)*TH/2-z*Z];
const pstr=ps=>ps.map(p=>p.map(n=>n.toFixed(1)).join(',')).join(' ');
const clamp=n=>Math.max(0,Math.min(1,n));
const palette={placed:['#db986c','#bd774f','#9d603e'],grass:['#9bb27b','#758e59','#607a4d'],moss:['#859f67','#627d4e','#4b663e'],stone:['#b6ba9c','#91967b','#787f66'],sand:['#d3be91','#af996e','#95815d'],earth:['#9d8b63','#796847','#66563d'],water:['#92bbc3','#76a5b0','#5d8b98'],deep:['#7ba8b0','#628e96','#527d84'],wood:['#947451','#73573d','#5d4933'],leaf:['#54764b','#3e5b37','#314a2c'],lightleaf:['#77935d','#587343','#476139']};
function cube(x,y,z,h,type,w=1,d=1){const c=palette[type],a=project(x,y,z),b=project(x+w,y,z),f=project(x,y+d,z),e=project(x+w,y+d,z),bl=project(x+w,y,z-h),fl=project(x,y+d,z-h),el=project(x+w,y+d,z-h);return `<polygon points="${pstr([f,e,el,fl])}" fill="${c[1]}"/><polygon points="${pstr([b,e,el,bl])}" fill="${c[2]}"/><polygon points="${pstr([a,b,e,f])}" fill="${c[0]}" stroke="${c[0]}" stroke-width=".35"/>`;}
const hash=(x,y)=>{let n=Math.sin(x*127.1+y*311.7)*43758.5453;return n-Math.floor(n);};
const land=(x,y)=>Math.hypot((x-9)/1.08,(y-9)/.9)<8.8+Math.sin(x*1.3+y)*.7;
const river=(x,y)=>Math.abs(x-(8.1+Math.sin(y*.38)*2.4))<1.25;
function height(x,y){if(river(x,y))return .85;const peak=Math.max(0,4.1-Math.hypot(x-14,y-5)*.66);return 1.65+Math.floor(peak+hash(x,y)*.8)*.65;}
const treeSites=[[4,5],[14,12],[3,8],[4,11],[6,14],[8,15],[12,14],[15,9],[14,5],[11,4],[6,3],[2,10],[5,7],[13,9],[5,13]].filter(([x,y])=>land(x,y)&&!river(x,y));
const trees=new Map(treeSites.map(([x,y],i)=>[`${x},${y}`,{tall:i<2,rate:.7+hash(x+3,y)*.3}]));
const blocks=[];
for(let x=0;x<N;x++)for(let y=0;y<N;y++)if(land(x,y)){const h=height(x,y),bed=river(x,y);let type=bed?'sand':h>3.1?'stone':hash(x,y)>.65?'moss':'grass';if(!bed&&Math.abs(x-(8.1+Math.sin(y*.38)*2.4))<2)type='sand';blocks.push({x,y,h,bed,type,depth:x+y});}
blocks.sort((a,b)=>a.depth-b.depth||a.x-b.x);
const riverRows=blocks.filter(b=>b.bed).map(b=>b.y),riverStart=Math.min(...riverRows),riverEnd=Math.max(...riverRows)+1;
function tree(x,y,h,scale,tall){const trunk=1.13*scale,wide=1.24*scale;return `<g data-tree="${tall?'tall':'small'}">${cube(x+.36,y+.36,h+trunk,trunk,'wood',.3*scale,.3*scale)}${cube(x+.5-wide/2,y+.5-wide/2,h+trunk+.65*scale,.8*scale,'leaf',wide,wide)}${cube(x+.5-wide*.34,y+.5-wide*.34,h+trunk+1.24*scale,.65*scale,'lightleaf',wide*.68,wide*.68)}</g>`;}
function waterfall(u){let body=cube(9.3,.4,5.65,.7,'stone',2.3,1.8)+cube(9.43,.53,5.74,.1,'water',2.04,1.54)+cube(10.1,2,5.74,.1,'water',.65,.6);body+=cube(10.1,2.12,5.66,4.35+Math.min(u*6,.27),'water',.63,.42);for(let k=0;k<6;k++){const z=1.65+((k*.7-u*5+12)%3.8),p=project(10.4,2.55,z);body+=`<path d="M${p[0]} ${p[1]}v5" stroke="#d0e8e9" opacity=".7" stroke-width="1.15"/>`;}if(u>.03){const p=project(10.4,2.7,1.16);body+=`<path d="m${p[0]-7} ${p[1]-1} 4-4m4 6 5-2m-7-3 1-5" stroke="#d9eded" stroke-width="1.1" fill="none"/>`;}return `<g data-waterfall="true">${body}</g>`;}
function scene(u=0,hero=false){const uid=hero?'hero':'scene';let body='',sourceDrawn=false;
for(const b of blocks){const{x,y,h,bed,type}=b;if(!sourceDrawn&&b.depth>13){body+=waterfall(u);sourceDrawn=true;}body+=cube(x,y,0,.75,'earth')+cube(x,y,h,h,type);
if(edited&&x===3&&y===12)body+=cube(x+.08,y+.08,h+.85,.85,'placed',.84,.84);
if(bed){const amount=u===0?0:clamp(riverStart+(riverEnd-riverStart)*u-y);if(amount>0){body+=`<g data-river-water="true">${cube(x+.025,y+.015,1.12,.12,'water',.95,amount*.97)}</g>`;if(hash(x+3,y+8)>.6&&amount>.5){const p=project(x+.3,y+.3,1.14);body+=`<path d="m${p[0]} ${p[1]} 4 2 5-2" fill="none" stroke="#d1e8e7" stroke-width=".7"/>`;}}else if(hash(x,y)>.7){const p=project(x+.45,y+.4,h+.02);body+=`<path d="m${p[0]-2} ${p[1]} 4 1" stroke="#998154" stroke-width=".7"/>`;}
}else{const t=trees.get(`${x},${y}`);if(t){const scale=t.tall?1.06:.22+u*.64*t.rate;body+=tree(x,y,h,scale,t.tall);}else if(type==='grass'&&hash(x+9,y+15)>.75){const p=project(x+.5,y+.5,h);body+=`<path d="m${p[0]-2} ${p[1]}v-3m3 4 1-4" stroke="#5f7f48" stroke-width="1"/>`;}}
}
if(!sourceDrawn)body+=waterfall(u);
const smooth=t=>{const v=clamp(t);return v*v*(3-2*v);};
const travel=hero?0:smooth((u-.12)/.26)*(1-smooth((u-.68)/.27));
const angle=Math.PI*travel,cam=[343-207*Math.cos(angle),244+100*Math.sin(angle)+61*Math.cos(angle)],turn=-36*travel;
const camera=`<g data-camera="true" data-travel="${travel.toFixed(5)}" transform="translate(${cam[0].toFixed(3)},${cam[1].toFixed(3)}) rotate(${turn.toFixed(3)} 18 4)"><path d="m0 0 17-9 16 8-17 10Z" fill="#f5f4ed" stroke="#6d8266"/><path d="m0 0v12l16 9V9m0 12 17-9V-1" fill="#e0e6d7" stroke="#6d8266"/><path d="m33 3 12-3v12l-12-5" fill="#c8d4ba" stroke="#6d8266"/><circle cx="8" cy="2" r="2" fill="#6d8266"/></g>`;
const rayStart=[cam[0]+34,cam[1]+4],look=[325*(1-travel)+650*travel,265*(1-travel)+200*travel],spread=37;
const rays=`<path d="M${rayStart[0].toFixed(3)} ${rayStart[1].toFixed(3)} ${look[0].toFixed(3)} ${(look[1]-spread).toFixed(3)} M${rayStart[0].toFixed(3)} ${rayStart[1].toFixed(3)} ${look[0].toFixed(3)} ${(look[1]+spread).toFixed(3)}" stroke="#7e956e" stroke-dasharray="4 5" opacity=".6"/>`;
const annotation='';
return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 445" width="100%" role="img" aria-labelledby="${uid}-title ${uid}-desc"><title id="${uid}-title">A persistent voxel world evolving over time</title><desc id="${uid}-desc">Concept animation: water spreads and trees grow while the camera looks away, then returns. A placed block persists across time.</desc><defs><filter id="${uid}-shadow" x="-30%" y="-100%" width="160%" height="300%"><feGaussianBlur stdDeviation="14"/></filter></defs><ellipse cx="328" cy="355" rx="185" ry="27" fill="#677d54" opacity=".12" filter="url(#${uid}-shadow)"/><g opacity=".33" stroke="#b2bea2" fill="none" stroke-width=".6"><path d="m58 263 269-140 270 140-270 141Z"/><path d="m86 278 269-140m-216 167 269-140m-216 167 269-140m-216 167 269-140M274 151l270 140M221 179l270 140M168 207l270 140M115 235l270 140"/></g>${body}${rays}${camera}${annotation}</svg>`;
}

$('hero-world').innerHTML=scene(.86,true);
const stages=[
  'A world ready to explore.',
  'The camera moves away.',
  'Water flows. Trees grow.',
  'Return to an evolved world.',
];
let progress=0,frame=null,lastRender=0,lastStage=-1,started=false;
function update(){
  const stage=progress===0?0:progress<.38?1:progress<.78?2:3;
  $('goal-world').innerHTML=scene(progress);
  $('world-time').value=Math.round(progress*100);
  $('world-time').setAttribute('aria-valuetext',`${Math.round(progress*100)}% — ${stages[stage]}`);
  if(stage!==lastStage){
    $('goal-stage-title').textContent=stages[stage];
    document.querySelectorAll('[data-world-progress]').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===stage)));
    lastStage=stage;
  }
}
function stop(){
  if(frame!==null)cancelAnimationFrame(frame);
  frame=null;
  $('world-play').textContent='Play';
  $('world-play').setAttribute('aria-label','Play concept animation');
}
function play(){
  if(frame!==null)return;
  started=true;
  if(progress>=1)progress=0;
  const initial=progress,begin=performance.now();lastRender=0;
  $('world-play').textContent='Pause';
  $('world-play').setAttribute('aria-label','Pause concept animation');
  function tick(now){
    progress=Math.min(1,initial+(now-begin)/14000);
    if(now-lastRender>80||progress===1){update();lastRender=now;}
    if(progress<1)frame=requestAnimationFrame(tick);else stop();
  }
  frame=requestAnimationFrame(tick);
}
$('world-play').onclick=()=>{started=true;if(frame!==null)stop();else play();};
$('world-time').oninput=e=>{started=true;stop();progress=Number(e.target.value)/100;update();};
$('world-reset').onclick=()=>{started=true;stop();progress=0;update();};
$('world-edit').onclick=()=>{
  edited=!edited;
  $('world-edit').textContent=edited?'Remove block':'Place block';
  $('world-edit').setAttribute('aria-pressed',String(edited));
  $('hero-world').innerHTML=scene(.86,true);update();
};
document.querySelectorAll('[data-world-progress]').forEach(button=>button.onclick=()=>{
  started=true;stop();progress=Number(button.dataset.worldProgress);update();
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
update();
if('IntersectionObserver' in window){
  new IntersectionObserver(entries=>{
    const visible=entries[0].isIntersecting;
    if(visible&&!started&&!reduced)play();
    else if(!visible)stop();
  },{threshold:.2}).observe($('goal-demonstration'));
  if(!reduced){
    const reveal=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(entry.isIntersecting){entry.target.classList.add('is-visible');reveal.unobserve(entry.target);}
    }),{threshold:.08});
    document.querySelectorAll('.goal-intro,.goal-demonstration,.module-copy,.module-figure,.roadmap').forEach(el=>{
      el.classList.add('motion-reveal');reveal.observe(el);
    });
  }
}
