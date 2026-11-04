const $ = id => document.getElementById(id);
const ROOT = new URL('./data/', import.meta.url);
const COLORS = {gt:'#346b52', s:'#b56a47', xl:'#7461a6'};
const NAMES = {gt:'Ground Truth', s:'PERSIST-S', xl:'PERSIST-XL'};
const METRICS = {
  water:{title:'Water Voxel Count vs. Time', unit:'Voxels', key:'water_voxels'},
  solid:{title:'Static Solid Disagreement', unit:'Percent'},
  connected:{title:'Source-connected Water vs. Time', unit:'Voxels', key:'source_connected_voxels'},
  basin:{title:'Water in Receiving Basin', unit:'Voxels', key:'water_in_pool'},
};
let reports, reportRequest, episode='C', frame=0, fps=24, series=[], metric='water';
let plot={left:58,right:1080,top:25,bottom:222,max:1,frames:1};
const fmt = n => n == null ? '—' : metric==='solid' ? `${n.toFixed(2)}%` : Math.round(n).toLocaleString('en-US');
const svgElement = (tag,attrs,text) => {
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs))el.setAttribute(key,value);
  if(text!==undefined)el.textContent=text;
  return el;
};
const getReport = async name => {
  const r=await fetch(new URL(name,ROOT));
  if(!r.ok)throw Error(`Unable to read ${name} (${r.status})`);
  return r.json();
};
function dataFor(kind){return reports[kind==='xl'?'xl':'s'][episode];}
function valuesFor(kind){
  return dataFor(kind).curves.map(row=>metric==='solid'
    ? row.static_solid_disagreement_fraction==null ? null : kind==='gt' ? 0 : 100*row.static_solid_disagreement_fraction
    : row[kind==='gt'?'gt_water':'pred_water'][METRICS[metric].key] ?? null);
}
function niceMaximum(max){
  if(max<=0)return 1;
  const scale=10**Math.floor(Math.log10(max/4));
  const step=[1,2,2.5,5,10].map(x=>x*scale).find(x=>x>=max/4);
  return Math.ceil(max/step)*step;
}
function xFor(i){return plot.left+i/Math.max(1,plot.frames-1)*(plot.right-plot.left);}
function yFor(v){return plot.bottom-v/plot.max*(plot.bottom-plot.top);}
function renderChart(){
  if(!reports||$('statistics').hidden)return;
  metric=$('metric').value;
  const spec=METRICS[metric],svg=$('chart');svg.replaceChildren();
  const width=svg.clientWidth||1100,height=svg.clientHeight||160;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  plot.right=width-16;plot.top=20;plot.bottom=height-30;
  $('chartTooltip').hidden=true;
  $('chartTitle').textContent=spec.title;
  series=['gt','s','xl'].map(key=>({key,values:valuesFor(key)}));
  plot.frames=dataFor('gt').frames;
  plot.max=niceMaximum(Math.max(0,...series.flatMap(s=>s.values.filter(x=>x!==null))));
  svg.append(svgElement('text',{x:plot.left,y:13,class:'chart-axis-text'},spec.unit));
  for(let i=0;i<=4;i++){
    const value=i*plot.max/4,y=yFor(value);
    svg.append(svgElement('line',{x1:plot.left,x2:plot.right,y1:y,y2:y,class:'chart-grid'}));
    svg.append(svgElement('text',{x:plot.left-12,y:y+4,'text-anchor':'end',class:'chart-axis-text'},value.toLocaleString('en-US',{maximumFractionDigits:metric==='solid'?1:0})));
    const seconds=i*(plot.frames-1)/fps/4;
    svg.append(svgElement('text',{x:xFor(i*(plot.frames-1)/4),y:plot.bottom+22,'text-anchor':i===0?'start':i===4?'end':'middle',class:'chart-axis-text'},`${seconds.toFixed(1)} s`));
  }
  for(const s of series){
    let d='',gap=true;
    s.values.forEach((v,i)=>{if(v==null){gap=true;return;}d+=`${gap?'M':'L'}${xFor(i).toFixed(2)},${yFor(v).toFixed(2)} `;gap=false;});
    svg.append(svgElement('path',{d,class:'chart-line',stroke:COLORS[s.key],'stroke-dasharray':s.key==='xl'?'6 4':'none'}));
  }
  svg.append(svgElement('line',{id:'chartCursor',y1:plot.top,y2:plot.bottom,class:'chart-cursor'}));
  for(const s of series)svg.append(svgElement('circle',{id:`dot-${s.key}`,r:3.7,fill:COLORS[s.key],stroke:'white','stroke-width':1.5}));
  svg.append(svgElement('rect',{x:plot.left,y:plot.top,width:plot.right-plot.left,height:plot.bottom-plot.top,class:'chart-hit'}));
  updateCursor();
}
function updateCursor(){
  if(!series.length)return;
  const at=Math.min(frame,plot.frames-1),x=xFor(at),line=$('chartCursor');
  if(!line)return;
  line.setAttribute('x1',x);line.setAttribute('x2',x);
  for(const s of series){
    const v=s.values[at],dot=$(`dot-${s.key}`);
    dot.setAttribute('visibility',v==null?'hidden':'visible');
    if(v!=null){dot.setAttribute('cx',x);dot.setAttribute('cy',yFor(v));}
    $({gt:'valueGt',s:'valueS',xl:'valueXl'}[s.key]).textContent=fmt(v);
  }
  $('chartFrame').textContent=`Frame ${at} · ${(at/fps).toFixed(2)} s`;
}
function frameAt(event){
  const svg=$('chart'),point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
  const p=point.matrixTransform(svg.getScreenCTM().inverse());
  if(p.x<plot.left||p.x>plot.right||p.y<plot.top||p.y>plot.bottom)return null;
  return Math.round((p.x-plot.left)/(plot.right-plot.left)*(plot.frames-1));
}
$('chart').addEventListener('pointermove',event=>{
  if(!reports)return;
  const at=frameAt(event),tip=$('chartTooltip');
  if(at===null){tip.hidden=true;return;}
  tip.replaceChildren();
  const heading=document.createElement('strong');heading.textContent=`Frame ${at} · ${(at/fps).toFixed(2)} s`;tip.append(heading);
  for(const s of series){const row=document.createElement('div');row.style.color=COLORS[s.key];row.textContent=`${NAMES[s.key]}  ${fmt(s.values[at])}`;tip.append(row);}
  tip.hidden=false;
  const bounds=$('chart').getBoundingClientRect();
  tip.style.left=`${Math.max(0,Math.min(event.clientX-bounds.left+14,bounds.width-tip.offsetWidth))}px`;
  tip.style.top='16px';
});
$('chart').addEventListener('pointerleave',()=>{$('chartTooltip').hidden=true;});
$('chart').addEventListener('click',event=>{
  const at=frameAt(event);if(at!==null)document.dispatchEvent(new CustomEvent('briefing:seek',{detail:at}));
});
$('metric').onchange=renderChart;
$('toggleStats').onclick=()=>{
  const expanded=$('statistics').hidden;
  $('statistics').hidden=!expanded;
  $('experiment').classList.toggle('has-stats',expanded);
  $('toggleStats').setAttribute('aria-expanded',String(expanded));
  loadReports().then(renderChart);
};
new ResizeObserver(()=>renderChart()).observe($('chart').parentElement);
function selectEpisode(next){
  episode=next;frame=0;
  if(!reports)return;
  fps=dataFor('gt').gt.fps||24;renderChart();
}
document.addEventListener('briefing:loading',e=>selectEpisode(e.detail.episode));
document.addEventListener('briefing:ready',e=>{fps=e.detail.fps;selectEpisode(e.detail.episode);});
document.addEventListener('briefing:frame',e=>{frame=e.detail.frame;updateCursor();});
function loadReports(){
  return reportRequest??=(async()=>{
    try{const [s,xl]=await Promise.all([getReport('comparison_s.json'),getReport('comparison_xl.json')]);reports={s,xl};episode=$('episode').value;fps=dataFor('gt').gt.fps||24;}
    catch(e){$('toggleStats').disabled=true;$('toggleStats').title=e.message;console.error(e);}
  })();
}
const viewerObserver=new IntersectionObserver(entries=>{
  if(!entries.some(entry=>entry.isIntersecting))return;
  viewerObserver.disconnect();
  import('./viewer.js?v=20261104-loading').catch(e=>{
    $('status').className='error';$('status').textContent='Viewer unavailable: '+e.message;
    $('voxelLoading').textContent='The interactive viewer requires WebGL and local experiment files.';
    console.error(e);
  });
},{rootMargin:'600px'});
viewerObserver.observe($('experiment'));
