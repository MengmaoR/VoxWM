"""GT-only ABV1 export and lightweight contact sheets from actual VWT1 observations."""
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image,ImageDraw
from voxwm_gt.storage import DeltaWriter
from .trace import TraceReader
from .protocol import Vocabulary,dump
from .native_six import TYPES
from .storage import atomic_json

COLORS={'water':'#298ad1','lava':'#fa6b16','fire':'#f8b12d','stone':'#89939c','cobble':'#687884','obsidian':'#533369'}

def classes(v):
    out=[]
    for i,(name,param)in enumerate(v.entries):
        category=next((kind for kind in('water','lava','fire','obsidian','cobble','stone')if kind in name),'other')
        color=COLORS.get(category,'#a29579')
        for word,value in(('dirt','#89623e'),('grass','#739b46'),('sand','#d4bd83'),('tree','#926038'),('leaves','#3d8551'),('snow','#e9f3f5')):
            if word in name:color=value
        out.append(dict(id=i,name=name,param2=param,category=category,color=color,air=name in('air','ignore')))
    return out

def top_image(frame,labels,scale=4):
    colors=np.array([tuple(bytes.fromhex(c['color'][1:]))for c in labels],np.uint8)
    visible=np.array([not c['air']for c in labels],bool)[frame.tokens]
    height=np.max(np.where(visible,np.arange(48)[None,None,:],-1),axis=2)
    top=np.take_along_axis(frame.tokens,np.maximum(height,0)[...,None],axis=2)[...,0]
    rgb=colors[top];rgb[height<0]=[24,31,39]
    image=Image.fromarray(np.flip(rgb.transpose(1,0,2),axis=0)).resize((48*scale,48*scale),Image.Resampling.NEAREST)
    ImageDraw.Draw(image).rectangle((16*scale,16*scale,32*scale-1,32*scale-1),outline='#f8dc75',width=1)
    return image

def one_trace(source,destination):
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    r=TraceReader(source);v=Vocabulary(source/'vocabulary.json');labels=classes(v);dump(destination/'classes.json',labels)
    shutil.copyfile(source/'vocabulary.json',destination/'vocabulary.json')
    writer=DeltaWriter(destination/'voxels.bin');images=[];times=[]
    rows=r.rows;chosen=r.meta.get('selected_indices',[])
    significant={0,len(rows)-1,*chosen}
    significant.update(row['index']for row in rows if any(s in row['stages']for s in('before_intervention','after_intervention')))
    for role in('reaction','post'):
        matches=[row['index']for row in rows if role in row['stages']]
        if matches:significant.add(matches[0])
    # Diagnose center dynamics even when the changed water lies under a static surface.
    # This is a real categorical slice with marked observed differences, not a new state.
    ref_index=next((row['index']for row in rows if'after_intervention'in row['stages']),0)
    reference=None;histogram=np.zeros(16,np.int64)
    for j,f in enumerate(r.frames()):
        if j==ref_index:reference=f.center.copy()
        if reference is not None:histogram+=(f.center!=reference).sum(axis=(0,1))
    focus=int(histogram.argmax())if histogram.any()else 8
    colors=np.array([tuple(bytes.fromhex(c['color'][1:]))for c in labels],np.uint8)
    tiles={}
    for f,row in zip(r.frames(),rows):
        writer.append(f.tokens)
        panel=Image.new('RGB',(284,164),'#121a23');panel.paste(top_image(f,labels,3),(0,0))
        sliced=colors[f.center[:,:,focus]].copy();changed=f.center[:,:,focus]!=reference[:,:,focus]
        sliced[changed]=(sliced[changed]*.35+np.array([245,65,55])*.65).astype(np.uint8)
        panel.paste(Image.fromarray(np.flip(sliced.transpose(1,0,2),axis=0)).resize((128,128),Image.Resampling.NEAREST),(152,8))
        text=ImageDraw.Draw(panel);text.text((3,146),'Full48 top | center16 slice',fill='white')
        text.text((152,138),f'Up={focus+16} red=change',fill='#f6bd9a')
        images.append(panel);times.append(f.time)
        if row['index']in significant:tiles[row['index']]=panel
    writer.close()
    plan={}
    planning_dir=source/('planning_followup'if(source/'planning_followup/summary.json').exists()else'planning')
    if (planning_dir/'summary.json').exists():
        plan=json.loads((planning_dir/'summary.json').read_text());(destination/'planning').mkdir(exist_ok=True)
        for path in planning_dir.glob('selected_*.bin'):shutil.copy2(path,destination/'planning'/path.name)
    summary=dict(unique_type=r.meta['unique_type'],source='real_native',frames=rows,selected_indices=chosen,
        native_events=r.meta.get('native_events',[]),selection=r.meta.get('selection',{}),planning=plan,
        axes='ENU, voxel center at world integer node coordinate; viewer centers the48cube',
        time_semantics='actual sim_time / simulation_step; no interpolated voxel states')
    dump(destination/'frames.json',summary)
    status=summary['selection'].get('status')or{};status_text=str(status.get('status','failed'))+': '+str(status.get('reason',''))
    if images:
        # Native times determine GIF durations; only display durations are bounded.
        duration=[max(40,min(1500,int((b-a)*400)))for a,b in zip(times,times[1:])]+[800]
        images[0].save(destination/'evolution.gif',save_all=True,append_images=images[1:],duration=duration,loop=0,optimize=True)
        sheet=Image.new('RGB',(4*292,55+max(1,(len(tiles)+3)//4)*202),'#121a23');draw=ImageDraw.Draw(sheet)
        draw.text((8,6),r.meta['unique_type']+' | Actual Native GT | '+status_text[:105],fill='white')
        draw.text((8,25),'ENU top-down + actual center slice; yellow=center16; red=observed change since initialization',fill='#b6c7d5')
        for n,(index,image)in enumerate(tiles.items()):
            x=n%4*292;y=55+n//4*202;sheet.paste(image,(x,y));row=rows[index]
            draw.text((x+2,y+168),f"#{index} t={row['metadata']['sim_time']:.3f}s step={row['metadata']['simulation_step']}",fill='white')
            draw.text((x+2,y+183),','.join(row['stages'])[:27],fill='#d8c873')
        sheet.save(destination/'contact_sheet.png')
    else:
        image=Image.new('RGB',(600,160),'#121a23');ImageDraw.Draw(image).text((12,12),r.meta['unique_type']+' | No native snapshot\n'+status_text[:90],fill='white')
        image.save(destination/'contact_sheet.png')
        image.save(destination/'evolution.gif')
    text=f"# {r.meta['unique_type']} — {status_text}\n\nActual observed frames: {len(rows)}; accepted five-state indices: {chosen}.\n\n"
    text+='Precomputation is an uncertain reachability envelope, not GT. All voxel states in this export come from native observations.\n\n'
    if plan.get('native_intervention_executed')is False:text+='**Post-run CPU plan only: proposed sources were NOT executed. The native rejection remains unchanged.**\n\n'
    text+='| Planning query | Legal | Center-reachable | Qualified | Kernel seconds |\n|---|---:|---:|---:|---:|\n'
    for name,m in plan.get('metrics',{}).items():text+=f"| {name} | {m['legal_source_count']} | {m['reachable_candidate_count']} | {m['qualified_candidate_count']} | {m['reachability_seconds']:.5f} |\n"
    text+='\nSources (world ENU): '+str([(d['kind'],d['position'])for d in plan.get('selected_sources',[])])+'.\n\n'
    text+=f"Relevant native center voxel updates: {status.get('center_target_changes','not reached')}; relevant change observations: {status.get('center_transition_observations','not reached')}; sampled raw reaction writes: {len(summary['native_events'])}.\n\n"
    text+='Accepted state times/steps: '+str([(rows[i]['metadata']['sim_time'],rows[i]['metadata']['simulation_step'])for i in chosen])+'.\n\n'
    text+='See `frames.json` for all observed times/stages and reaction/product positions; `report.json` has machine-readable statistics. Full raw trace and all-point source tables remain in the parent attempt.\n'
    if summary['unique_type']=='Static'and rows:
        m=rows[0]['metadata']
        text=f"# Static — {status_text}\n\nOne actual native snapshot, not a dynamic chain.\n\nBiome: {m['biome_group']} ({m['surface_biome']}); stratum: {m['stratum']}.\n\nS={m['surface_reference_y']}, C={m['selected_chunk_center_y']}; full Region Up=[{m['region_height_min']},{m['region_height_max']}].\n\nAir={m['air_fraction']:.6f}; Void={m['void_fraction']:.6f}. See frames.json for native coordinates and source metadata.\n"
    (destination/'report.md').write_text(text)
    dump(destination/'report.json',dict(status=status,planning=plan,native_frames=len(rows),accepted_indices=chosen,
        actual_center_changes=sum(row['center_changes']for row in rows if'native_observation'in row['stages']),
        actual_neighbor_changes=sum(row['neighbor_changes']for row in rows if'native_observation'in row['stages']),native_events=summary['native_events']))
    return summary

def export(root,report_dir=None):
    root=Path(root);out=root/'viewer';out.mkdir(exist_ok=True);entries=[]
    prior=json.loads((out/'manifest.json').read_text()).get('types',[])if(out/'manifest.json').exists()else[]
    retained={entry['id']:entry for entry in prior if entry.get('retained_from')}
    native=json.loads((root/'native_results.json').read_text())if(root/'native_results.json').exists()else{}
    labels=[*TYPES,'Static']if(root/'traces/Static').exists()or'Static'in retained else TYPES
    for label in labels:
        pattern='scene_*/trace.json'if label=='Static'else'attempt_*/trace.json'
        attempts=sorted((root/'traces'/label).glob(pattern));candidates=[];options=[]
        for attempt in attempts:
            meta=json.loads(attempt.read_text());source=attempt.parent
            relative=f'{label}/{source.name}';one_trace(source,out/relative)
            option=dict(attempt=source.name,path=relative,available=bool(meta['frames']))
            if label=='Static'and meta['frames']:
                m=meta['frames'][0]['metadata'];option['label']=f"{m['biome_group']} / {m['stratum']} / C={m['selected_chunk_center_y']}"
            options.append(option)
            candidates.append((bool(meta.get('selected_indices')),bool(meta['frames']),attempt))
        chosen=max(candidates,key=lambda item:(item[0],item[1],str(item[2])))if candidates else None
        if chosen:
            source=chosen[2].parent;relative=f'{label}/{source.name}'
            entries.append(dict(id=label,path=relative,attempt=source.name,available=bool(chosen[1]),attempts=options))
            if report_dir:
                dest=Path(report_dir)/label;dest.mkdir(parents=True,exist_ok=True)
                for name in('contact_sheet.png','evolution.gif','report.md','report.json'):shutil.copy2(out/relative/name,dest/name)
        elif(label=='Static'or label in native.get('skipped_types',[]))and label in retained and(out/retained[label]['path']/'frames.json').exists():entries.append(retained[label])
        else:entries.append(dict(id=label,path=None,available=False,attempts=[],reason='skipped by user; not rerun'if label in native.get('skipped_types',[])else'not attempted within native budget'))
    dump(out/'manifest.json',dict(source='Actual Native GT only',types=entries,run_directory=str(root),retained_types=[entry['id']for entry in entries if entry.get('retained_from')]))
    return out


def publish_static(source,target):
    """Append only the new derived Static preview; old native GT stays untouched."""
    source,target=Path(source),Path(target);out=target/'viewer'
    manifest=json.loads((out/'manifest.json').read_text())
    if any(entry['id']=='Static'for entry in manifest['types']):raise FileExistsError('This viewer already has a Static collection; do not overwrite it')
    current=json.loads((source/'viewer/manifest.json').read_text())
    entry=dict(next(item for item in current['types']if item['id']=='Static'),retained_from=str(source))
    shutil.copytree(source/'viewer/Static',out/'Static')
    manifest['types'].append(entry)
    manifest['retained_types']=list(dict.fromkeys([*manifest.get('retained_types',[]),'Static']))
    atomic_json(out/'manifest.json',manifest)
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True,type=Path);parser.add_argument('--report-dir',type=Path)
    a=parser.parse_args();print(export(a.dataset,a.report_dir))
