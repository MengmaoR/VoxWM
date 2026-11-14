"""Scientific state/change projections, including intervention and decay diagnostics."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from vae_p0.visualize import projection
from .storage import Reader
from .tracking import Vocabulary

def save_event(root,scene,event,states,suffix=""):
    if not states:return None
    directory=Path(root)/'visualizations'/scene['scene_id'];directory.mkdir(parents=True,exist_ok=True)
    origin=states[0].meta['chunk_origin'];tiles=[];initial=states[0].tokens
    for state in states[:8]:
        tile=Image.new('RGB',(520,225),'white');draw=ImageDraw.Draw(tile)
        label='SYNTHETIC DRY RUN' if scene.get('synthetic') else 'NATIVE OBSERVATIONS'
        draw.text((5,3),label+' | '+scene['category']+' '+scene['subtype'],fill='black')
        phase=state.meta.get('phase',f'progress={state.progress}')
        draw.text((5,20),f't={state.meta["timestamp"]:.3f}s step={state.meta["simulation_step"]} {phase}',fill='black')
        tile.paste(projection(state,event.v,scene['category']),(5,43))
        old_mask=event.mask(event.records[0]);new_mask=event.mask(state)
        rgb=np.full((16,16,3),245,np.uint8)
        rgb[np.any(old_mask&~new_mask,axis=2)]=(235,60,70)
        rgb[np.any(new_mask&~old_mask,axis=2)]=(45,190,90)
        rgb[np.any((initial!=state.tokens)&old_mask&new_mask,axis=2)]=(235,160,35)
        tile.paste(Image.fromarray(np.flipud(rgb.swapaxes(0,1))).resize((160,160),Image.Resampling.NEAREST),(350,43))
        draw.text((350,207),'green + / red - / amber state',fill='black');tiles.append(tile)
    name='_'.join(map(str,origin))+suffix;sheet=Image.new('RGB',(1040,225*((len(tiles)+1)//2)),(230,235,240))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*520,(i//2)*225))
    path=directory/(name+'.png');sheet.save(path)
    if len(tiles)>1:tiles[0].save(directory/(name+'.gif'),save_all=True,append_images=tiles[1:],duration=600,loop=0)
    return str(path.relative_to(root))

def inspect(root,scene_id):
    """Render pre/post intervention and native trajectory directly from lossless raw logs."""
    from .tracking import Event
    reader=Reader(root);v=Vocabulary(Path(root)/'vocabulary.json');groups={}
    for row in reader.rows:
        if row['scene_id']==scene_id:groups.setdefault(tuple(row['chunk_origin']),[]).append(row['record_id'])
    for origin,ids in groups.items():
        phase_ids={}
        for i in ids:phase_ids.setdefault(reader.rows[i]['phase'],i)
        chosen=sorted(set([*phase_ids.values(),*ids[-max(1,8-len(phase_ids)):]]))[:8];states=[reader.frame(i) for i in chosen]
        if not states:continue
        row=states[0].meta;event=Event(states[0],row['category'],v,row['subtype'])
        scene=dict(scene_id=scene_id,category=row['category'],subtype=row['subtype'],synthetic=row.get('data_source')=='synthetic_dry_run')
        save_event(root,scene,event,states,suffix="_raw_phases")

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--scene-id',required=True)
    a=p.parse_args();inspect(a.root,a.scene_id)
