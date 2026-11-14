"""Small diagnostic projections of actual collected chunks, including failed clips."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

COLORS={'D1':(40,132,242),'D2':(244,93,20),'D3':(240,45,25),'D4':(38,190,65)}

def projection(state,vocabulary,category):
    names,_=vocabulary.decode(state.tokens)
    solid=(names!='air')&(names!='ignore')
    rgb=np.zeros((16,16,16,3),np.uint8);rgb[:]=(172,150,113)
    for token,(name,param2) in enumerate(vocabulary.entries):
        color=(172,150,113)
        if 'stone' in name:color=(128,130,137)
        if 'dirt' in name:color=(133,88,47)
        if any(n in name for n in ('grass','leaves','bamboo','vine')):color=(54,140,57)
        if 'water' in name:color=COLORS['D1']
        if 'lava' in name:color=COLORS['D2']
        if name.startswith('mcl_fire:'):color=COLORS['D3']
        shade=0.65+0.35*((int(param2)%16)/15)
        rgb[state.tokens==token]=tuple(int(c*shade) for c in color)
    panels=[]
    # Top view and side view are maximum target projections, with a surface context.
    target=vocabulary.mask(state.tokens,category) if category!='Ordinary' else np.zeros(solid.shape,bool)
    for axis in (2,1):
        occ=np.moveaxis(solid,axis,-1);cols=np.moveaxis(rgb,axis,-2)
        hit=np.any(occ,axis=-1);top=15-np.argmax(occ[...,::-1],axis=-1)
        rows,columns=np.indices(hit.shape)
        picture=cols[rows,columns,top].copy();picture[~hit]=(242,245,248)
        if category!='Ordinary':picture[np.any(target,axis=axis)]=COLORS[category]
        picture=np.flipud(picture.swapaxes(0,1))
        panels.append(Image.fromarray(picture).resize((160,160),Image.Resampling.NEAREST))
    result=Image.new('RGB',(330,160),'white');result.paste(panels[0],(0,0));result.paste(panels[1],(170,0))
    return result


def save_visuals(root,items,vocabulary):
    folder=Path(root)/'visualizations';folder.mkdir(exist_ok=True);paths={}
    for category,(states,valid) in items.items():
        if not states:continue
        states=states[:8];tiles=[];records=[]
        for index,state in enumerate(states):
            tile=Image.new('RGB',(350,225),'white');draw=ImageDraw.Draw(tile)
            draw.text((8,4),f'{category} state {index} | step {state.meta["simulation_step"]}',fill='black')
            count=int(vocabulary.mask(state.tokens,category).sum()) if category!='Ordinary' else 0
            draw.text((8,21),f't={state.meta["timestamp"]:.3f}s progress={state.progress} target={count}',fill='black')
            tile.paste(projection(state,vocabulary,category),(8,42))
            draw.text((8,207),'Top / side: target projection + surface context',fill='black')
            tiles.append(tile);records.append(dict(progress=state.progress,**state.meta))
        title=f'{category}: '+('sampled observations' if valid else 'NOT A STANDARD CLIP; real candidate only')
        sheet=Image.new('RGB',(700,35+225*((len(tiles)+1)//2)),(230,234,238))
        ImageDraw.Draw(sheet).text((8,10),title,fill='black')
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*350,35+(i//2)*225))
        image=folder/(category+'.png');sheet.save(image)
        if len(tiles)>1:tiles[0].save(folder/(category+'.gif'),save_all=True,append_images=tiles[1:],duration=650,loop=0)
        np.savez_compressed(folder/(category+'.npz'),tokens=np.stack([s.tokens for s in states]),light=np.stack([s.light for s in states]))
        (folder/(category+'.json')).write_text(json.dumps(dict(category=category,standard_clip=valid,states=records),indent=2)+'\n')
        paths[category]=str(image.relative_to(root))
    return paths
