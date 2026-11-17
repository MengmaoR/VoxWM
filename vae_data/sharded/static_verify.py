"""Finite read-only Native Static capture verification; never creates a shard."""
from pathlib import Path
import time,traceback
import numpy as np
from .native import Native
from .protocol import Vocabulary,dump,REGION_CAPTURE_VERSION,STATIC_HEIGHT_POLICY_VERSION
from .generate import prepare
from .static_height import ratios,validate_static,HeightExhausted
from .validate import validate_native
from .storage import atomic_json

CENTERS=(-39,-38,-37,0,8,9,10,15,16,17,23,24,169,170,171)


def preview(frames,vocabulary,path):
    from PIL import Image,ImageDraw
    from .export_trace import classes,top_image
    labels=classes(vocabulary);colors=np.array([tuple(bytes.fromhex(c['color'][1:]))for c in labels],np.uint8)
    colors[[c['air']for c in labels]]=[24,31,39]
    sheet=Image.new('RGB',(len(frames)*230,320),'#121a23');draw=ImageDraw.Draw(sheet)
    for index,(label,frame)in enumerate(frames.items()):
        x=index*230;sheet.paste(top_image(frame,labels,4),(x+15,28))
        cut=colors[frame.center[8,:,:]].transpose(1,0,2)[::-1]
        sheet.paste(Image.fromarray(cut).resize((160,80),Image.Resampling.NEAREST),(x+30,228))
        draw.text((x+8,8),label,fill='white')
        draw.text((x+8,311),f"C={frame.meta['selected_chunk_center_y']} air={frame.meta['air_fraction']:.3f}",fill='white')
    sheet.save(path)


def run(args):
    root=Path(args.dataset)
    if root.exists():raise FileExistsError('Use a fresh Static verification directory')
    root.mkdir(parents=True);started=time.perf_counter();rng=np.random.default_rng(args.seed)
    result=dict(status='INCONCLUSIVE',static_height_policy_version=STATIC_HEIGHT_POLICY_VERSION,
        region_capture_version=REGION_CAPTURE_VERSION,world_seed=args.seed,site=list(args.site),
        capture_checks=[],examples={},invalid_centers_rejected=[],formal_shard_generated=False)
    frames={}
    try:
        with Native(args.seed,Vocabulary(),time.monotonic()+args.wall_seconds-10,args.craftium_dir,args.renderer)as world:
            site=[int(v)//16*16 for v in args.site];location=world.locate(*site)
            if location['status']!='located':raise ValueError(location['status'])
            result['native_surface']=location
            for c in CENTERS:
                if c in(-39,171):
                    try:world.prepare_at([*site,c-8],location['surface_y'],static_window=True)
                    except ValueError:result['invalid_centers_rejected'].append(c);continue
                    raise ValueError('Out-of-world center was accepted: '+str(c))
                frame,_=world.prepare_at([*site,c-8],location['surface_y'],static_window=True)
                if frame.meta['capture_reference_equal']is not True:raise ValueError('Missing same-tick C++ / VoxelManip equality')
                air,void=ratios(frame,world.v)
                result['capture_checks'].append(dict(center_y=c,center_origin=frame.meta['center_origin'],region_origin=frame.meta['region_origin'],
                    region_height_min=c-24,region_height_max=c+23,cpp_voxelmanip_equal=True,compared_voxels=48**3,
                    compared_fields=['node_content_id','param1_light','raw_param2'],engine_mapblock_count=frame.meta['engine_mapblock_count'],
                    native_clock=frame.meta['native_clock'],air_fraction=air,void_fraction=void))
                atomic_json(root/'static_verification.json',result)
            for stratum in('elevated_complex','shallow_underground','deep_cave'):
                row={}
                try:
                    frame,_=prepare(world,site,stratum,rng,location,row)
                    validate_native(frame);validate_static(frame,world.v)
                    folder=root/'examples'/stratum;folder.mkdir(parents=True)
                    np.savez_compressed(folder/'state.npz',tokens=frame.tokens,light=frame.light)
                    dump(folder/'metadata.json',frame.meta);frames[stratum]=frame
                    row.update(status='PASS',center_origin=frame.meta['center_origin'],center_y=frame.meta['selected_chunk_center_y'],
                        region_height_min=frame.meta['region_height_min'],region_height_max=frame.meta['region_height_max'],
                        air_fraction=frame.meta['air_fraction'],void_fraction=frame.meta['void_fraction'])
                except HeightExhausted as exc:row.update(status='NO_CANDIDATE',reason=str(exc))
                result['examples'][stratum]=row;atomic_json(root/'static_verification.json',result)
            world.v.save(root/'vocabulary.json')
            if frames:preview(frames,world.v,root/'examples.png')
        result['capture_status']='PASS'if len(result['capture_checks'])==13 else'INCONCLUSIVE'
        result['status']='GO'if result['capture_status']=='PASS'and len(frames)==3 else'INCONCLUSIVE'
    except Exception as exc:
        result.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}');(root/'error.txt').write_text(traceback.format_exc())
        if 'world'in locals():world.v.save(root/'vocabulary.json')
    result['total_wall_seconds']=time.perf_counter()-started;atomic_json(root/'static_verification.json',result)
    return result
