"""One-voxel Static sampling windows; independent of the engine MapBlock grid."""
import numpy as np
from .protocol import (STATIC_HEIGHT_POLICY_VERSION,REGION_CAPTURE_VERSION,
    STATIC_CENTER_MIN,STATIC_CENTER_MAX,WORLD_BOTTOM,WORLD_TOP)

MAX_HEIGHT_ATTEMPTS=16

class HeightExhausted(ValueError):pass


def height_bounds(surface,stratum):
    if not isinstance(surface,(int,np.integer)):raise ValueError('Native surface height must be an integer')
    intervals={'surface':(surface-7,surface+8),'elevated_complex':(surface+9,STATIC_CENTER_MAX),
        'shallow_underground':(surface-24,surface-8),'deep_cave':(STATIC_CENTER_MIN,surface-32)}
    lower,upper=intervals[stratum]
    return max(STATIC_CENTER_MIN,lower),min(STATIC_CENTER_MAX,upper)


class HeightSearch:
    def __init__(self,surface,stratum,rng):
        self.lower,self.upper=height_bounds(surface,stratum);self.rng=rng;self.tried=set()
    def draw(self):
        remaining=[c for c in range(self.lower,self.upper+1)if c not in self.tried]
        if not remaining:return None
        # Uniform integers over the complete interval initially, then uniformly
        # over remaining integers: failed heights are never sampled twice.
        c=int(self.rng.integers(self.lower,self.upper+1))if not self.tried else remaining[int(self.rng.integers(len(remaining)))]
        self.tried.add(c);return c
    def reject(self,c,reason):
        if reason=='air_too_high':self.upper=min(self.upper,c-1)


def ratios(frame,vocabulary):
    names=np.array([name for name,_ in vocabulary.entries],object)
    full=names[frame.tokens]
    if np.any(full=='ignore'):raise ValueError('Data integrity: ignore/unloaded voxel in Static region')
    center=full[16:32,16:32,16:32]
    return float(np.mean(center=='air')),float(np.mean(center=='mcl_core:void'))


def rejection(stratum,air,void):
    if stratum=='elevated_complex':
        if air>=.98:return 'air_too_high'
        if air<.10:return 'air_too_low'
    if stratum in('shallow_underground','deep_cave')and void==1.:return 'all_void'
    return None


def sample(world,site,stratum,rng,location,attempt=None):
    raw=location['surface_y']
    if not np.isfinite(raw)or raw!=int(raw):raise ValueError('Native surface must be an integer coordinate')
    surface=int(raw);search=HeightSearch(surface,stratum,rng)
    lower,upper=search.lower,search.upper;history=[]
    if attempt is not None:attempt['height_attempts']=history
    for index in range(1,MAX_HEIGHT_ATTEMPTS+1):
        c=search.draw()
        if c is None:break
        origin=[int(site[0])//16*16,int(site[1])//16*16,c-8]
        frame,response=world.prepare_at(origin,surface,static_window=True)
        air,void=ratios(frame,world.v);reason=rejection(stratum,air,void)
        history.append(dict(center_y=c,air_fraction=air,void_fraction=void,rejection_reason=reason))
        if reason:
            search.reject(c,reason);continue
        frame.meta.update(stratum=stratum,sampling_window_kind='static',surface_reference_y=surface,
            selected_chunk_center_y=c,region_height_min=c-24,region_height_max=c+23,
            height_candidate_lower=lower,height_candidate_upper=upper,height_attempt_count=index,
            height_attempts=list(history),air_fraction=air,void_fraction=void,height_rejection_reason=None,
            static_height_policy_version=STATIC_HEIGHT_POLICY_VERSION,
            terrain_heights=location['heights'],terrain_native=response['chunks'][13]['stratum'])
        return frame,response
    exhausted=not any(c not in search.tried for c in range(search.lower,search.upper+1))
    reason='height_candidates_exhausted'if exhausted else'height_attempt_limit'
    if attempt is not None:attempt['height_rejection_reason']=reason
    raise HeightExhausted(reason+': '+stratum)


def validate_static(frame,vocabulary):
    """Recompute geometry and ratios from actual arrays, not reported fractions."""
    m=frame.meta
    if m.get('sampling_window_kind')!='static'or m.get('static_height_policy_version')!=STATIC_HEIGHT_POLICY_VERSION:raise ValueError('Static height policy differs; old datasets remain read-only')
    origin=np.asarray(m['center_origin']);region=np.asarray(m['region_origin'])
    if origin.shape!=(3,)or region.shape!=(3,)or origin.dtype.kind not in'iu'or region.dtype.kind not in'iu':raise ValueError('Static coordinates must be integers')
    c=int(origin[2])+8
    surface=m['surface_reference_y'];stratum=m['stratum'];lower,upper=height_bounds(surface,stratum)
    if not np.array_equal(region,origin-16):raise ValueError('Static region/center geometry mismatch')
    if not WORLD_BOTTOM<=region[2] or region[2]+47>WORLD_TOP:raise ValueError('Full Static region exceeds overworld bounds')
    if m['selected_chunk_center_y']!=c or m['region_height_min']!=region[2]or m['region_height_max']!=region[2]+47:raise ValueError('Static height metadata differs from coordinates')
    if m['height_candidate_lower']!=lower or m['height_candidate_upper']!=upper or not lower<=c<=upper:raise ValueError('Static center is outside its stratum interval')
    if m.get('sample_surface_y')!=surface:raise ValueError('Static reference differs from native terrain source')
    air,void=ratios(frame,vocabulary)
    if m['air_fraction']!=air or m['void_fraction']!=void:raise ValueError('Static Air/Void fractions differ from voxels')
    reason=rejection(stratum,air,void)
    if reason or m['height_rejection_reason']is not None:raise ValueError('Rejected Static height: '+str(reason))
    history=m.get('height_attempts',[]);seen=set();ceiling=upper
    if len(history)!=m['height_attempt_count']or not 1<=len(history)<=MAX_HEIGHT_ATTEMPTS:raise ValueError('Invalid Static height attempt count')
    for row in history:
        h=row['center_y']
        if not isinstance(h,int)or h in seen or not lower<=h<=ceiling:raise ValueError('Repeated/out-of-range Static height retry')
        seen.add(h)
        if row['rejection_reason']=='air_too_high':ceiling=h-1
    if history[-1]['center_y']!=c or history[-1]['rejection_reason']is not None:raise ValueError('Static history does not end at accepted height')
    if m.get('data_source')=='native':
        if m.get('region_capture_version')!=REGION_CAPTURE_VERSION or m.get('capture_reference_equal')is not True:raise ValueError('Static arbitrary-origin capture was not compared with VoxelManip')
    return True
