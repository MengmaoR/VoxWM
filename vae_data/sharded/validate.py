"""Independent per-shard data quality checks; never repairs data or alters READY shards."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import zarr
import pyarrow.parquet as pq
from .protocol import CONFIG,VERSION,CATEGORIES,STRATA,BIOMES,PRODUCTS,Frame,Chain,Vocabulary,split_for_seed,target_changes,d4_targets,SIMULATION_HZ


def validate_native(frame):
    meta=frame.meta
    if meta.get('data_source')!='native':raise ValueError('Non-native sample in formal dataset')
    if meta.get('simulation_hz')!=SIMULATION_HZ:raise ValueError('Expected fixed24 native data; do not mix historical variable-step samples')
    n=meta.get('native_simulation_step');clock=meta.get('native_clock',{})
    if n!=frame.step or any(clock.get(k)!=n for k in('server_tick','client_request_tick','client_completed_tick','lua_tick')):
        raise ValueError('Native clock alignment failed')
    if meta.get('absolute_sim_time')!=n/SIMULATION_HZ or frame.time!=n/SIMULATION_HZ:raise ValueError('Native tick-derived absolute time mismatch')
    index=meta.get('frame_index')
    if index is not None and(meta.get('relative_sim_time')is None or abs(meta['relative_sim_time']-index/SIMULATION_HZ)>1e-6):raise ValueError('Relative native tick clock mismatch')
    evidence=meta.get('native_provenance',{});audit=evidence.get('voxel_only',{})
    if not audit.get('compatible')or any(audit.get(k)!=0 for k in('nontechnical_entities','dropped_items','entity_induced_changes','unsupported_falling_attempts')):
        raise ValueError('Voxel-only provenance failed')
    if not all(evidence.get('engine_build',{}).get(k)for k in('craftium','voxelibre')):raise ValueError('Missing native engine/game revision')
    if meta.get('verified_native_voxels')!=48**3:raise ValueError('Full native region was not verified')
    if meta.get('split')!=split_for_seed(meta['world_seed'],[80,10,10]):raise ValueError('World seed split mismatch')


def confirmed_transition(transition,pre,reaction,vocabulary):
    if not transition.get('confirmed')or transition.get('after')not in PRODUCTS:return False
    p=np.asarray(transition['position'],int);center=np.asarray(reaction.meta['center_origin'])
    if not(np.all(p>=center)and np.all(p<center+16)):return False
    contact=transition.get('contact',{});water=np.asarray(contact.get('water',[]));lava=np.asarray(contact.get('lava',[]))
    if water.shape!=(3,)or lava.shape!=(3,)or np.abs(water-lava).sum()!=1:return False
    water_name=contact.get('water_node','');lava_name=contact.get('lava_node','')
    if 'water'not in water_name or 'lava'not in lava_name:return False
    relative=tuple((p-np.asarray(reaction.meta['region_origin'])).tolist())
    after=vocabulary.entries[int(reaction.tokens[relative])][0];before=vocabulary.entries[int(pre.tokens[relative])][0]
    if after!=transition['after']or before==after:return False
    if after=='mcl_core:stone':valid=np.array_equal(p,water)and np.array_equal(lava,water+[0,0,1])and'water'in transition.get('before','')
    else:
        expected='mcl_core:obsidian'if 'source'in lava_name else'mcl_core:cobble'
        valid=np.array_equal(p,lava)and after==expected and water[2]>=lava[2]and'lava'in transition.get('before','')
    event_tick=transition.get('clock',{}).get('native_simulation_step')
    if event_tick is not None:return bool(valid and pre.step<event_tick<=reaction.step)
    event_us=transition.get('clock',{}).get('engine_us')
    return bool(valid and event_us is not None and pre.meta['engine_us']<event_us<=reaction.meta['engine_us'])


def validate_chain(chain,vocabulary,require_native=True):
    frames=chain.frames
    if len(frames)!=5:raise ValueError('Chain must have exactly five states')
    if len({frame.center.tobytes()for frame in frames})!=5:raise ValueError('Center states are duplicated')
    first=frames[0]
    if any('candidate_index'in frame.meta for frame in frames):
        indices=[frame.meta.get('candidate_index')for frame in frames];count=first.meta.get('candidate_count')
        if indices!=list(range(indices[0],indices[0]+5))or not isinstance(count,int)or not 5<=count<=10 or indices[0]<0 or indices[-1]>=count:raise ValueError('Selected states are not a consecutive candidate window')
    for frame in frames:
        if frame.meta['center_origin']!=first.meta['center_origin']or frame.meta['world_seed']!=first.meta['world_seed']or frame.meta['rollout_id']!=first.meta['rollout_id']:raise ValueError('Chain mixes source chunks/worlds/rollouts')
        if require_native:validate_native(frame)
    for pre,post in zip(frames,frames[1:]):
        if pre.step>=post.step or pre.time>=post.time:raise ValueError('Chain time/step not increasing')
        if target_changes(pre,post,vocabulary,chain.category)==0:raise ValueError('Only neighbor/unrelated changes in chain')
    if chain.category=='D4':
        if chain.anchor is None:raise ValueError('D4 anchor missing')
        if chain.anchor.key in{frame.key for frame in frames}:raise ValueError('Initial anchor was used as a candidate')
        if require_native:validate_native(chain.anchor)
        if chain.anchor.time>=first.time:raise ValueError('Anchor must precede all selected states')
        if any(chain.anchor.meta[key]!=first.meta[key]for key in('world_seed','center_origin','rollout_id')):raise ValueError('Anchor comes from a different world/region/rollout')
        keys={frame.key:frame for frame in frames};complete=False
        for event in chain.events:
            if not all(event.get(role)in keys for role in('pre','reaction','post')):continue
            pre,reaction,post=[keys[event[role]]for role in('pre','reaction','post')]
            if not pre.time<reaction.time<post.time:continue
            observed=event.get('pre_observation')
            if observed and(not observed['equal_center_to_reference']or observed['sim_time']<pre.time or observed['sim_time']>=reaction.time):continue
            if not any(confirmed_transition(t,pre,reaction,vocabulary)and(not observed or observed['engine_us']<t['clock']['engine_us'])for t in event['transitions']):continue
            if not all(dict(event_id=event['id'],role=role)in keys[event[role]].meta.get('event_roles',[])for role in('pre','reaction','post')):continue
            complete=True
        if not complete:raise ValueError('No complete central Pre-Reaction-Post event in window')
    return True


def quota_errors(rows,chains,shard):
    errors=[];counts=Counter(r['category']for r in rows)
    if dict(counts)!=CATEGORIES:errors.append('Expected Static500 and D1/D2/D3/D4 each125')
    static=Counter((r.get('biome_group'),r.get('stratum'))for r in rows if r['category']=='Static')
    expected=Counter({(biome,stratum):quota for biome in BIOMES for stratum,quota in STRATA.items()})
    if static!=expected:errors.append('Static biome/stratum quotas do not match 25 × (8/4/4/4)')
    by_category=Counter(chain['category']for chain in chains)
    if by_category!=Counter({kind:25 for kind in('D1','D2','D3','D4')}):errors.append('Expected 25 complete chains per dynamic category')
    subtypes=Counter(chain['subtype']for chain in chains if chain['category']=='D4')
    if subtypes!=Counter(d4_targets(shard)):errors.append('D4 subtype chain rotation does not match shard recipe')
    return errors


def check_shard(root,path,require_complete=True,require_native=True):
    from .storage import rows_at,source_key
    root,path=Path(root),Path(path);errors=[]
    if not path.exists():return dict(valid=False,complete=False,errors=['Shard directory not found'])
    try:
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['version']!=VERSION:raise ValueError('Wrong shard format')
        if require_native and manifest.get('source_mode')!='native':raise ValueError('Fixture shard is not native')
        rows,chains=rows_at(path);vocabulary=Vocabulary(root/'vocabulary.json')
        arrays=zarr.open_group(str(path/'snapshots.zarr'),mode='r');anchors=zarr.open_group(str(path/'anchors.zarr'),mode='r')
        for name,dtype in(('tokens',np.uint16),('light',np.uint8)):
            if arrays[name].shape!=(len(rows),48,48,48)or arrays[name].dtype!=dtype:raise ValueError('Snapshot shape/dtype mismatch')
            expected_anchors=sum(c['category']=='D4'for c in chains)
            if anchors[name].shape!=(expected_anchors,48,48,48)or anchors[name].dtype!=dtype:raise ValueError('Anchor shape/dtype mismatch')
        def frame_at(index):
            frame=Frame(arrays['tokens'][index],arrays['light'][index],rows[index])
            if frame.tokens.max()>=min(len(vocabulary.entries),int(rows[index]['vocabulary_size_at_commit'])):raise ValueError('Token is outside its committed vocabulary prefix')
            return frame
        seen=set();physical=set();seed_splits={};seed_counts=Counter()
        for i,row in enumerate(rows):
            if row['snapshot_id']!=i:raise ValueError('Non-contiguous snapshot rows')
            frame=frame_at(i)
            if row.get('physical_frame_id')!=frame.key:raise ValueError('Physical frame registry key mismatch')
            if require_native:validate_native(frame)
            if row['split']!=split_for_seed(row['world_seed'],[80,10,10]):raise ValueError('Seed split mismatch')
            if seed_splits.setdefault(row['world_seed'],row['split'])!=row['split']:raise ValueError('Seed leaked across splits')
            seed_counts[row['world_seed']]+=1
            key=source_key(frame,vocabulary)
            if key in seen:raise ValueError('Repeated same-source center state')
            seen.add(key)
            if frame.key in physical:raise ValueError('Physical frame stored twice')
            physical.add(frame.key)
            if require_native and row['category']=='Static':
                from .protocol import biome_group
                if biome_group(row['surface_biome'],BIOMES)!=row['biome_group']:raise ValueError('Static biome label differs from native source')
                surface=row['sample_surface_y'];height=row['center_origin'][2];stratum=row['stratum']
                if stratum=='surface'and not height<=surface<height+16:raise ValueError('Surface sample misses native surface')
                if stratum=='shallow_underground'and not 8<=surface-(height+8)<32:raise ValueError('Shallow stratum differs from native height')
                if stratum=='deep_cave'and surface-(height+8)<24:raise ValueError('Deep stratum differs from native height')
                if stratum=='elevated_complex':
                    air=np.array([name=='air'for name,_ in vocabulary.entries],bool)[frame.center].mean()
                    if not .02<air<.98 or height<=surface:raise ValueError('Elevated sample is not a mixed native structure')
            if row['category']=='Static'and row.get('biome_transition')and len(set(row.get('biome_groups_present',[])))<2:raise ValueError('Unsupported biome transition label')
        if any(n>100 for n in seed_counts.values()):errors.append('One seed exceeds 100 samples in this shard')
        used=[];anchor_ids=[]
        for record in chains:
            ids=record['snapshot_indices'];used.extend(ids)
            if any('candidate_index'not in rows[i]for i in ids):raise ValueError('Missing candidate-window indices')
            frames=[frame_at(i)for i in ids]
            if any(row['category']!=record['category']or row['chain_id']!=record['id']for row in(rows[i]for i in ids)):raise ValueError('Chain row/category mismatch')
            anchor=None
            if record['category']=='D4':
                index=record['anchor_idx'];anchor_ids.append(index)
                anchor=Frame(anchors['tokens'][index],anchors['light'][index],record['anchor_metadata'])
                if anchor.tokens.max()>=min(len(vocabulary.entries),record.get('anchor_vocabulary_size_at_commit',len(vocabulary.entries))):raise ValueError('Anchor token not in committed vocabulary')
            validate_chain(Chain(record['category'],frames,anchor,record['events'],record['intervention'],record['subtype']),vocabulary,require_native)
        expected_dynamic=[i for i,row in enumerate(rows)if row['category']!='Static']
        if sorted(used)!=expected_dynamic:raise ValueError('Incomplete, repeated, or orphaned dynamic chain membership')
        if sorted(anchor_ids)!=list(range(len(anchor_ids))):raise ValueError('Anchor belongs to rejected/duplicate D4 event')
        parquet=pq.read_table(path/'metadata.parquet').to_pylist()
        if len(parquet)!=len(rows)or any(json.loads(row['metadata_json'])!=meta for row,meta in zip(parquet,rows)):raise ValueError('Parquet differs from committed journal')
        quota=quota_errors(rows,chains,manifest['shard']);complete=not quota
        if require_complete:errors.extend(quota)
        transitions=Counter(row['biome_group']for row in rows if row['category']=='Static'and row.get('biome_transition'))
        return dict(valid=not errors,complete=complete,samples=len(rows),counts=dict(Counter(row['category']for row in rows)),
            transition_counts=dict(transitions),transition_deficits={b:max(0,4-transitions[b])for b in BIOMES if transitions[b]<4},errors=errors)
    except Exception as exc:return dict(valid=False,complete=False,errors=[f'{type(exc).__name__}: {exc}'])
