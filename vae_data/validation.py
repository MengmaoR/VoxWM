"""Bounded REAL native integration. Failures remain failures; no synthetic replacement."""
import json
from pathlib import Path
import time
from .core import dump
from .pipeline import discover,reconstruct,collection,schedule

def report(dataset,results,start):
    rows=results.get('collections',[])
    for row in rows:
        if 'standard_snapshots' not in row:
            row['standard_snapshots']=dataset.db.execute("SELECT COUNT(*) FROM samples WHERE pool='standard' AND json_extract(metadata,'$.rollout_id')=?",(row['rollout_id'],)).fetchone()[0]
    replay=results.get('replay',{});rollouts=replay.get('rollouts',[])
    if rollouts:
        keys=rollouts[0]['initial_hashes']
        replay['matching_initial_chunks']=[key for key in keys if all(r['initial_hashes'].get(key)==keys[key] for r in rollouts)]
        replay['varying_initial_chunks']=[key for key in keys if any(r['initial_hashes'].get(key)!=keys[key] for r in rollouts)]
    native=[r for r in rows if r.get('status')=='complete']
    audits=[r.get('audit',{}) for r in native]+[r.get('audit',{}) for r in results.get('replay',{}).get('rollouts',[])]
    voxel_only=bool(audits) and all(a.get('nontechnical_entities')==0 and a.get('dropped_items')==0 and a.get('entity_induced_changes')==0 and a.get('compatible') for a in audits)
    groups={}
    for category in ('Ordinary','D1','D2','D3','D4'):
        subset=[r for r in rows if r['category']==category]
        sim=sum(r.get('observation_seconds',0) for r in subset);wall=sum(r.get('wall_seconds',0) for r in subset)
        snapshots=sum(r.get('standard_snapshots',0) for r in subset)
        groups[category]=dict(attempts=len(subset),completed=sum(r.get('status')=='complete' for r in subset),
            standard_snapshots=snapshots,aux_snapshots=sum(r.get('aux_snapshots',0) for r in subset),
            native_changes=sum(r.get('native_changes',0) for r in subset),native_growth_positions=sum(r.get('native_growth_positions',0) for r in subset),
            simulation_seconds=sim,wall_seconds=wall,snapshots_per_sim_hour=snapshots/sim*3600 if sim else None,
            snapshots_per_wall_hour=snapshots/wall*3600 if wall else None)
    trimmed=[action for row in rows if row['category']=='D4' for action in row.get('interventions',[]) if action.get('cut_positions')]
    trim_verified=bool(trimmed) and all(action.get('root_preserved') and action.get('metadata_unchanged') and action.get('matches_requested_height') for action in trimmed)
    passed=dict(native_trim_verified=trim_verified,natural_water=any(r['category']=='D1' and not r.get('interventions') and r.get('native_changes',0)>0 for r in rows),
        controlled_fluid=any(r['category'] in('D1','D2') and r.get('interventions') and r.get('native_changes',0)>0 for r in rows),
        fire_propagation=any(r['category']=='D3' and r.get('native_growth_positions',0)>0 for r in rows),
        natural_plant_trim_and_growth=trim_verified and any(r['category']=='D4' and r.get('interventions') and r.get('native_growth_positions',0)>0 for r in rows),
        voxel_only=voxel_only,three_reconstructions=results.get('replay',{}).get('repeats',0)>=3,
        spatial_reconstruction=results.get('replay',{}).get('spatial_initial_state_reproduced',False))
    paths=all(groups[k]['standard_snapshots']>=5 for k in ('D1','D2','D3','D4'))
    results.update(status='GO' if all(passed.values()) and paths else 'INCONCLUSIVE',passed=passed,category_measurements=groups,
        measured_wall_seconds=time.monotonic()-start,counts=dataset.counts(),formal_200k_ready=all(passed.values()) and paths,full_200k_started=False,gpu_training=False)
    dump(dataset.root/'statistics.json',results)
    lines=[results['status']+' — Unified VAE Collection / native integration','',
        'All measurements below come from Craftium/VoxeLibre. Unit fixtures are not included.',
        '',f'Total measured wall time: {results["measured_wall_seconds"]:.2f}s. Voxel-only validation: {voxel_only}.',
        '', '| Category | Attempts | Complete | Standard snapshots | Native changes | Growth positions | Sim seconds | Wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for category,r in groups.items():lines.append(f'| {category} | {r["attempts"]} | {r["completed"]} | {r["standard_snapshots"]} | {r["native_changes"]} | {r["native_growth_positions"]} | {r["simulation_seconds"]:.2f} | {r["wall_seconds"]:.2f} |')
    lines+=['','Validation endpoints: '+json.dumps(passed),
        '','Initial layout equality and exact temporal equality are reported separately in replay_consistency.json. Initial hashes use stable Node Names/raw param2, excluding light.',
        'Falling-node entity transitions are explicitly excluded; a requested falling entity quarantines the whole rollout before training samples commit. No replacement physics is claimed.',
        'Plant interventions use native node_dig, never planting, metadata edits or grow calls. Actual remaining height/root/anchor and drops audit are retained.',
        'No 200K collection or GPU training was started. Dataset-wide biome coverage and sustainable quotas are not proven by this bounded integration.']
    for r in rows:
        if r.get('error'):lines.append('- '+r['category']+': '+r['error'])
        if r.get('visualization'):lines.append('- ['+r['category']+' native visualization]('+r['visualization']+')')
    if results.get('error'):lines.append('Execution error: '+results['error'])
    (dataset.root/'NATIVE_VALIDATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return results

def validate(dataset,deadline,craftium,renderer):
    start=time.monotonic();results=dict(source='real_native',collections=[],replay={})
    previous=dataset.root/'statistics.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('source')=='real_native':
            results=old;results.setdefault('attempt_history',[]).append(dict(status=old.get('status'),error=old.get('error'),wall_seconds=old.get('measured_wall_seconds')))
            results.pop('error',None)
    try:
        # Previously discovered NATURAL locations are transparent search hints, not synthetic scenes.
        sites=[dict(x=80,z=1984,depth=0),dict(x=96,z=1936,depth=0),dict(x=800,z=-1232,depth=0),dict(x=-288,z=1456,depth=0),dict(x=-256,z=1472,depth=32)]
        known=dataset.scenes()
        if not any(s['discovery_source']['site']['x']==80 for s in known):discover(dataset,[191664964],sites,deadline,craftium,renderer)
        all_scenes=dataset.scenes()
        flow=[s for s in all_scenes if s['signals']['water'] and s['discovery_source']['site']['x']==80]
        water=max(flow or[s for s in all_scenes if s['signals']['water']],key=lambda s:s['signals']['water'])
        if not results.get('replay',{}).get('rollouts'):results['replay']=reconstruct(dataset,water,3,2,deadline,craftium,renderer)
        report(dataset,results,start)
        requests=[(water,'Ordinary','',0,'never',None),(water,'D1','',12,'never',None),
            (water,'D1','',15,'always','suspended'),(water,'D2','',20,'always','attached')]
        fire=max((s for s in all_scenes if 'D3'in s['scene_types']),key=lambda s:s['signals']['flammable_signal'])
        requests.append((fire,'D3','',30,'always',None))
        plants=[s for s in all_scenes if s['signals']['bamboo']+s['signals']['reeds']]
        plant=max(plants,key=lambda s:s['signals']['bamboo']+s['signals']['reeds'])
        subtype='bamboo' if plant['signals']['bamboo']>=plant['signals']['reeds']else'reeds'
        for i,(scene,category,subtype0,seconds,intervention,mode) in enumerate(requests):
            if time.monotonic()+25>deadline:break
            key='native_validation_'+str(i)
            if dataset.completed(key):
                r=json.loads(dataset.db.execute('SELECT data FROM jobs WHERE id=?',(key,)).fetchone()[0])
            else:r=collection(dataset,scene,category,subtype0,seconds,deadline,craftium,renderer,intervention,mode,key)
            if not any(x['rollout_id']==r['rollout_id'] for x in results['collections']):results['collections'].append(r)
            report(dataset,results,start);print('NATIVE',category,r['status'],r.get('native_changes'),flush=True)
        remaining=max(0,min(600,deadline-time.monotonic()-30))
        if dataset.completed('native_validation_plant'):
            row=json.loads(dataset.db.execute('SELECT data FROM jobs WHERE id=?',('native_validation_plant',)).fetchone()[0])
            if not any(x['rollout_id']==row['rollout_id'] for x in results['collections']):results['collections'].append(row)
        elif remaining>20:
            r=collection(dataset,plant,'D4',subtype,remaining,deadline,craftium,renderer,'always',None,'native_validation_plant')
            results['collections'].append(r)
    except Exception as exc:results['error']=f'{type(exc).__name__}: {exc}'
    finally:report(dataset,results,start);dataset.flush()
    print('Native report:',dataset.root/'NATIVE_VALIDATION_RESULTS.md',flush=True)
    return results
