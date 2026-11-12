"""Actual yields, missing categories and conditional 200K cost estimates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import numpy as np


def distribution(values):
    if not values:return dict(count=0,min=None,p50=None,p90=None,max=None)
    return dict(count=len(values),min=min(values),p50=float(np.median(values)),p90=float(np.quantile(values,.9)),max=max(values))


def report(root,status=None,error=None,wall_seconds=None):
    root=Path(root);db=sqlite3.connect(root/'dataset.sqlite')
    config=json.loads(db.execute('SELECT config FROM settings').fetchone()[0])
    worlds=[json.loads(row[0]) for row in db.execute('SELECT info_json FROM worlds')]
    sites=[json.loads(row[0]) for row in db.execute('SELECT info_json FROM sites')]
    clips=[(kind,source,json.loads(info)) for kind,source,info in db.execute('SELECT kind,source,info_json FROM clips')]
    counts=dict(db.execute('SELECT category,count(*) FROM snapshots GROUP BY category'))
    seen=sum(s.get('chunk_trajectories',0) for s in sites)
    simulated=sum(s.get('seconds',0) for s in sites)
    all_engine_time=sum(w.get('engine_sim_seconds',0) for w in worlds)
    all_wall_time=sum(w.get('wall_seconds',0) for w in worlds)
    dynamic={}
    for kind in ('water','lava','fire','vegetation'):
        rows=[(source,info) for k,source,info in clips if k==kind]
        natural={i['trajectory_id'] for source,i in rows if source=='natural'}
        valid=[i for _,i in rows if i['standard_clip']]
        dynamic[kind]=dict(natural_changed_trajectories=len(natural),observed_chunk_trajectories=seen,
            natural_discovery_rate_per_chunk_visit=len(natural)/seen if seen else None,
            sites_with_native_presence=sum(kind in s.get('presence',[]) for s in sites),
            valid_clips=len(valid),saved_snapshots=counts.get(kind,0),
            natural_valid_trajectories=len({i['trajectory_id'] for source,i in rows if source=='natural' and i['standard_clip']}),
            natural_saved_snapshots=sum(i['sampled_states'] for source,i in rows if source=='natural'),
            ignition_seeded_saved_snapshots=sum(i['sampled_states'] for source,i in rows if source=='ignited'),
            valid_snapshot_yield_per_world=counts.get(kind,0)/len(worlds) if worlds else None,
            valid_snapshot_yield_per_observation_second=counts.get(kind,0)/simulated if simulated else None,
            distinct_states=distribution([i['distinct_states'] for _,i in rows]),
            sampled_state_count_histogram=dict(Counter(i['sampled_states'] for _,i in rows)),
            cumulative_voxel_updates=distribution([i['cumulative_voxel_updates'] for _,i in rows]),
            change_band_histogram=dict(Counter(i['change_band'] for _,i in rows)),
            valid_change_band_histogram=dict(Counter(i['change_band'] for i in valid)),
            ignition_seeded_trajectories=len({i['trajectory_id'] for source,i in rows if source=='ignited'}),
            invalid_or_duplicate_limited_clips=sum(not i['standard_clip'] for _,i in rows))
    biome_counts=dict(db.execute("SELECT biome_group,count(*) FROM snapshots WHERE category='ordinary' GROUP BY biome_group"))
    estimates={};missing=[];nworld=len(worlds)
    for kind,target in config['quotas'].items():
        count=counts.get(kind,0)
        needed=max(0,target-count)
        estimate=math.ceil(needed*nworld/count) if count and nworld else None
        estimates[kind]=dict(target=target,observed=count,shortfall=needed,
                            estimated_additional_worlds=estimate,
                            estimated_additional_simulated_seconds=estimate*all_engine_time/nworld if estimate is not None and nworld else None,
                            estimated_additional_wall_seconds=estimate*all_wall_time/nworld if estimate is not None and nworld else None)
        if needed and not count:missing.append(kind)
    per_biome_target=config['quotas']['ordinary']//len(config['biome_groups'])
    biome_estimates={}
    for biome in config['biome_groups']:
        count=biome_counts.get(biome,0)
        biome_estimates[biome]=dict(count=count,target=per_biome_target,shortfall=max(0,per_biome_target-count),
                                  estimated_additional_worlds=math.ceil(max(0,per_biome_target-count)*nworld/count) if count and nworld else None)
    unsupported_biomes=[]
    registry=root/'registered_biomes.json'
    if registry.is_file():
        registered=json.loads(registry.read_text())
        unsupported_biomes=[group for group,aliases in config['biome_groups'].items() if not any(a in registered for a in aliases)]
    total_estimate=None
    values=[v['estimated_additional_worlds'] for v in estimates.values()]+[v['estimated_additional_worlds'] for v in biome_estimates.values()]
    if all(v is not None for v in values):total_estimate=max(values,default=0)
    existing=json.loads((root/'report.json').read_text()) if (root/'report.json').is_file() else {}
    result=dict(status=status or existing.get('status','unknown'),error=error if status else existing.get('error'),
                collector_ok=bool(counts and sites) and (status or existing.get('status'))!='failed',
                quotas_met=all(counts.get(k,0)>=n for k,n in config['quotas'].items()),
                snapshot_count=sum(counts.values()),counts={k:counts.get(k,0) for k in config['quotas']},quotas=config['quotas'],
                worlds_attempted=nworld,sites_observed=len(sites),observation_simulated_seconds=simulated,
                engine_simulated_seconds=all_engine_time,world_wall_seconds=all_wall_time,
                latest_command_wall_seconds=wall_seconds if wall_seconds is not None else existing.get('latest_command_wall_seconds'),
                dynamic=dynamic,ordinary_biome_counts=biome_counts,ordinary_biome_targets=biome_estimates,
                raw_biome_counts=dict(db.execute('SELECT biome,count(*) FROM snapshots GROUP BY biome')),
                terrain_counts=dict(db.execute('SELECT terrain,count(*) FROM snapshots GROUP BY terrain')),
                height_distribution=distribution([json.loads(r[0])['height'] for r in db.execute('SELECT info_json FROM snapshots')]),
                split_counts=dict(db.execute('SELECT split,count(*) FROM snapshots GROUP BY split')),
                unsupported_configured_biomes=unsupported_biomes,estimates=estimates,
                estimated_additional_worlds_for_all_quotas_and_biomes=total_estimate,
                zero_yield_categories=missing,
                notes=['Rates are observations from this bounded search, not promises of eventual quota completion.',
                       'Natural discoveries are changed fixed world-chunk trajectories / all observed chunk visits; seeded fire is separate.',
                       'A null estimate means insufficient measured yield; zero observed events do not prove the event is impossible.',
                       'Exact node+param2 content equality prevents duplicate snapshots; no content hashes or artificial states.',
                       'Dynamic clip length 5..8 is unrelated to the VAE optimizer batch size.',
                       'Engine stepping time excludes the initial reset warmup; measured wall time includes initialization.'])
    with (root/'clips.jsonl').open('w') as out:
        for event,kind,seed,source,info in db.execute('SELECT event_id,kind,seed,source,info_json FROM clips ORDER BY event_id'):
            out.write(json.dumps(dict(event_id=event,kind=kind,seed=seed,**json.loads(info)),ensure_ascii=False)+'\n')
    db.close()
    (root/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# VAE pilot native collection report','',f'Status: {result["status"]}; snapshots: {result["snapshot_count"]}; worlds: {nworld}; observed sites: {len(sites)}','',
           '| Category | Snapshots | Target | Shortfall | Natural changed trajectories | Valid clips |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind,target in config['quotas'].items():
        row=dynamic.get(kind,{})
        lines.append(f'| {kind} | {counts.get(kind,0)} | {target} | {target-counts.get(kind,0)} | {row.get("natural_changed_trajectories","—")} | {row.get("valid_clips","—")} |')
    lines+=['',f'Observed simulation: {simulated:.3f} s; engine stepping after initialization (including site search): {all_engine_time:.3f} s; world wall time: {all_wall_time:.3f} s.',
            '',f'Zero-yield categories: {", ".join(missing) or "none"}.',
            f'Combined additional-world estimate: {total_estimate if total_estimate is not None else "UNKNOWN: at least one category/biome lacks measured yield"}.',
            '', 'Per-clip data, exact timestamps/progress, biome/height counts and conditional estimates are in dataset.sqlite and report.json.',
            'This report does not change quotas, accelerate native physics or synthesize missing clips.']
    lines+=['', '| Biome group | Ordinary snapshots |', '| --- | ---: |']
    for group,row in biome_estimates.items():lines.append(f'| {group} | {row["count"]} |')
    lines+=['', 'Terrain counts: '+json.dumps(result['terrain_counts']), 'Height distribution: '+json.dumps(result['height_distribution']),
            '', '| Dynamic | Distinct states min / p50 / max | Valid clip size histogram | Change bands (all candidates) |', '| --- | --- | --- | --- |']
    for kind,row in dynamic.items():
        d=row['distinct_states']
        lines.append(f'| {kind} | {d["min"]} / {d["p50"]} / {d["max"]} | {row["sampled_state_count_histogram"]} | {row["change_band_histogram"]} |')
    if result['error']:lines+=['',f'Error: {result["error"]}']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('Native collection report:',root/'report.md',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)
if __name__=='__main__':main()
