"""Scene-level rates; synthetic dry-run quantities never become native measurements."""
from collections import Counter,defaultdict
from pathlib import Path
from .storage import dump

def summarize(root,scenes,events,native,budget):
    groups=defaultdict(list)
    for scene in scenes:groups[(scene['setting'],scene['category'],scene['subtype'])].append(scene)
    metrics=[]
    for key,rows in sorted(groups.items()):
        ids={r['scene_id'] for r in rows};clips=[e for e in events if e['scene_id'] in ids]
        samples=sum(e['standard_states'] for e in clips);sim=sum(r.get('observation_seconds',0) for r in rows)
        wall=sum(r.get('wall_seconds',0) for r in rows);eligible=sum(r.get('discovery',{}).get('eligible',0) for r in rows)
        examined=sum(r.get('discovery',{}).get('examined',0) for r in rows)
        latencies=[r['first_native_change_latency'] for r in rows if r.get('first_native_change_latency') is not None]
        candidates=[c for r in rows for c in r.get('selected_candidates',[])]
        attempts=sum(r.get('intervention_attempts',0) for r in rows);successes=sum(r.get('intervention_successes',0) for r in rows)
        rejects=Counter()
        for r in rows:rejects.update(r.get('discovery',{}).get('rejected',{}))
        metrics.append(dict(setting=key[0],category=key[1],subtype=key[2],scene_attempts=len(rows),censored_scene_attempts=sum(bool(r.get('censored')) for r in rows),
            standard_snapshots=samples if native else None,valid_snapshots_per_simulation_hour=samples/sim*3600 if native and sim else None,
            valid_snapshots_per_wall_hour=samples/wall*3600 if native and wall else None,
            end_to_end_scene_wall_seconds=wall if native else None,observation_simulation_seconds=sim if native else None,
            candidate_discovered=sum(r.get('discovery',{}).get('discovered',0) for r in rows) if native else None,
            candidate_examined=examined if native else None,physical_rejections=dict(rejects) if native else {},
            selected_candidates=len(candidates) if native else None,
            biome_counts=dict(Counter(c['features'].get('biome','UNRESOLVED') for c in candidates)) if native else {},
            terrain_counts=dict(Counter(c['features'].get('terrain','unknown') for c in candidates)) if native else {},
            unique_candidate_coordinates=len({tuple(c['position']) for c in candidates}) if native else None,
            controlled_d2_attempt_contexts=dict(Counter(r['interaction_context'] for r in rows)) if native and key[:2]==('C','D2') else {},
            intervention_success_rate=successes/attempts if native and attempts else None,
            candidate_eligibility_rate=eligible/examined if native and examined else None,
            native_event_rate=sum(bool(r.get('native_evolution')) for r in rows)/len(rows) if native else None,
            standard_clip_acceptance_rate=sum(e['standard_states']>0 for e in clips)/len(clips) if native and clips else None,
            aux_short_events=sum(e['aux_short_states']>0 for e in clips) if native else None,
            aux_short_snapshots=sum(e['aux_short_states'] for e in clips) if native else None,
            first_native_change_latencies=latencies if native else [],
            distinct_state_counts=[e['distinct_states'] for e in clips] if native else [],
            changed_voxel_counts=[e['unique_changed_positions'] for e in clips] if native else [],
            intervention_successes=sum(r.get('intervention_successes',0) for r in rows) if native else None,
            intervention_attempts=sum(r.get('intervention_attempts',0) for r in rows) if native else None,
            confirmed_water_lava_products=sum(r.get('confirmed_products',0) for r in rows) if native else None,
            failures=dict(Counter(r['status'] for r in rows if r['status']!='accepted'))))
    data=dict(status='NATIVE_RUN' if native else 'SYNTHETIC_DRY_RUN_ONLY',native_behaviors_verified={category:bool(native and any(s.get('native_evolution') for s in scenes if s['category']==category)) for category in ('D1','D2','D3','D4')},
        statistical_unit='scene attempt / world seed; snapshots within an event are not independent trials',
        authority='User P1 implementation prompt; named 2026-11-14 Drive P1 protocol not found in accessible Drive search.',
        plan_budget=budget,groups=metrics,scenes=scenes,
        paired_natural_scenes=[dict(pair_id=a['pair_id'],a_scene=a['scene_id'],b_scene=b['scene_id'],
            a_status=a['status'],b_status=b['status'],a_native=a.get('native_evolution'),b_native=b.get('native_evolution'),
            same_selected_coordinates=[c['position'] for c in a.get('selected_candidates',[])]==[c['position'] for c in b.get('selected_candidates',[])])
            for a in scenes for b in scenes if a['setting']=='A' and b['setting']=='B' and a['pair_id']==b['pair_id']],
        fixture_standard_snapshots=sum(e['standard_states'] for e in events) if not native else None,
        note='World boot/emerge/activation/capture timestamps are observed engine clocks, not a claim of mapgen t=0. Costs exclude unmeasured throughput assumptions.')
    dump(Path(root)/'statistics.json',data)
    lines=['INCONCLUSIVE — P1 '+('native results; inspect category deficits' if native else 'implementation and synthetic dry run only'),'',
        data['authority'],'',data['statistical_unit'],'',
        '| Setting | Category | Plant | Scenes | Snapshots | Snapshots / sim h | Snapshots / wall h |',
        '|---|---|---|---:|---:|---:|---:|']
    for m in metrics:
        values=[m['standard_snapshots'],m['valid_snapshots_per_simulation_hour'],m['valid_snapshots_per_wall_hour']]
        display=['NOT MEASURED' if v is None else f'{v:.2f}' for v in values]
        lines.append(f'| {m["setting"]} | {m["category"]} | {m["subtype"] or "—"} | {m["scene_attempts"]} | '+ ' | '.join(display)+' |')
    lines+=['','Native ABMs, probabilities, engine timestep, fluids and VAE architecture are unchanged.',
        'A/B use separate engine sessions, frozen candidate coordinates and whole-seed splits. Reproducible selection does not guarantee bit-identical native random outcomes.',
        'Controlled intervention changes are excluded by starting progress at the immediate post-intervention capture. Raw logs retain both sides.',
        'Water/lava products require a changed water/lava voxel and sampled contact geometry. Pre-existing stone is never a reaction. Unsampled fast contacts remain unconfirmed.',
        'Fire merge evidence uses connected observed components; sparse observations can miss short-lived connections.',
        'Bamboo placement calls native on_place (including its native random height); no height forcing or growth function calls.',
        '','No 200K generation or GPU training was run. Native throughput and total collection cost require a user-started measurement.']
    (Path(root)/'VAE_P1_RESULTS.md').write_text('\n'.join(lines)+'\n');return data
