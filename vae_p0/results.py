"""Measured P0 verdict and transparent yield estimates, with no fabricated coverage."""
from collections import Counter
from math import ceil
from pathlib import Path

TITLES={'D1':'Water Flow','D2':'Lava Flow','D3':'Fire Propagation','D4':'Vegetation Growth'}
QUOTAS={'Ordinary':100000,**{k:25000 for k in TITLES}}

def chosen_chunk(trial):
    chunks=trial.get("chunks",[])
    origin=trial.get("selected_chunk_origin")
    if origin is not None:return next(c for c in chunks if c["chunk_origin"]==origin)
    return max(chunks,key=lambda c:(c["distinct_states"],c["cumulative_target_changes"]),default={})

def summarize(output,run):
    rows=output.metadata;counts=Counter(r['category'] for r in rows)
    categories={}
    for category in TITLES:
        trials=[t for t in run['trials'] if t['category']==category]
        valid=[t for t in trials if t['valid_clip']]
        seconds=sum(t.get('observed_seconds',0) for t in trials)
        wall=sum(t['wall_seconds'] for t in trials)
        observations=sum(c['observations'] for t in trials for c in t.get('chunks',[]))
        identical=sum(c['consecutive_identical'] for t in trials for c in t.get('chunks',[]))
        successful_chunks=[chosen_chunk(t) for t in valid]
        categories[category]=dict(snapshots=counts[category],attempts=len(trials),valid_clips=len(valid),
            valid_attempt_fraction=len(valid)/len(trials) if trials else None,
            native_candidates_found=sum(bool(t.get('selected_candidate')) for t in trials),
            candidate_encounter_fraction=sum(bool(t.get('selected_candidate')) for t in trials)/len(trials) if trials else None,
            natural_fire_trials=sum(t.get('ignition_mode')=='natural' for t in trials) if category=='D3' else None,
            controlled_ignitions=sum(t.get('ignitions',0) for t in trials),
            observed_simulation_seconds=seconds,wall_seconds=wall,
            snapshots_per_simulation_hour=counts[category]/seconds*3600 if seconds else None,
            snapshots_per_wall_hour=counts[category]/wall*3600 if wall else None,
            valid_clips_per_simulation_hour=len(valid)/seconds*3600 if seconds else None,
            consecutive_identical_fraction=identical/max(1,observations),
            selected_clip_lengths=[t['sampled_states'] for t in valid],
            cumulative_changes=[c['cumulative_target_changes'] for c in successful_chunks],
            unique_changed_positions=[c['unique_changed_positions'] for c in successful_chunks],
            net_token_differences=[c['net_token_difference'] for c in successful_chunks],
            post_intervention_addition_events_lower_bound=[max(0,c['native_target_additions']-(1 if t.get('ignition_mode')=='controlled_random' else 0)) for t,c in zip(valid,successful_chunks)],
            max_distinct_states_with_native_additions=max((c['distinct_states'] for t in trials for c in t.get('chunks',[]) if c['native_target_additions']>0),default=0),
            distinct_states_per_observed_chunk=[c['distinct_states'] for t in trials for c in t.get('chunks',[])])
    extraction=run.get('verified_native_nodes',0)>0 and run.get('roundtrip_voxels_checked',0)>0
    all_events=all(c['valid_clips'] for c in categories.values())
    verdict='INCONCLUSIVE' if run['status']!='completed' else ('GO' if extraction and all_events else 'NO-GO')
    biome=Counter(r['biome_group'] or 'UNRESOLVED' for r in rows if r['category']=='Ordinary')
    strata=Counter(r['stratum'] for r in rows if r['category']=='Ordinary')
    transitions=sum(r['biome_transition'] for r in rows if r['category']=='Ordinary')
    clips=[chosen_chunk(t) for t in run['trials'] if t['valid_clip']]
    change_bins=Counter('small(<16)' if c['cumulative_target_changes']<16 else 'medium(16..255)' if c['cumulative_target_changes']<256 else 'large(>=256)' for c in clips)
    estimates={}
    for category,quota in QUOTAS.items():
        c=categories.get(category,{})
        rate=c.get('snapshots_per_simulation_hour')
        estimates[category]=dict(target=quota,observed=counts[category],remaining=max(0,quota-counts[category]),
            linear_world_equivalents=ceil(quota/counts[category]) if counts[category] else None,
            linear_simulation_hours=quota/rate if rate else None)
    stats=dict(run,verdict=verdict,counts={category:counts[category] for category in QUOTAS},total_snapshots=len(rows),categories=categories,
        extraction_passed=extraction,ordinary_biome_counts=dict(biome),ordinary_stratum_counts=dict(strata),
        ordinary_transition_count=transitions,major_biome_groups_covered=len(biome),major_biome_groups_target=25,
        successful_chunk_change_bins=dict(change_bins),estimates_200k=estimates,
        estimate_limitation='One known seed and previously discovered candidate hints; linear extrapolations are not a collection forecast. Zero yield => no estimate.',
        recommended_next_step='User reviews P0 before any 200K collection; do not use the older separate-node/param2 collector as a protocol-compliant dataset.')
    lines=[f'{verdict} — VoxWM VAE P0 real native collection', '',
        'Authority: 2026-11-13 data protocol supersedes the 2026-11-12 manifest for collection. No VAE training or 200K generation.',
        '',f'World seed: {run["world_seed"]}; status: {run["status"]}; wall time: {run["wall_seconds"]:.2f}s.',
        f'Native fixed-coordinate comparisons: {run.get("verified_native_nodes",0):,} voxels; exact Combined-Token round trips: {run.get("roundtrip_voxels_checked",0):,} voxels; observed vocabulary K={run["vocabulary_K"]}.',
        '', '| Category | Saved snapshots | Valid clips / attempts | Native observation s | Snapshots / sim hour |',
        '|---|---:|---:|---:|---:|',f'| Ordinary | {counts["Ordinary"]} | — | — | — |']
    for category,c in categories.items():
        rate=c['snapshots_per_simulation_hour'];display='not measurable' if rate is None else f'{rate:.1f}'
        lines.append(f'| {category} {TITLES[category]} | {c["snapshots"]} | {c["valid_clips"]}/{c["attempts"]} | {c["observed_simulation_seconds"]:.2f} | {display} |')
    lines+=['','Only accepted distinct states count as training snapshots. Invalid diagnostic sequences are stored separately, without padding.',
        'One seed cannot validate train/validation/test balance or 25-biome coverage. Whole-seed split assignments are preserved.',
        '',f'Ordinary biome groups: {dict(biome)}.',f'Ordinary actual terrain strata: {dict(strata)}; transition snapshots: {transitions}.',
        f'Observed saved-clip change bins: {dict(change_bins)}. These are descriptive bins, not new sampling quotas.',
        'Vine/bamboo losses alone do not count as D4 growth. Fire requires native spread beyond the single intervention, not just ignition or fire-age changes.',
        '', '## Dynamic outcomes and single bounded repairs']
    for t in run['trials']:
        best=chosen_chunk(t)
        lines.append(f'- {t["category"]} attempt {t["attempt"]}: {t["status"]}; best fixed Chunk has {best.get("distinct_states",0)} distinct states, cumulative target changes {best.get("cumulative_target_changes",0)}, changed positions {best.get("unique_changed_positions",0)}; ignition={t.get("ignition_mode","none")}.')
        if t.get('repair_reason'):lines.append('  Bounded repair: '+t['repair_reason'])
    if run.get('error'):lines+=['','Execution error: '+run['error']]
    lines+=['','## 200K feasibility','',stats['estimate_limitation'],
        'Each category retains its original quota. Missing production cannot be filled with other categories, duplicates, manual fluid sources or growth commands.',
        'Full-dataset world count and cost remain unmeasured unless every required category has demonstrated sustainable output.',
        '', '| Category | Target | Observed | Conditional world-equivalents | Conditional simulation hours |',
        '|---|---:|---:|---:|---:|']
    for category,e in estimates.items():
        worlds=str(e['linear_world_equivalents']) if e['linear_world_equivalents'] is not None else 'not estimable'
        hours=f"{e['linear_simulation_hours']:.2f}" if e['linear_simulation_hours'] is not None else 'not estimable'
        lines.append(f"| {category} | {e['target']} | {e['observed']} | {worlds} | {hours} |")
    lines += [
        '', '## Actual artifacts','',
        '- `snapshots.zarr/tokens`: uint16[N,16,16,16]; `light`: raw param1 uint8[N,16,16,16], auxiliary only.',
        '- `vocabulary.json`: append-only Node Name / raw param2 pairs; `metadata.parquet`: seed, fixed ENU origin, biome, stratum, category, event, true time/step and split.',
        '- `statistics.json`: complete observations, rejected states, yields and explicitly conditional linear estimates.',
        '- Diagnostic PNG/GIF/NPZ files are actual observed states, not generated illustrations or extra training snapshots.']
    for category,path in run.get('visualizations',{}).items():lines.append(f'- [{category} diagnostic]({path})')
    lines+=['','Voxel extraction uses native VoxelManip node/light/param2 arrays in the same Lua callback, cross-checked against get_node for all 4096 positions. The existing Craftium Python C++ channel supplies node/param2 only; no synthetic light or camera-derived Chunk origin is used.',
        'No engine fluids, ABM intervals, growth metadata or native physics were modified. Fire permits one explicitly recorded ignition per event, with a pre-ignition state.',
        'Earlier random-world candidate locations are reused as discovery hints; these are not an unbiased event-discovery survey. No global state deduplication across independent worlds/contexts.',
        '','## Decision','',
        ('All four categories demonstrated measured standard clips; user approval is still required before 200K.' if verdict=='GO' else 'P0 does not yet establish all four native dynamic collection paths. Stop here; no large collection is authorized.')]
    if run.get('initialization_failure'):lines+=['', 'Initial startup attempt: '+run['initialization_failure']]
    if run.get('qualification_note'):lines+=['', 'Offline qualification: '+run['qualification_note']]
    (Path(output.root)/'VAE_P0_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return stats
