"""CPU-only, read-only full48 token audit of the static/dynamic category packs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from .full48_source import iter_archive_shards, load_reference, validate_source_header
from .full48_scan import analyze_shard


REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "voxel_vae/reports"
SPLITS = ("train", "validation", "test")
CATEGORIES = ("Static", "Dynamic", "D1", "D2", "D3", "D4")
NVOX = 48 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frequency_groups(k):
    return {split: {category: dict(voxel_count=np.zeros(k, np.int64),
                                   sample_count=np.zeros(k, np.int64), snapshot_count=0)
                    for category in CATEGORIES} for split in SPLITS}


def token_rows(ids, pairs, **arrays):
    return [dict(canonical_token_id=int(i), node_name=pairs[i][0], raw_param2=pairs[i][1],
                 **{name: int(values[i]) for name, values in arrays.items()}) for i in ids]


def pooled_unseen(groups, unseen, pairs):
    present = {c: unseen & (sum((groups[s][c]["voxel_count"] for s in SPLITS),
                                np.zeros(len(pairs), np.int64)) > 0)
               for c in ("Static", "Dynamic")}
    union = present["Static"] | present["Dynamic"]
    return dict(static=int(present["Static"].sum()), dynamic=int(present["Dynamic"].sum()),
                ordinary_union=int(union.sum()), absent_from_all_ordinary=int((unseen & ~union).sum()),
                absent_tokens=token_rows(np.flatnonzero(unseen & ~union),pairs))


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output, pairs, unseen, groups, anchor, links):
    k = len(pairs)
    fields = ["split", "category", "canonical_token_id", "node_name", "raw_param2",
              "voxel_count", "voxel_probability", "sample_count", "sample_probability"]
    rows = []
    for split in SPLITS:
        for category in ("Static", "Dynamic"):
            group = groups[split][category]
            samples = group["snapshot_count"]
            for i in sorted(range(k), key=lambda i: (-int(group["voxel_count"][i]), i)):
                rows.append(dict(split=split, category=category, canonical_token_id=i,
                                 node_name=pairs[i][0], raw_param2=pairs[i][1],
                                 voxel_count=int(group["voxel_count"][i]),
                                 voxel_probability=int(group["voxel_count"][i]) / (samples*NVOX) if samples else 0.,
                                 sample_count=int(group["sample_count"][i]),
                                 sample_probability=int(group["sample_count"][i]) / samples if samples else 0.))
    write_csv(output/"static_dynamic_embedding_frequency.csv", fields, rows)

    fields = ["canonical_token_id", "node_name", "raw_param2", "is_center_train_unseen",
              "anchor_occurrence_count", "anchor_sample_count", "paired_chain_occurrence_count",
              "anchor_exclusive_occurrence_count", "anchor_exclusive_anchor_count",
              "is_anchor_exclusive", "anchor_ids", "chain_ids", "d4_subtypes", "anchor_chain_links"]
    rows = []
    for i in range(k):
        detail = links.get(i, [])
        rows.append(dict(canonical_token_id=i, node_name=pairs[i][0], raw_param2=pairs[i][1],
                         is_center_train_unseen=bool(unseen[i]),
                         anchor_occurrence_count=int(anchor["anchor_counts"][i]),
                         anchor_sample_count=int(anchor["anchor_sample_counts"][i]),
                         paired_chain_occurrence_count=int(anchor["paired_counts"][i]),
                         anchor_exclusive_occurrence_count=int(anchor["exclusive_counts"][i]),
                         anchor_exclusive_anchor_count=int(anchor["exclusive_anchor_counts"][i]),
                         is_anchor_exclusive=bool(anchor["exclusive_counts"][i]),
                         anchor_ids=json.dumps([r["anchor_id"] for r in detail]),
                         chain_ids=json.dumps([r["chain_id"] for r in detail]),
                         d4_subtypes=json.dumps(sorted({r["subtype"] for r in detail})),
                         anchor_chain_links=json.dumps(detail, ensure_ascii=False, separators=(",", ":"))))
    write_csv(output/"anchor_unseen_token_audit.csv", fields, rows)

    fields = ["canonical_token_id", "node_name", "raw_param2", "is_center_train_unseen",
              "in_U_static", "in_U_dynamic", "static_only_unseen", "dynamic_only_unseen"]
    for category in CATEGORIES:
        fields += [f"{category}_voxel_count", f"{category}_sample_count"]
    rows = []
    for i in range(k):
        static = bool(unseen[i] and groups["train"]["Static"]["voxel_count"][i])
        dynamic = bool(unseen[i] and groups["train"]["Dynamic"]["voxel_count"][i])
        row = dict(canonical_token_id=i, node_name=pairs[i][0], raw_param2=pairs[i][1],
                   is_center_train_unseen=bool(unseen[i]), in_U_static=static, in_U_dynamic=dynamic,
                   static_only_unseen=static and not dynamic, dynamic_only_unseen=dynamic and not static)
        for category in CATEGORIES:
            for name in ("voxel_count", "sample_count"):
                row[f"{category}_{name}"] = int(groups["train"][category][name][i])
        rows.append(row)
    write_csv(output/"static_dynamic_unseen_comparison.csv", fields, rows)


def render_report(summary):
    lines = ["# Full 48³ embedding occurrence and unseen-token audit", "", "PASS", "",
             "Counts describe token IDs / embedding rows present in data; no embedding values, model forward or training were computed.",
             "U0 remains the 274 tokens with zero occurrences in the original **center16³ train** baseline.", "",
             "## Source and validation", "",
             "Per the user's source clarification, this audit reads the complete local Static/Dynamic canonical repacks directly from tar.zst streams. No raw dataset was extracted to disk or modified.",
             "The original clip-local IDs and old dataset_index.json are not substituted or rebuilt. Current pack vocabularies are checked semantically against the frozen training vocabulary; original LUT availability is recorded below.", "",
             "```json", json.dumps(summary["validation"], ensure_ascii=False, indent=2), "```", "",
             "## 1. Full48 Static / Dynamic frequency", "",
             "| Split | Category | Snapshots (sample probability denominator) | Voxels (voxel probability denominator) | Unique tokens | U0 tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for category in ("Static", "Dynamic"):
            s = summary["groups"][split][category]
            lines.append(f"| {split} | {category} | {s['snapshots']:,} | {s['voxels']:,} | {s['unique_tokens']} | {s['untrained_unique_tokens']} |")
    lines += ["", "`static_dynamic_embedding_frequency.csv` contains all 1,580 tokens for each of the six groups (9,480 rows), sorted by voxel count descending within each group. Probabilities are fractions in [0,1], not percentages. Each of the five dynamic states counts as one snapshot; anchors are excluded.",
              "Voxel probabilities sum to one in each group. Sample probabilities are per-token presence probabilities and do not have to sum to one.",
              "", f"Pooling all splits for distribution only: U0 appears in {summary['unseen_full48_all_splits']['static']} Static token types and {summary['unseen_full48_all_splits']['dynamic']} Dynamic token types; their union covers {summary['unseen_full48_all_splits']['ordinary_union']} of 274. The remaining {summary['unseen_full48_all_splits']['absent_from_all_ordinary']} are absent from every ordinary full48 snapshot. This pooled result is not used in the Train-only comparison below.",
              "", "## 2. D4 anchors vs their own five-state chains", ""]
    a = summary["anchors"]
    lines += [f"- Anchors: **{a['total']}** (all splits).",
              f"- Anchors containing U0: **{a['containing_U0']}**.",
              f"- Unique U0 tokens observed in anchors: **{a['unseen_unique']}**.",
              f"- Anchor-exclusive unique U0 tokens: **{a['exclusive_unique']}**.",
              f"- Anchor-exclusive voxel occurrences: **{a['exclusive_voxel_occurrences']:,}**.", "",
              "Exclusivity is decided for each anchor and its own linked five full48 states first. A token may be exclusive in one anchor and shared in another. The exclusive occurrence total includes only the exclusive anchor-token pairs.",
              "The anchor CSV includes every canonical token. Detailed anchor/chain links concern U0 only; trained-token rows retain aggregate counts but no per-anchor unseen-audit links. Paired-chain occurrence counts are summed only over anchors that contain that token, once per anchor's own chain.", "",
              "Complete anchor-exclusive list:", "", "| Token | Node name | Raw param2 | Exclusive anchor voxels |", "|---:|---|---:|---:|"]
    for row in a["exclusive_tokens"]:
        lines.append(f"| {row['canonical_token_id']} | `{row['node_name']}` | {row['raw_param2']} | {row['exclusive_voxel_count']} |")
    if not a["exclusive_tokens"]:
        lines.append("| — | None | — | 0 |")
    u = summary["unseen_train_full48"]
    lines += ["", "## 3. Train full48 unseen-token sets", "",
              f"- U_static: **{u['static']}**; U_dynamic: **{u['dynamic']}**.",
              f"- Intersection: **{u['intersection']}**; Static-only: **{u['static_only']}**.",
              f"- **Dynamic-only Unseen Tokens: {u['dynamic_only']}**.",
              f"- U0 absent from both full48 train categories: **{u['absent_from_train_full48']}**.", "",
              "Only Train ordinary snapshots are used here. All set sizes are unique canonical tokens, with no rarity threshold. Category contributions may overlap and must not be added as disjoint sets.", "",
              "| Dynamic category | Dynamic-only unique tokens contributed |", "|---|---:|"]
    for category, value in u["dynamic_only_by_category"].items():
        lines.append(f"| {category} | {value['unique']} |")
    lines += ["", "Complete Dynamic-only list:", "", "| Token | Node name | Raw param2 | Static voxels | Dynamic voxels | D1 | D2 | D3 | D4 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in u["dynamic_only_tokens"]:
        lines.append(f"| {row['canonical_token_id']} | `{row['node_name']}` | {row['raw_param2']} | {row['static_voxel_count']} | {row['dynamic_voxel_count']} | {row['D1']} | {row['D2']} | {row['D3']} | {row['D4']} |")
    if not u["dynamic_only_tokens"]:
        lines.append("| — | None | — | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines += ["", f"Analysis wall time: {summary['wall_seconds']:.2f} seconds. This is a data-distribution audit, not evidence of training or generalization quality.",
              "", "## Reproduce", "",
              "Run from the VoxWM repository root (CPU Python with NumPy/Zarr/numcodecs and the `zstd` CLI). Output must be new/empty; raw archives are only read.",
              "", "```bash", ".venv/bin/python -B -m voxel_vae.full48_audit \\",
              *[f"  --archive '{path}' \\" for path in summary['source_archives']],
              "  --output outputs/voxel_vae/full48_embedding_audit_20261126_rerun", "```", ""]
    return "\n".join(lines)


def run(args):
    start = time.monotonic()
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), f"Output must be new/empty: {output}")
    require(len(args.archive)==2 and len({p.resolve() for p in args.archive})==2,
            "Provide exactly the two distinct Static and Dynamic archives")
    ref = load_reference(args.vocabulary, args.baseline, args.split)
    k, pairs, unseen = ref["K"], ref["pairs"], ref["unseen"]
    groups = frequency_groups(k)
    center = {s: np.zeros(k, np.int64) for s in SPLITS}
    anchor = {name: np.zeros(k, np.int64) for name in ["anchor_counts", "anchor_sample_counts", "paired_counts", "exclusive_counts", "exclusive_anchor_counts"]}
    links = defaultdict(list)
    validations, seeds, keys = [], {}, set()
    anchor_total = with_unseen = shard_count = sample_total = 0
    anchor_fields = dict(anchor_counts="anchor_voxel_count", anchor_sample_counts="anchor_sample_count",
                         paired_counts="paired_chain_voxel_count", exclusive_counts="exclusive_voxel_count",
                         exclusive_anchor_counts="exclusive_anchor_count")
    for archive in args.archive:
        header_seen = False
        for item in iter_archive_shards(archive):
            if not header_seen:
                validation = validate_source_header(item.header, ref)
                validations.append(validation["validation"])
                lut = validation["lut"]
                header_seen = True
            result = analyze_shard(item.files, item.root_name, item.shard_name, lut, k,
                                   ref["split_moves"], unseen, np.array([p[1] for p in pairs], np.uint8))
            require(all((r['category']=='Static') == (validation['category']=='Static')
                        for r in result['samples']), "Snapshot category differs from its category package")
            for s in SPLITS:
                center[s] += result["center_counts"][s]
                for category in CATEGORIES:
                    target, source = groups[s][category], result["frequency"][s][category]
                    target["snapshot_count"] += source["snapshot_count"]
                    for name in ("voxel_count", "sample_count"):
                        target[name] += source[name]
            for name in anchor:
                anchor[name] += result[anchor_fields[name]]
            for token, rows in result["anchor_links"].items():
                links[int(token)].extend(rows)
            anchor_total += result["anchor_total"]
            with_unseen += result["anchors_with_unseen"]
            for row in result["samples"]:
                original = row.get("repack_source")
                require(isinstance(original, dict) and all(k in original for k in ("clip","shard","snapshot_id")),
                        f"Missing original snapshot identity in {row['key']}")
                identity = (original["clip"], original["shard"], original["snapshot_id"])
                require(identity not in keys, f"Duplicate original snapshot: {identity}")
                keys.add(identity)
                seed, split = row["world_seed"], row["split"]
                require(seeds.setdefault(seed, split) == split, f"World seed {seed} crosses splits")
            sample_total += len(result["samples"])
            shard_count += 1
            if shard_count % 10 == 0:
                print(f"Scanned {shard_count} shards / {sample_total:,} snapshots / {anchor_total:,} anchors; {time.monotonic()-start:.1f}s", flush=True)
        require(header_seen, f"No committed shards in {archive}")
        validations[-1]["dataset_index_search"] = item.header.get("index_search", {})
    expected = {"train": 160000, "validation": 20000, "test": 20000}
    summary_groups = {s: {} for s in SPLITS}
    for s in SPLITS:
        require(sum(groups[s][c]["snapshot_count"] for c in ("Static", "Dynamic")) == expected[s], f"Wrong {s} snapshot count")
        require(np.array_equal(center[s], ref["baseline"][s]), f"Recomputed center16 {s} counts differ from frozen baseline")
        for category in CATEGORIES:
            group = groups[s][category]
            voxels = int(group["voxel_count"].sum())
            require(voxels == group["snapshot_count"] * NVOX, f"Occurrence sum mismatch: {s}/{category}")
            require(np.all(group["sample_count"] <= group["snapshot_count"]), "Sample count exceeds denominator")
            summary_groups[s][category] = dict(snapshots=group["snapshot_count"], voxels=voxels,
                                              unique_tokens=int(np.count_nonzero(group["voxel_count"])),
                                              untrained_unique_tokens=int(np.count_nonzero(unseen & (group["voxel_count"]>0))))
        require(np.array_equal(groups[s]["Dynamic"]["voxel_count"], sum((groups[s][c]["voxel_count"] for c in CATEGORIES[2:]), np.zeros(k,np.int64))), "Dynamic category sum mismatch")
    require(sample_total == 200000 and shard_count == 400, "Expected two category roots, 400 shards / 200K snapshots")
    require(anchor_total == 5000, "Expected all 5,000 D4 anchors")
    require(sum(groups[s]["Static"]["snapshot_count"] for s in SPLITS) == 100000, "Expected 100K Static")
    for category in CATEGORIES[2:]:
        require(sum(groups[s][category]["snapshot_count"] for s in SPLITS) == 25000, f"Expected 25K {category}")
    us = unseen & (groups["train"]["Static"]["voxel_count"] > 0)
    ud = unseen & (groups["train"]["Dynamic"]["voxel_count"] > 0)
    only = ud & ~us
    exclusive = unseen & (anchor["exclusive_counts"] > 0)
    summary = dict(status="PASS", source_archives=[str(p.resolve()) for p in args.archive],
                   source_selection="user-authorized canonical Static/Dynamic repacks; original clip roots superseded for this analysis",
                   shape=[48,48,48], total_snapshots=sample_total, total_full48_voxels=sample_total*NVOX,
                   groups=summary_groups, unseen_full48_all_splits=pooled_unseen(groups,unseen,pairs),
                   validation=dict(reference=ref["validation"], sources=validations,
                       actual_split_counts=expected, center16_all_token_counts_match_baseline=True,
                       all_snapshot_shapes_and_token_ranges_pass=True, all_chain_anchor_links_pass=True,
                       all_occurrence_denominators_pass=True, original_source_snapshots_unique=True,
                       world_seed_split_consistency_pass=True, old_dataset_index_rebuilt=False),
                   anchors=dict(total=anchor_total, containing_U0=with_unseen,
                       unseen_unique=int(np.count_nonzero(unseen & (anchor["anchor_counts"]>0))),
                       exclusive_unique=int(np.count_nonzero(exclusive)),
                       exclusive_voxel_occurrences=int(anchor["exclusive_counts"].sum()),
                       exclusive_tokens=token_rows(np.flatnonzero(exclusive),pairs,exclusive_voxel_count=anchor["exclusive_counts"])),
                   unseen_train_full48=dict(static=int(us.sum()),dynamic=int(ud.sum()),intersection=int((us&ud).sum()),
                       static_only=int((us&~ud).sum()),dynamic_only=int(only.sum()),absent_from_train_full48=int((unseen&~(us|ud)).sum()),
                       dynamic_only_tokens=token_rows(np.flatnonzero(only),pairs,
                           static_voxel_count=groups["train"]["Static"]["voxel_count"],
                           dynamic_voxel_count=groups["train"]["Dynamic"]["voxel_count"],
                           **{c:groups["train"][c]["voxel_count"] for c in CATEGORIES[2:]}),
                       dynamic_only_by_category={c:dict(unique=int((only&(groups["train"][c]["voxel_count"]>0)).sum()),
                           token_ids=np.flatnonzero(only&(groups["train"][c]["voxel_count"]>0)).tolist()) for c in CATEGORIES[2:]}),
                   wall_seconds=time.monotonic()-start)
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output,pairs,unseen,groups,anchor,links)
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (output/"report.md").write_text(render_report(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--vocabulary", type=Path, default=REPORTS/"e01_masked_20261122/training_vocabulary.json")
    parser.add_argument("--baseline", type=Path, default=REPORTS/"e01_freeze_audit_20261122/token_frequency.csv")
    parser.add_argument("--split", type=Path, default=REPORTS/"e01_masked_20261122/split_160k.json")
    parser.add_argument("--output", type=Path, default=REPO/"outputs/voxel_vae/full48_embedding_audit_20261126")
    args=parser.parse_args(argv)
    try:
        summary=run(args)
    except Exception as exc:
        print(f"FAIL: {exc}",file=sys.stderr)
        if not args.output.exists() or not any(args.output.iterdir()):
            args.output.mkdir(parents=True,exist_ok=True)
            failure=dict(status="FAIL",error=f"{type(exc).__name__}: {exc}")
            (args.output/"summary.json").write_text(json.dumps(failure,ensure_ascii=False,indent=2)+"\n")
            (args.output/"report.md").write_text(f"FAIL\n\n{exc}\nNo data was repaired or changed.\n")
        return 1
    print(json.dumps(dict(status=summary["status"],anchor_exclusive=summary["anchors"]["exclusive_unique"],
                          dynamic_only_unseen=summary["unseen_train_full48"]["dynamic_only"],
                          wall_seconds=summary["wall_seconds"]),ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
