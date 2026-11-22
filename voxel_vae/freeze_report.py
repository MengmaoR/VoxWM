"""Render the freeze decision from the actual held-out and train-set audit outputs."""
from __future__ import annotations

import json


def _pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _number(value):
    return "N/A" if value is None else f"{value:.6g}"


def _table(header, rows):
    return ["| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def _transition_table(rows):
    return _table(
        ("GT t", "GT t+1", "Voxels", "Pair exact", "Node pair exact", "Pair errors"),
        [(f"`{r['gt_before_node_name']}` ({r['gt_before_param2']})",
          f"`{r['gt_after_node_name']}` ({r['gt_after_param2']})", r["count"],
          _pct(r["pair_exact_reconstruction_rate"]), _pct(r["node_name_pair_exact_rate"]),
          r["pair_exact_errors"]) for r in rows[:5]])


def render_report(output):
    def read(name):
        return json.loads((output / f"{name}.json").read_text())

    provenance, split, test = map(read, ("checkpoint_provenance", "split_audit", "test_metrics"))
    d3, spatial, embedding = map(read, ("d3_error_summary", "d3_spatial_errors", "embedding_summary"))
    latent, norm = map(read, ("latent_statistics", "latent_normalization"))
    if (not all(d["checkpoint_step"] == provenance["checkpoint_step"] for d in (test, embedding, latent, norm))
            or test["samples"] != split["split_counts"]["test"]
            or latent["samples"] != split["split_counts"]["train"]
            or norm["samples"] != latent["samples"] or norm["source_split"] != "train"
            or norm["checkpoint_sha256"] != provenance["checkpoint_sha256"]):
        raise ValueError("Freeze report inputs disagree on the audited samples or checkpoint")

    static, dynamic = test["static_preservation"], test["dynamic_reconstruction"]
    pairs, test_latent, loss = test["dynamic"], test["latent"], test["loss_components"]
    adjacent = test_latent["adjacent_dynamic_latent_l2"]
    channels = provenance["latent_shape"][0]
    expected_reconstruction = [
        ("Static exact", static["exact_token_accuracy"], 0.95),
        ("Static non-air exact", static["non_air_exact_token_accuracy"], 0.90),
        ("Changed exact", dynamic["changed_voxel_exact_accuracy"], 0.80),
        ("Predicted change recall", pairs["predicted_change_recall"], 0.90),
        ("Unchanged stability", pairs["unchanged_stability"], 0.97),
    ]
    reconstruction_pass = all(actual is not None and actual >= minimum
                              for _, actual, minimum in expected_reconstruction)
    latent_pass = (latent["active_latent_channels"] == channels and adjacent["count"] > 0
                   and adjacent["zero_count"] == 0
                   and min(test_latent["latent_variance_per_channel"]) >= 1e-6)
    embedding_pass = not embedding["seen_but_no_drift"]
    node_limited = d3["node_error_count"] > 0 and d3["node_error_count"] >= d3["param2_only_error_count"]
    limitation = ("D3 structural reconstruction remains a material limitation." if node_limited else
                  "D3 errors are predominantly fine-state errors among the reconstructed changed voxels.")
    if not (reconstruction_pass and latent_pass and embedding_pass):
        decision = "DO NOT FREEZE"
    elif d3["combined_token_accuracy"] < 0.80:
        decision = "FREEZE WITH KNOWN D3 LIMITATION"
    else:
        decision = "FREEZE"

    overall = [("Cross entropy", test["cross_entropy"]), ("Masked total", loss["loss_total"]),
               ("Static CE", loss["loss_static"]), ("Dynamic unchanged CE", loss["loss_dynamic_unchanged"]),
               ("Dynamic changed CE", loss["loss_dynamic_changed"]),
               ("Raw KL", loss["loss_kl_raw"]), ("Weighted KL", loss["loss_beta_kl"])]
    reconstruction = [("Dynamic exact", dynamic["exact_token_accuracy"]),
                      ("Changed exact", dynamic["changed_voxel_exact_accuracy"]),
                      ("Unchanged exact", dynamic["unchanged_voxel_exact_accuracy"]),
                      ("Predicted change recall", pairs["predicted_change_recall"]),
                      ("Changed-pair exact", pairs["changed_pair_exact_reconstruction"]),
                      ("Unchanged stability", pairs["unchanged_stability"])]
    category_rows = []
    for name, values in test["dynamic_by_category"].items():
        category_rows.append((name, values["changed_voxels"], *[_pct(values[k]) for k in (
            "exact_token_accuracy", "non_air_exact_token_accuracy", "changed_voxel_exact_accuracy",
            "unchanged_voxel_exact_accuracy", "predicted_change_recall",
            "changed_pair_exact_reconstruction", "unchanged_stability")],
            _number(values["changed_ce"]), _number(values["unchanged_ce"])))
    exact, param2, node = [d3[k] for k in ("exact_correct_count", "param2_only_error_count", "node_error_count")]
    total = d3["total_changed_voxels"]
    outcomes = [("Exact", exact), ("Param2 only", param2), ("Node error", node)]
    boundary = spatial["marginals"]["distance_to_boundary"]
    spatial_rows = [
        ("Boundary", boundary["errors"][0], boundary["changed_support"][0]),
        ("Interior", sum(boundary["errors"][1:]), sum(boundary["changed_support"][1:])),
        ("Isolated", spatial["isolated"]["errors"], spatial["isolated"]["changed_support"]),
        ("Non-isolated", spatial["non_isolated"]["errors"], spatial["non_isolated"]["changed_support"]),
    ]
    distances = [f"{stat} {_number(adjacent[stat])}" for stat in ("mean", "std", "min", "max")]
    layers = [("Global", latent)] + list(latent["by_category"].items())
    embedding_finding = (
        "No embedding optimization failure detected. Non-drifting rows correspond to unseen input tokens."
        if embedding_pass else
        f"Flagged {len(embedding['seen_but_no_drift']):,} seen tokens with low drift. "
        "Review their input support and row updates before freezing.")
    lines = [
        "# VoxWM VAE Freeze Audit", "",
        "## 1. Checkpoint Provenance", "",
        f"`{provenance['checkpoint']}` at step {provenance['checkpoint_step']:,} was selected on "
        f"validation total loss ({provenance['selected_validation_total_loss']:.10f}).",
        f"Training source commit `{provenance['source_commit']}`.",
        f"Audit source commit `{provenance['audit_source_commit']}`.",
        f"Checkpoint SHA256 `{provenance['checkpoint_sha256']}`.", "",
        "## 2. Split Integrity", "",
        *_table(("Partition", "States", "World seeds"), [
            (name, count, split["seed_counts"][name]) for name, count in split["split_counts"].items()]),
        "",
        "World-seed intersections are " + ", ".join(
            f"{name} = {count}" for name, count in split["seed_intersections"].items()) + ".",
        f"Test covers all {test['samples']:,} states and {pairs['complete_chains']:,} complete dynamic "
        f"chains, with {split['d4_anchor_samples']} D4 anchor samples.", "",
        "## 3. Full Held-Out Test Results", "",
        "The 20,000-state test was a confirmatory posterior-mean reconstruction evaluation of the "
        "validation-selected 27k model.",
        f"Exact Combined-Token accuracy **{_pct(test['exact_token_accuracy'])}**. "
        f"Non-air exact accuracy **{_pct(test['non_air_exact_token_accuracy'])}**.", "",
        *_table(("Metric", "Value"), [(k, _number(v)) for k, v in overall]), "",
        "## 4. Static Reconstruction", "",
        f"Static exact token accuracy **{_pct(static['exact_token_accuracy'])}**. "
        f"Non-air accuracy **{_pct(static['non_air_exact_token_accuracy'])}**.",
        "Biome and stratum results are in [test_metrics.json](test_metrics.json).", "",
        "## 5. Dynamic Reconstruction", "",
        *_table(("Metric", "Test"), [(k, _pct(v)) for k, v in reconstruction]),
        "", "Changed-voxel metrics use the union of adjacent changes at each state's position. "
        "Pair metrics use changes between two adjacent states.", "",
        "## 6. D1 / D2 / D3 / D4", "",
        *_table(("Category", "Changed voxels", "Exact", "Non-air", "Changed exact", "Unchanged exact",
                 "Change recall", "Pair exact", "Stability", "Changed CE", "Unchanged CE"), category_rows),
        "", "## 7. D3 Fire Error Decomposition", "",
        *_table(("Changed outcome", "Voxels", "Share"), [(k, n, _pct(n / total)) for k, n in outcomes]),
        "",
        f"Node-name accuracy {_pct(d3['node_name_accuracy'])}. "
        f"Param2 accuracy given a correct node {_pct(d3['param2_accuracy_given_node_correct'])} "
        f"({exact:,}/{d3['param2_accuracy_given_node_correct_denominator']:,}).",
        f"Fire presence recall {_pct(d3['fire_presence']['recall'])}, with "
        f"{d3['fire_presence']['gt_fire']:,} Changed positions containing fire in the ground truth.", "",
        f"**{limitation}**", "",
        f"The adjacent-pair analysis covers {d3['distinct_transitions']:,} distinct semantic transitions. "
        f"{d3['supported_transitions']:,} meet the support threshold of "
        f"{d3['transition_support_threshold']} voxels.", "",
        "### Most frequent transitions", "", *_transition_table(d3["top_transitions_by_frequency"]),
        "", "### Largest contributions to pair errors", "",
        *_transition_table(d3["top_transitions_by_error_contribution"]), "",
        "### Lowest pair accuracy among sufficiently supported transitions", "",
        *(_transition_table(d3["worst_supported_transitions"]) if d3["worst_supported_transitions"]
          else ["The observed transition counts are available in the CSV."]), "",
        "### Spatial distribution", "",
        *_table(("D3 Changed location", "Errors", "Changed sites", "Error rate"),
                [(name, errors, support, _pct(errors / support if support else None))
                 for name, errors, support in spatial_rows]),
        "",
        "Boundary sites touch a face of the center crop. Isolated sites have no face-adjacent "
        "Changed voxel in the same state. Axis and distance histograms are in "
        "[d3_spatial_errors.json](d3_spatial_errors.json).", "",
        "## 8. Embedding Coverage Audit", "",
        f"{embedding['seen_in_train']:,} tokens occur in train. "
        f"{embedding['unseen_in_train_but_seen_in_held_out']:,} occur only in validation/test. "
        f"{embedding['never_in_any_center_split']:,} are absent from all center crops.",
        f"{embedding['non_drifting_rows']:,} embedding rows have L2 drift at most `1e-6`.",
        f"Spearman correlation between frequency and drift is "
        f"{_number(embedding['spearman_log1p_train_frequency_vs_drift'])}.",
        embedding_finding, "",
        *_table(("Train frequency", "Tokens", "Median L2", "Mean L2"),
                [(name, row["token_count"], _number(row["median_drift"]), _number(row["mean_drift"]))
                 for name, row in embedding["frequency_bins"].items()]),
        "", "## 9. Latent Distribution", "",
        f"The train set has {latent['active_latent_channels']}/{channels} active channels at "
        f"the standard-deviation threshold `1e-3`. Test has "
        f"{test_latent['active_latent_channels']}/{channels} active channels at the E01 variance threshold "
        f"`1e-4`.",
        f"Test posterior mean has mean {_number(test_latent['mean_mu'])} and std "
        f"{_number(test_latent['std_mu'])}. Mean posterior std is "
        f"{_number(test_latent['mean_posterior_std'])}.",
        f"Adjacent dynamic latent L2 over {adjacent['count']:,} test pairs has " + ", ".join(distances)
        + f", with {adjacent['zero_count']} exact collisions.", "",
        *_table(("Train group", "States", "Global mean", "Global std", "Minimum", "Maximum"),
                [(name, row["samples"], _number(row["global_mean"]), _number(row["global_std"]),
                  _number(row["global_min"]), _number(row["global_max"])) for name, row in layers]),
        "", "## 10. Downstream Normalization", "",
        f"[latent_normalization.json](latent_normalization.json) contains a single per-channel mean "
        f"and population std from {norm['samples']:,} train posterior means. Downstream volumes use "
        f"`(mu - mean) / (std + 1e-6)` and invert with the same channel statistics.", "",
        "## 11. Remaining Limitations", "",
        "Reconstruction scores assess the current voxel representation. The dynamics and generation "
        "models require their own evaluation.",
        "The semantic decomposition uses hard-token predictions. The reported fire errors reflect "
        "the tested 16³ crops and the fixed held-out split.", "",
        "## 12. Freeze Decision", "", f"**{decision}**", "",
        *_table(("Criterion", "Observed", "Target", "Status"),
                [(name, _pct(value), _pct(target), "PASS" if value is not None and value >= target else "FAIL")
                 for name, value, target in expected_reconstruction] + [
                    ("Latent", f"{latent['active_latent_channels']}/{channels} train channels, "
                     f"{adjacent['zero_count']} zero-L2 pairs",
                     "Active channels and distinguishable adjacent pairs", "PASS" if latent_pass else "FAIL"),
                    ("Embedding", f"{len(embedding['seen_but_no_drift'])} seen low-drift rows",
                     "Meaningful seen-row updates", "PASS" if embedding_pass else "FAIL"),
                ]),
        "",
    ]
    if decision == "FREEZE WITH KNOWN D3 LIMITATION" and not node_limited:
        lines.extend(["VAE usable with known D3 fine-state limitation.", ""])
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
