"""Semantic and spatial accounting on held-out D3 states and adjacent pairs."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .evaluate import _ratio


CONFUSION_FIELDS = ("gt_node_name", "gt_param2", "pred_node_name", "pred_param2",
                    "count", "fraction_of_d3_changed_errors")
TRANSITION_FIELDS = (
    "gt_before_node_name", "gt_before_param2", "gt_after_node_name", "gt_after_param2",
    "count", "state_t_exact_reconstruction_rate", "state_t1_exact_reconstruction_rate",
    "pair_exact_reconstruction_rate", "node_name_pair_exact_rate", "pair_exact_errors",
)


class D3Audit:
    def __init__(self, vocabulary):
        self.tokens = tuple(vocabulary.pairs)
        self.names, self.node_ids = np.unique([name for name, _ in self.tokens], return_inverse=True)
        self.fire_ids = np.isin(self.names, ("mcl_fire:fire", "mcl_fire:eternal_fire"))
        self.counts = Counter()
        self.token_errors = Counter()
        self.node_errors = Counter()
        self.transitions = defaultdict(lambda: np.zeros(5, dtype=np.int64))
        xyz = np.indices((16, 16, 16))
        self.features = {
            "x": xyz[0], "y": xyz[1], "z": xyz[2],
            "distance_to_center_floor": np.sqrt(((xyz - 7.5) ** 2).sum(axis=0)).astype(int),
            "distance_to_boundary": np.minimum(xyz, 15 - xyz).min(axis=0),
        }
        self.support = {key: np.zeros(int(field.max()) + 1, dtype=np.int64)
                        for key, field in self.features.items()}
        self.spatial_errors = {key: np.zeros_like(value) for key, value in self.support.items()}

    def observe_states(self, target, predicted, mask):
        coordinate = np.nonzero(mask)
        gt, pr = target[coordinate], predicted[coordinate]
        gt_node, pr_node = self.node_ids[gt], self.node_ids[pr]
        exact = gt == pr
        same_node = gt_node == pr_node
        fire_gt, fire_pr = self.fire_ids[gt_node], self.fire_ids[pr_node]
        self.counts.update(
            total_changed_voxels=len(gt), exact_correct_count=int(exact.sum()),
            param2_only_error_count=int((same_node & ~exact).sum()), node_error_count=int((~same_node).sum()),
            gt_fire=int(fire_gt.sum()), predicted_fire=int(fire_pr.sum()),
            fire_presence_match=int((fire_gt & fire_pr).sum()),
            fire_node_match=int((fire_gt & same_node).sum()),
            fire_exact_match=int((fire_gt & exact).sum()))
        self.token_errors.update(zip(gt[~exact].tolist(), pr[~exact].tolist()))
        self.node_errors.update(zip(gt_node[~same_node].tolist(), pr_node[~same_node].tolist()))

        for key, field in self.features.items():
            positions = field[coordinate[-3:]]
            size = len(self.support[key])
            self.support[key] += np.bincount(positions, minlength=size)
            self.spatial_errors[key] += np.bincount(positions[~exact], minlength=size)

        # A changed site is isolated when no face neighbor changed in its frame.
        neighbors = np.zeros_like(mask)
        for axis in range(1, 4):
            lower, upper = [slice(None)] * 4, [slice(None)] * 4
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            lower, upper = tuple(lower), tuple(upper)
            neighbors[lower] |= mask[upper]
            neighbors[upper] |= mask[lower]
        isolated = ~neighbors[coordinate]
        self.counts.update(isolated_changed=int(isolated.sum()),
                           isolated_errors=int((isolated & ~exact).sum()))

    def observe_transition(self, before, after, predicted_before, predicted_after):
        changed = before != after
        gt0, gt1 = before[changed], after[changed]
        pr0, pr1 = predicted_before[changed], predicted_after[changed]
        correct0, correct1 = gt0 == pr0, gt1 == pr1
        correct_nodes = ((self.node_ids[gt0] == self.node_ids[pr0])
                         & (self.node_ids[gt1] == self.node_ids[pr1]))
        outcomes = np.stack([np.ones_like(correct0), correct0, correct1,
                             correct0 & correct1, correct_nodes], axis=1)
        for a, b, outcome in zip(gt0.tolist(), gt1.tolist(), outcomes):
            self.transitions[a, b] += outcome

    def confusion_rows(self, *, node_only=False):
        counts = self.node_errors if node_only else self.token_errors
        all_errors = self.counts["param2_only_error_count"] + self.counts["node_error_count"]
        rows = []
        for (gt, pr), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:50]:
            name_gt, param_gt = (str(self.names[gt]), "") if node_only else self.tokens[gt]
            name_pr, param_pr = (str(self.names[pr]), "") if node_only else self.tokens[pr]
            rows.append(dict(zip(CONFUSION_FIELDS, (
                name_gt, param_gt, name_pr, param_pr, count, _ratio(count, all_errors)))))
        return rows

    def transition_rows(self):
        rows = []
        for (a, b), counts in sorted(self.transitions.items(), key=lambda item: (-item[1][0], item[0])):
            count, first, second, pair, nodes = map(int, counts)
            rows.append(dict(zip(TRANSITION_FIELDS, (
                *self.tokens[a], *self.tokens[b], count, first / count, second / count,
                pair / count, nodes / count, count - pair))))
        return rows

    def summary(self, transitions):
        c = self.counts
        total = c["total_changed_voxels"]
        exact, param2, node = (c["exact_correct_count"], c["param2_only_error_count"],
                               c["node_error_count"])
        if exact + param2 + node != total:
            raise ValueError("D3 outcomes do not account for all changed voxels")
        supported = [r for r in transitions if r["count"] >= 20]
        return {
            "total_changed_voxels": total, "exact_correct_count": exact,
            "param2_only_error_count": param2, "node_error_count": node,
            "combined_token_accuracy": _ratio(exact, total),
            "node_name_accuracy": _ratio(exact + param2, total),
            "param2_only_error_fraction": _ratio(param2, total),
            "node_error_fraction": _ratio(node, total),
            "param2_accuracy_given_node_correct": _ratio(exact, exact + param2),
            "param2_accuracy_given_node_correct_denominator": exact + param2,
            "fire_presence": {**{key: c[key] for key in (
                "gt_fire", "predicted_fire", "fire_presence_match", "fire_node_match", "fire_exact_match")},
                "recall": _ratio(c["fire_presence_match"], c["gt_fire"]),
                "precision": _ratio(c["fire_presence_match"], c["predicted_fire"])},
            "transition_voxels": sum(r["count"] for r in transitions),
            "distinct_transitions": len(transitions),
            "transition_support_threshold": 20, "supported_transitions": len(supported),
            "top_transitions_by_frequency": transitions[:10],
            "top_transitions_by_error_contribution": sorted(
                (r for r in transitions if r["pair_exact_errors"]),
                key=lambda r: -r["pair_exact_errors"])[:10],
            "worst_supported_transitions": sorted(
                supported, key=lambda r: (r["pair_exact_reconstruction_rate"], -r["count"]))[:10],
        }

    def spatial_summary(self):
        c = self.counts
        return {
            "axis_order": "x,y,z on the center 16³ array",
            "center": [7.5, 7.5, 7.5],
            "distance_bins": "floor of Euclidean distance to center and voxel steps to nearest boundary",
            "marginals": {k: {"changed_support": self.support[k].tolist(),
                              "errors": self.spatial_errors[k].tolist()}
                          for k in self.features},
            "isolated": {"changed_support": c["isolated_changed"], "errors": c["isolated_errors"]},
            "non_isolated": {
                "changed_support": c["total_changed_voxels"] - c["isolated_changed"],
                "errors": c["node_error_count"] + c["param2_only_error_count"] - c["isolated_errors"],
            },
        }
