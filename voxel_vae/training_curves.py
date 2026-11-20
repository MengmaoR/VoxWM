"""Dependency-free SVG/JSON loss-curve snapshots from Masked metrics.jsonl."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


RECON_SERIES = (
    ("loss_reconstruction", "Reconstruction", "#70b7ff"),
    ("loss_static", "Static", "#6ee7a0"),
    ("loss_dynamic_unchanged", "Unchanged", "#f4c95d"),
    ("loss_dynamic_changed", "Changed", "#ff6b8a"),
)
LOSS_KEYS = ("loss_total", *(key for key, _, _ in RECON_SERIES),
             "loss_beta_kl", "loss_kl_raw")
CATEGORIES = ("D1", "D2", "D3", "D4")
CHANGED_SERIES = tuple((f"changed_{category}", category, color)
                       for category, color in zip(
                           CATEGORIES, ("#5cc8ff", "#65e6a5", "#ffc857", "#ff7b9c")))


def read_metric_rows(metrics_path):
    """Replay the active history while preserving raw logs from interrupted work."""
    rows = {}
    try:
        stream = Path(metrics_path).open(encoding="utf-8")
    except FileNotFoundError:
        return []
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # The last line can be incomplete while training appends it.
            step, event = row["global_step"], row.get("event")
            if event == "resume":
                rows = {key: item for key, item in rows.items() if key[0] <= step}
            else:
                rows[step, event] = row
    return sorted(rows.values(), key=lambda row: row["global_step"])


def read_curve_data(metrics_path):
    train, validation = [], []
    for row in read_metric_rows(metrics_path):
        event = row.get("event")
        if event == "validation":
            components = row.get("loss_components") or {}
        elif event is None and "loss_total" in row:
            components = row
        else:
            continue
        if components.get("loss_total") is None:
            continue
        point = {"step": row["global_step"], **{key: components.get(key) for key in LOSS_KEYS}}
        for kind in ("changed", "unchanged"):
            values = components.get(f"dynamic_{kind}_by_category") or {}
            point.update({f"{kind}_{category}": values.get(category) for category in CATEGORIES})
        if event == "validation":
            reconstruction = row.get("dynamic_reconstruction") or {}
            point["static_accuracy"] = (row.get("static_preservation") or {}).get(
                "exact_token_accuracy")
            point["changed_accuracy"] = reconstruction.get("changed_voxel_exact_accuracy")
            point["unchanged_accuracy"] = reconstruction.get("unchanged_voxel_exact_accuracy")
            validation.append(point)
        else:
            train.append(point)
    return {"train": train, "validation": validation}


def _panel(series, title, left, top, width, x_max):
    height = 138
    points = [([(row["step"], row[key]) for row in rows
                if row.get(key) is not None and math.isfinite(row[key])], label, color)
              for rows, key, label, color in series]
    values = [value for curve, _, _ in points for _, value in curve]
    y_max = max(max(values, default=1.0) * 1.05, 1e-12)
    lines = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#111a24" stroke="#405064"/>',
        f'<text x="{left}" y="{top - 14}" fill="#dce8f1" font-size="17">{title}</text>',
        f'<text x="{left}" y="{top + height + 17}">0</text>',
        f'<text x="{left + width}" y="{top + height + 17}" text-anchor="end">step {x_max}</text>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        y = top + height * (1 - fraction)
        lines.extend((
            f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#263647"/>',
            f'<text x="{left - 7}" y="{y + 4}" text-anchor="end">{y_max * fraction:.3g}</text>',
        ))
    legend_x = left
    for curve, label, color in points:
        xy = [(left + width * step / x_max, top + height * (1 - value / y_max))
              for step, value in curve]
        if len(xy) > 1:
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in xy)
            lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        if xy:
            x, y = xy[-1]
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        lines.append(f'<text x="{legend_x}" y="{top + height + 37}" fill="{color}">{label}</text>')
        legend_x += 24 + len(label) * 7
    if not values:
        lines.append(f'<text x="{left + 12}" y="{top + 24}">Waiting for metrics</text>')
    return "\n".join(lines)


def render_svg(data):
    train, validation = data["train"], data["validation"]
    x_max = max(1, max((row["step"] for row in train + validation), default=0))

    def paired(key):
        return [(train, key, "Training batch", "#70b7ff"),
                (validation, key, "Validation", "#ff9d7a")]

    panels = [_panel(paired("loss_total"), "Total loss", 68, 105, 1104, x_max)]
    for label, rows, left in (("Training", train, 68), ("Validation", validation, 643)):
        panels.append(_panel([(rows, *line) for line in RECON_SERIES],
                             f"{label} reconstruction losses", left, 322, 529, x_max))
        panels.append(_panel([(rows, *line) for line in CHANGED_SERIES],
                             f"{label} Dynamic-Changed CE by category", left, 539, 529, x_max))
    panels += [_panel(paired("loss_beta_kl"), "KL contribution (beta × KL)", 68, 756, 529, x_max),
               _panel(paired("loss_kl_raw"), "Raw KL", 643, 756, 529, x_max)]
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 960" '
        'font-family="system-ui, sans-serif" font-size="12" fill="#91aabd">',
        '<rect width="1200" height="960" fill="#0d141d"/>',
        '<text x="68" y="42" fill="#f0f5f8" font-size="24">VoxWM E00-Masked loss curves</text>',
        *panels,
        '</svg>',
    ])


def write_training_curves(metrics_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = read_curve_data(metrics_path)
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    json_path = output / "loss_curves.json"
    json_tmp = json_path.with_name("." + json_path.name + ".tmp")
    json_tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    json_tmp.replace(json_path)
    svg_path = output / "loss_curves.svg"
    svg_tmp = svg_path.with_name("." + svg_path.name + ".tmp")
    svg_tmp.write_text(render_svg(data))
    svg_tmp.replace(svg_path)
    return svg_path, json_path
