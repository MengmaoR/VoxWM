"""Validation-loss early stopping state for Masked VAE training."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .training_curves import read_metric_rows


def validate_early_stopping_config(config):
    value = {
        "enabled": bool(config.get("enabled", True)),
        "monitor": str(config.get("monitor", "validation_loss_total")),
        "mode": str(config.get("mode", "min")),
        "patience_validations": int(config.get("patience_validations", 2)),
        "min_delta": float(config.get("min_delta", 0.001)),
        "min_steps": int(config.get("min_steps", 500)),
    }
    if value["monitor"] != "validation_loss_total" or value["mode"] != "min":
        raise ValueError("Masked early stopping monitors validation_loss_total in min mode")
    if value["patience_validations"] < 1:
        raise ValueError("patience_validations must be positive")
    if (not math.isfinite(value["min_delta"]) or value["min_delta"] < 0
            or value["min_steps"] < 0):
        raise ValueError("min_delta/min_steps must be finite and non-negative")
    return value


@dataclass
class EarlyStoppingState:
    best_metric: float | None = None
    best_step: int | None = None
    patience_reference: float | None = None
    bad_validations: int = 0
    validations: int = 0
    stopped: bool = False
    stop_step: int | None = None
    stop_reason: str | None = None
    last_metric: float | None = None

    def observe(self, step, metric, config):
        step, metric = int(step), float(metric)
        if not math.isfinite(metric):
            raise FloatingPointError("Non-finite validation loss for early stopping")
        new_best = self.best_metric is None or metric < self.best_metric
        if new_best:
            self.best_metric = metric
            self.best_step = step
        if (self.patience_reference is None
                or metric < self.patience_reference - config["min_delta"]):
            self.patience_reference = metric
            self.bad_validations = 0
        elif step >= int(config["min_steps"]):
            self.bad_validations += 1
        self.validations += 1
        self.last_metric = metric
        if (config["enabled"] and step >= int(config["min_steps"])
                and self.bad_validations >= int(config["patience_validations"])):
            self.stopped = True
            self.stop_step = step
            self.stop_reason = (
                f"validation_loss_total did not improve by {config['min_delta']} "
                f"for {self.bad_validations} validation checks")
        return new_best, self.stopped

    def to_dict(self, config):
        return {**asdict(self), "config": dict(config)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: item for key, item in value.items() if key != "config"})


def state_from_metrics(metrics_path, config, maximum_step=None):
    state = EarlyStoppingState()
    for row in read_metric_rows(metrics_path):
        step = row["global_step"]
        if row.get("event") != "validation" or (maximum_step is not None
                                                  and step > maximum_step):
            continue
        components = row.get("loss_components") or {}
        if components.get("loss_total") is not None:
            _, stopped = state.observe(step, components["loss_total"], config)
            if stopped:
                break
    return state
