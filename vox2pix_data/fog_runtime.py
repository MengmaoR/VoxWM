"""The final native fog state, expressed in block units."""

import json
import math


def read_fog(native):
    path = native.session.world / "client_clock.txt.fog"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Missing native fog frame probe at {path}") from exc


def validate_fog(probe, *, enabled=True, ratio=.6666667, tick=None):
    """Require the final driver and node-shader state for this exposure."""
    if probe.get("version") not in ("voxwm-native-fog-v1", "voxwm-native-fog-v2"):
        raise ValueError("Native fog probe version differs")
    if tick is not None and probe.get("render_tick") != tick:
        raise ValueError("Fog probe belongs to another RGB frame")
    if probe.get("fog_enabled") is not enabled or probe.get("directional_colored_fog") is not True:
        raise ValueError("Native fog enable or directional setting differs")
    for key, expected in (("viewing_range_config", 30), ("draw_control_wanted_range", 30),
                          ("runData_fog_range", 30), ("sky_fog_start_ratio", ratio)):
        value = probe.get(key)
        if not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-4):
            raise ValueError(f"Unexpected native {key}: {value}, expected {expected}")
    if not probe.get("shader_uniforms_bound") or probe.get("fog_type") != "linear":
        raise ValueError("Native node shader did not bind linear fog uniforms")
    if enabled:
        for key, expected in (("actual_fog_start", 30 * ratio), ("actual_fog_end", 30),
                              ("shader_fog_distance", 30),
                              ("shader_fog_shading_parameter", 1 / (1 - ratio))):
            value = probe.get(key)
            if not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-4):
                raise ValueError(f"Unexpected native {key}: {value}, expected {expected}")
    if not math.isfinite(probe.get("camera_far_plane", math.nan)) or probe["camera_far_plane"] <= 30:
        raise ValueError("GPU camera far plane has been reduced to the map range")
    return probe


def validate_fog_absolute(probe, *, viewing_range=48, enabled=True, tick=None, start=20, end=30):
    """Check independent map draw range and absolute native fog endpoints."""
    if probe.get("version") != "voxwm-native-fog-v2":
        raise ValueError("Absolute fog requires the native v2 range probe")
    if tick is not None and probe.get("render_tick") != tick:
        raise ValueError("Fog probe belongs to another RGB frame")
    if probe.get("fog_enabled") is not enabled or probe.get("directional_colored_fog") is not True:
        raise ValueError("Native fog enable or directional setting differs")
    if probe.get("fog_type") != "linear":
        raise ValueError("Native fog type differs from linear")
    for key in ("viewing_range_config", "draw_control_wanted_range", "actual_map_draw_range"):
        value = probe.get(key)
        if not isinstance(value, (int, float)) or not math.isclose(value, viewing_range, abs_tol=1e-4):
            raise ValueError(f"Unexpected native {key}: {value}, expected {viewing_range}")
    for key in ("loaded_mapblocks_xyz", "submitted_mapblocks_xyz"):
        if not isinstance(probe.get(key), list):
            raise ValueError(f"Missing actual native {key}")
    for key in ("max_block_send_distance_nodes", "max_block_generate_distance_nodes"):
        value = probe.get(key)
        if not isinstance(value, (int, float)) or value < viewing_range:
            raise ValueError(f"Native {key} cannot cover draw range {viewing_range}: {value}")
    bound = probe.get("shader_uniforms_bound")
    if not bound and probe["submitted_mapblocks_xyz"]:
        raise ValueError("Native map geometry was submitted without node fog uniforms")
    if enabled:
        fields = [("actual_fog_start_distance", start), ("actual_fog_end_distance", end)]
        if bound:
            fields.extend((("shader_fog_distance", end),
                           ("shader_fog_shading_parameter", end / (end - start))))
        for key, expected in fields:
            value = probe.get(key)
            if not isinstance(value, (int, float)) or not math.isclose(value, expected, abs_tol=1e-4):
                raise ValueError(f"Unexpected native {key}: {value}, expected {expected}")
    far = probe.get("camera_far_plane", math.nan)
    if not isinstance(far, (int, float)) or not math.isfinite(far) or far <= viewing_range:
        raise ValueError("GPU far plane is restricted to the map draw range")
    return probe
