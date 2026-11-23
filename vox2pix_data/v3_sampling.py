"""Native v3 anchor screening, independent of the VAE survey band."""

from functools import lru_cache
import math

import numpy as np

from .protocol import (CENTER, NOMINAL_CAMERA_XYZ, SIDE, RejectedScene, _component_floors,
                       direction_for_angles, node_flags, require_air_camera, visibility_lookup)


SURVEY_Y = (-112, 240)
SCENE_KINDS = ("surface", "cave", "underwater")
ROOF_DISTANCE_MAX = 12.0
ROOF_THICKNESS_MIN = 2
ROOF_COVERAGE_MIN = 5 / 9
SKY_OPEN_MAX = 0.375
SIDE_ENCLOSED_MIN = 0.25


def require_camera(nodes, definitions, local_xyz, kind, water_contains=None):
    if kind != "underwater":
        require_air_camera(nodes, definitions, local_xyz)
        return
    x, y, z = (math.floor(value + .5) + SIDE // 2 for value in local_xyz)
    if not all(0 <= index < SIDE for index in (x, z, y)):
        raise RejectedScene("Underwater camera is outside the voxel map")
    if not definitions[int(nodes[x, z, y])].get("water", False):
        raise RejectedScene("Underwater camera cell is not water")
    if water_contains is None or not water_contains(local_xyz):
        raise RejectedScene("Underwater camera is outside the actual liquid volume")


def scene_visibility(nodes, definitions, kind, camera_origin=NOMINAL_CAMERA_XYZ):
    if nodes.shape != (SIDE,) * 3:
        raise ValueError("Scene requires one 48³ voxel map")
    air, free, _, _ = node_flags(nodes, definitions)
    x, y, z = (math.floor(value + .5) + SIDE // 2 for value in camera_origin)
    eye = (x, z, y)
    expected = (free[eye] and not air[eye]) if kind == "underwater" else air[eye]
    if not expected:
        raise RejectedScene(f"Nominal camera cell has wrong medium for {kind}")
    active = np.ones(40, dtype=bool)
    for offset, bins in visibility_lookup(tuple(camera_origin)).items():
        if not free[tuple(SIDE // 2 + d for d in offset)]:
            active[bins] = False
    if active.sum() < 20:
        raise RejectedScene("Fewer than 20 active view regions")
    return float(free[CENTER].mean()), float(free.mean()), active.tolist()


def settled_anchor(player_xyz):
    """The voxel containing the resting foot, just above a half-cell boundary."""
    x, y, z = player_xyz
    return tuple(math.floor(value + .5) for value in (x, y + .002, z))


@lru_cache(maxsize=1)
def _enclosure_probe_indices():
    # The 32 upper-hemisphere rays and eight horizontal rays probe local
    # enclosure, not global connectivity to distant sky beyond the 48³ crop.
    positions = [np.asarray(NOMINAL_CAMERA_XYZ) + np.arange(.25, radius + .25, .25)[:, None]
                 * direction_for_angles(yaw, pitch)
                 for pitch, radius in ((-18, 18), (-38, 18), (-58, 18), (-78, 18), (0, 12))
                 for yaw in range(0, 360, 45)]
    grid = np.floor(np.stack(positions[:32])[:, :, [0, 2, 1]] + SIDE // 2 + .5).astype(int)
    side = np.floor(np.stack(positions[32:])[:, :, [0, 2, 1]] + SIDE // 2 + .5).astype(int)
    return tuple(np.moveaxis(grid, -1, 0)), tuple(np.moveaxis(side, -1, 0))


def cave_evidence(nodes, definitions):
    """Evaluate provisional roof, upper-ray openness and horizontal wall gates."""
    if nodes.shape != (SIDE,) * 3:
        raise ValueError("Cave membership requires one 48³ voxel map")
    _, _, _, terrain = node_flags(nodes, definitions)
    mid = SIDE // 2
    start = mid + 3  # First voxel above the body clearance cells.
    column = terrain[mid, mid, start:]
    roofs = np.flatnonzero(column)
    if not roofs.size:
        raise RejectedScene("Cave: no natural overhead roof")
    first = int(roofs[0])
    breaks = np.flatnonzero(~column[first:])
    thickness = int(breaks[0]) if breaks.size else len(column) - first
    distance = start + first - mid - .5 - NOMINAL_CAMERA_XYZ[1]
    if distance > ROOF_DISTANCE_MAX or thickness < ROOF_THICKNESS_MIN:
        raise RejectedScene("Cave: overhead roof is too distant or thin")
    nearby = terrain[mid - 2:mid + 3:2, mid - 2:mid + 3:2, start:start + 13]
    roof_coverage = float((nearby[..., :-1] & nearby[..., 1:]).any(axis=2).mean())
    if roof_coverage < ROOF_COVERAGE_MIN:
        raise RejectedScene("Cave: insufficient nearby roof coverage")
    upper, lateral = _enclosure_probe_indices()
    open_fraction = float((~terrain[upper].any(axis=1)).mean())
    if open_fraction > SKY_OPEN_MAX:
        raise RejectedScene("Cave: too many upper rays open to sky")
    side_fraction = float(terrain[lateral].any(axis=1).mean())
    if side_fraction < SIDE_ENCLOSED_MIN:
        raise RejectedScene("Cave: insufficient lateral enclosure")
    return dict(roof_distance=distance, roof_thickness=thickness, roof_coverage=roof_coverage,
                sky_open_fraction=open_fraction,
                side_enclosed_fraction=side_fraction, sky_rays=32, lateral_rays=8,
                threshold_status="provisional")


def site_candidates(survey, definitions, kind, rng, rejected, y_min, hints=()):
    """Yield standable native floors; hints are survey-relative XYZ coordinates."""
    if kind not in SCENE_KINDS or survey.ndim != 3 or min(survey.shape) < SIDE:
        raise ValueError("Invalid v3 scene kind or terrain survey shape")
    air, free, _, terrain = node_flags(survey, definitions)
    layers = np.arange(survey.shape[2])
    surface = np.where(terrain, layers, -1).max(axis=2)
    if kind == "underwater":
        wet = free & ~air
        # Choose a submerged start, not a prescribed floor. The search band is
        # only for discovering a water column. Its actual bed is resolved later.
        submerged = wet[..., :-2] & wet[..., 1:-1] & wet[..., 2:]
        starts = np.where(submerged, layers[:-2], -1).max(axis=2)
        candidates = np.argwhere(starts >= 0)
        rejected["candidate_floors"] += len(candidates)
        if not len(candidates):
            rejected["site: no submerged starts"] += 1
        for index in rng.permutation(len(candidates)):
            x, z = map(int, candidates[index])
            y = int(starts[x, z])
            yield (x, y, z), max(int(surface[x, z]), y + 2) + y_min, "underwater_floor"
        return
    y = slice(24, survey.shape[2] - 23)
    positions = (slice(24, survey.shape[0] - 23), slice(24, survey.shape[1] - 23))
    index = (*positions, y)
    xs, zs = positions
    ys = layers[y]
    # Three native standing cells: one below the eye, the eye cell, and the foot cell.
    valid = air[index] & air[xs, zs, ys + 1] & air[xs, zs, ys + 2]
    if kind == "surface":
        # Tree canopies are not standable terrain anchors. Air above a natural top is.
        valid &= (ys == surface[xs, zs, None] + 1)
    else:
        valid &= terrain[xs, zs, ys - 1]
    candidates = np.argwhere(valid) + 24
    rejected["candidate_floors"] += len(candidates)
    if kind == "cave" and len(candidates):
        xs, zs, ys = candidates.T
        overhead = terrain[xs[:, None], zs[:, None], ys[:, None] + np.arange(3, 16)]
        # Keep the full roof check for the complete 48³ geometry. The cheap
        # filter helps avoid connected-component walks through outdoor air.
        roof = (overhead[:, :-1] & overhead[:, 1:]).any(axis=1)
        rejected["Cave: no nearby two-cell roof"] += int((~roof).sum())
        candidates = candidates[roof]
    if not len(candidates):
        rejected[f"site: no {kind} floor candidates"] += 1
        return

    floors = [tuple(map(int, row)) for row in candidates]
    if kind == "cave":
        underground_air = air & (layers < surface[:, :, None])
        floors = list(_component_floors(underground_air, floors, rng))
        if len(hints):
            ranked = np.asarray(floors)[:, [0, 2, 1]]
            distances = ((ranked[:, None, :] - np.asarray(hints)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            floors = [floors[i] for i in np.argsort(distances, kind="stable")]
    else:
        floors = [floors[i] for i in rng.permutation(len(floors))]
    for x, z, y in floors:
        if kind == "cave":
            region = survey[x - 24:x + 24, z - 24:z + 24, y - 24:y + 24]
            try:
                cave_evidence(region, definitions)
            except RejectedScene as exc:
                rejected[str(exc)] += 1
                continue
        yield (x, y, z), int(surface[x, z]) + y_min, f"{kind}_floor"
