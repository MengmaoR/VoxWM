"""Pilot quota, foot-anchor selection and fixed camera-region definitions."""
from collections import deque
from functools import lru_cache
from itertools import product
import math

import numpy as np

from vae_data.sharded.protocol import BIOMES, WORLD_BOTTOM, WORLD_TOP

VERSION = "vox2pix-pilot-v2-camera16"
V3_VERSION = "vox2pix-v3-surface-watercount-cave-underwater-torch"
V3_KINDS = ("surface", "cave", "underwater")
HEIGHTS = ("surface", "shallow", "deep")
CENTER = (slice(16, 32),) * 3
SIDE = 48
SURVEY_SIDE = SIDE + 15
NOMINAL_CAMERA_HEIGHT = 1.6
NOMINAL_CAMERA_XYZ = (0., NOMINAL_CAMERA_HEIGHT, 0.)
VISIBILITY_RADIUS = 1.6
VISIBILITY_LOOKUP_VERSION = "voxel-centers-r1.6-v2"
# Standing collision box extends 1.8 voxels above the integer foot. These
# air cells cover that volume with the project's half-voxel boundaries.
PLAYER_AIR_OFFSETS = (0, 1, 2)
SETTINGS = dict(
    fov=72, display_gamma=3.0, camera_smoothing=0., view_bobbing_amount=0.,
    cinematic=False, time_speed=0, world_start_time=12000,
    mcl_doWeatherCycle=False, mobs_spawn=False, viewing_range=72,
    enable_fog=False, enable_dynamic_shadows=False, enable_auto_exposure=False,
    enable_waving_water=False, enable_waving_leaves=False, enable_waving_plants=False,
    enable_particles=False, antialiasing="none", undersampling=1,
    enable_post_processing=False, enable_bloom=False, enable_water_reflections=False,
    starting_inventory_creative="",
)
V3_RGB = dict(viewing_range=30, enable_fog=True, directional_colored_fog=True,
              fog_start=.8, display_gamma=3.0)


def v3_plan(config):
    kinds = config["scene_kinds"]
    count = config["scenes_per_kind"]
    if tuple(kinds) != V3_KINDS or type(count) is not int or count not in (1, 3):
        raise ValueError("V3 smoke requires 1 or 3 scenes per kind for surface, cave and underwater")
    return [dict(scene_id=index * count + offset, kind=kind)
            for index, kind in enumerate(kinds) for offset in range(count)]


class RejectedScene(RuntimeError):
    """The observed terrain, camera or world state failed the pilot criteria."""


def plan(seed):
    """Seeded biome permutation assigns height patterns independently of biome names."""
    biomes = np.random.default_rng(seed).permutation(sorted(BIOMES)).tolist()
    patterns = [(5, 2, 1)] * 10 + [(5, 1, 2)] * 10 + [(4, 2, 2)] * 5
    slots = []
    for biome, quotas in zip(biomes, patterns, strict=True):
        for height, quota in zip(HEIGHTS, quotas, strict=True):
            for _ in range(quota):
                slots.append(dict(scene_id=len(slots), biome=biome, height=height))
    return slots


def camera_angles(direction_xyz):
    """Luanti yaw zero points +Z, positive yaw points -X, positive pitch looks down."""
    x, y, z = direction_xyz
    return math.degrees(math.atan2(-x, z)) % 360, math.degrees(math.atan2(-y, math.hypot(x, z)))


def direction_for_angles(yaw_deg, pitch_deg):
    yaw, pitch = map(math.radians, (yaw_deg, pitch_deg))
    return np.array((-math.sin(yaw) * math.cos(pitch), -math.sin(pitch),
                     math.cos(yaw) * math.cos(pitch)))


def region_id(yaw_deg, pitch_deg):
    if not -90 <= pitch_deg <= 90:
        raise ValueError("Pitch is outside [-90, 90]")
    return min(4, int((pitch_deg + 90) / 36)) * 8 + int((yaw_deg % 360) / 45)


def camera_local_xyz(actual_world_xyz, foot_world_xyz):
    """Check the native eye pose and return its XYZ relative to the foot."""
    camera = np.asarray(actual_world_xyz, dtype=np.float64)
    foot = np.asarray(foot_world_xyz, dtype=np.float64)
    if camera.shape != (3,) or foot.shape != (3,) or not np.isfinite(camera).all() or not np.isfinite(foot).all():
        raise RejectedScene("Camera or foot position is invalid")
    local = camera - foot
    if np.max(np.abs(local[[0, 2]])) > .002 or abs(local[1] - NOMINAL_CAMERA_HEIGHT) > .05:
        raise RejectedScene("Native camera did not settle near the nominal first-person pose")
    return local


def require_air_camera(nodes, definitions, local_xyz):
    """Find the integer-centered cell containing the real or nominal eye."""
    x, y, z = (math.floor(coord + .5) + SIDE // 2 for coord in local_xyz)
    index = (x, z, y)
    if any(i < 0 or i >= size for i, size in zip(index, nodes.shape, strict=True)):
        raise RejectedScene("Camera cell is outside the voxel map")
    if definitions[int(nodes[index])]["name"] != "air":
        raise RejectedScene("Camera cell is not air")


@lru_cache(maxsize=1)
def visibility_lookup():
    """Rasterize nearby foot-relative ENU node centers once per process."""
    rays = np.array([
        [direction_for_angles(yaw * 45 + (u + .5) * 5, -90 + pitch * 36 + (v + .5) * 4)
         for u, v in product(range(9), repeat=2)]
        for pitch, yaw in product(range(5), range(8))
    ])
    rays[np.abs(rays) < 1e-12] = 1e-12
    camera_x, camera_y, camera_z = NOMINAL_CAMERA_XYZ
    extents = [range(math.ceil(coord - VISIBILITY_RADIUS), math.floor(coord + VISIBILITY_RADIUS) + 1)
               for coord in (camera_x, camera_z, camera_y)]
    camera_xyz = np.asarray(NOMINAL_CAMERA_XYZ)
    result = {}
    for dx, dz, dy in product(*extents):
        center = np.array([dx, dy, dz]) - camera_xyz
        if center @ center > VISIBILITY_RADIUS ** 2:
            continue
        a, b = (center - .5) / rays, (center + .5) / rays
        enters = np.maximum(np.minimum(a, b).max(axis=-1), 0)
        exits = np.maximum(a, b).min(axis=-1)
        result[(dx, dz, dy)] = np.flatnonzero((exits > enters).any(axis=-1))
    return result


def node_flags(nodes, definitions):
    """Map engine or dataset IDs to free space, solid support and terrain flags."""
    ids = np.unique(nodes)
    free = np.zeros(65536, dtype=bool)
    air, water, support, terrain = [free.copy() for _ in range(4)]
    for ident in ids:
        entry = definitions[int(ident)]
        if entry["name"] == "ignore":
            raise RejectedScene("Unloaded native voxels")
        air[ident] = entry["name"] == "air"
        water[ident] = entry.get("water", False)
        support[ident] = entry.get("walkable", False) and not air[ident] and not water[ident]
        terrain[ident] = entry.get("terrain", False) and support[ident]
    free = air | water
    return air[nodes], free[nodes], support[nodes], terrain[nodes]


def scene_visibility(nodes, definitions):
    if nodes.shape != (48, 48, 48):
        raise ValueError("Scene requires one 48³ voxel map")
    _, free, _, _ = node_flags(nodes, definitions)
    require_air_camera(nodes, definitions, NOMINAL_CAMERA_XYZ)
    center_ratio, full_ratio = float(free[CENTER].mean()), float(free.mean())
    if center_ratio < 1 / 3 or full_ratio < 1 / 4:
        raise RejectedScene("Scene fails the center or full free-space gate")
    active = np.ones(40, dtype=bool)
    for offset, bins in visibility_lookup().items():
        index = tuple(SIDE // 2 + d for d in offset)
        if not free[index]:
            active[bins] = False
    if active.sum() < 20:
        raise RejectedScene("Fewer than 20 active view regions")
    return center_ratio, full_ratio, active.tolist()


def sample_views(active, rng):
    views = []
    for region in rng.choice(np.flatnonzero(active), size=5, replace=False):
        pitch, yaw = divmod(int(region), 8)
        # Stay clear of exact bin edges and the poles under client float rounding.
        views.append(dict(region_id=int(region),
                          requested_yaw_deg=float(45 * yaw + rng.uniform(.05, 44.95)),
                          requested_pitch_deg=float(-90 + 36 * pitch + rng.uniform(.05, 35.95))))
    return views


def _component_floors(free, floors, rng):
    """Group floor candidates by their connected six-neighbor free-space volume."""
    pending = set(floors)
    shape = free.shape
    for first in rng.permutation(len(floors)):
        start = floors[first]
        if start not in pending:
            continue
        pending.remove(start)
        queue = deque([start])
        free[start] = False
        found = [start]
        while queue:
            x, z, y = queue.popleft()
            for q in ((x - 1, z, y), (x + 1, z, y), (x, z - 1, y),
                      (x, z + 1, y), (x, z, y - 1), (x, z, y + 1)):
                if all(0 <= coordinate < size for coordinate, size in zip(q, shape)) and free[q]:
                    free[q] = False
                    queue.append(q)
                    if q in pending:
                        pending.remove(q)
                        found.append(q)
        for i in rng.permutation(len(found)):
            yield found[i]


def site_candidates(survey, definitions, height, rng, rejections=None):
    """Find actual native floors; return XYZ offsets relative to the survey origin."""
    width = survey.shape[0]
    if survey.shape != (width, width, WORLD_TOP - WORLD_BOTTOM + 1) or width < SURVEY_SIDE:
        raise ValueError("Native terrain survey has the wrong shape")
    air, free, support, terrain = node_flags(survey, definitions)
    layers = np.arange(survey.shape[2])
    surface = np.where(terrain, layers, -1).max(axis=2)
    # Area integrals allow exact free-space screening once for every possible floor.
    summed = np.pad(free, ((1, 0),) * 3).astype(np.int32)
    for axis in range(3):
        np.cumsum(summed, axis=axis, out=summed)

    def free_cells(x, z, y, radius):
        a, b, c = x - radius, z - radius, y - radius
        i, j, k = x + radius, z + radius, y + radius
        return (summed[i, j, k] - summed[a, j, k] - summed[i, b, k] - summed[i, j, c]
                + summed[a, b, k] + summed[a, j, c] + summed[i, b, c] - summed[a, b, c])

    def body_clear(x, z, y):
        return all(air[x, z, y + dy] for dy in PLAYER_AIR_OFFSETS)

    columns = list(product(range(24, width - 23), repeat=2))
    floors = []
    eligible_band = standing_floor = center_clear = False
    for p in rng.permutation(len(columns)):
        x, z = columns[p]
        sy = int(surface[x, z])
        if sy < 0:
            continue
        if height == "surface":
            sky_surface = support[x, z] | (free[x, z] & ~air[x, z])
            for top in np.flatnonzero(sky_surface)[::-1].tolist():
                if top < sy:
                    break
                is_water = bool(free[x, z, top])
                y = top + 1
                if (24 <= y <= survey.shape[2] - 24 and body_clear(x, z, y)
                    and free_cells(x, z, y, 8) * 3 >= 16 ** 3 and free_cells(x, z, y, 24) * 4 >= 48 ** 3):
                    # S remains the native terrain reference; the actual top is encoded by y.
                    yield (x, y, z), sy + WORLD_BOTTOM, "surface_water" if is_water else "surface_floor"
                if is_water:
                    break  # No submerged fallback below the first water surface.
            continue
        low, high = (sy - 24, sy - 8) if height == "shallow" else (24, sy - 32)
        band = range(max(24, low), min(survey.shape[2] - 24, high) + 1)
        eligible_band |= bool(band)
        for y in band:
            if not (support[x, z, y - 1] and body_clear(x, z, y) and terrain[x, z, y + 3:sy + 1].any()):
                continue
            standing_floor = True
            if free_cells(x, z, y, 8) * 3 < 16 ** 3:
                continue
            center_clear = True
            if free_cells(x, z, y, 24) * 4 >= 48 ** 3:
                floors.append((x, z, y))
    if not floors and height != "surface" and rejections is not None:
        if not eligible_band:
            reason = "site: no legal height band in the survey"
        elif not standing_floor:
            reason = "site: no standing cave floor with roof in height band"
        elif not center_clear:
            reason = "site: cave floors fail center16 free-space gate"
        else:
            reason = "site: cave floors fail full48 free-space gate"
        rejections[reason] += 1
    if floors:
        underground = free & (layers < surface[:, :, None])
        for x, z, y in _component_floors(underground, floors, rng):
            yield (x, y, z), int(surface[x, z]) + WORLD_BOTTOM, "cave_floor"
