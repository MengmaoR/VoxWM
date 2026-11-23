"""CPU pinhole projection of engine-derived voxel surfaces.

Luanti source references: client/content_mapblock.cpp (nodeboxes, fencelike,
plantlike and flowing-liquid geometry), mapnode.cpp (param2 interpretation),
and Irrlicht's buildCameraLookAtMatrixLH (camera basis).
"""
from collections import Counter
import ctypes as ct
from pathlib import Path
import math
import os
import subprocess
import sys
import weakref

import numpy as np


SIZE = 48
FULL_DRAW_TYPES = {"normal", "liquid", "glasslike", "allfaces", "allfaces_optional"}
PLANT_STYLES = (
    ((46, 0, False), (-44, 0, False)),
    ((91, 0, False), (1, 0, False)),
    ((121, 0, False), (241, 0, False), (1, 0, False)),
    tuple((yaw, 0.25, False) for yaw in (1, 91, 181, 271)),
    tuple((yaw, -0.5, True) for yaw in (1, 91, 181, 271)),
)
# Irrlicht coordinates are (x, y, z) with +Y up.
WALL_ROTATIONS = {
    0: ((1, 2, 180), (0, 2, 180)), 1: (),
    2: ((0, 1, 90),), 3: ((0, 1, -90), (1, 2, 180)),
    4: ((1, 2, -90), (0, 1, 90)), 5: ((1, 2, 90), (0, 1, 90)),
}


def build_library(directory):
    """Compile once per collection invocation, with no engine rebuild."""
    output = Path(directory).resolve() / "vox2pix_trace.so"
    linker = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-O3", "-std=c++17", "-fPIC", linker,
         str(Path(__file__).with_name("trace.cpp")), "-o", str(output)],
        check=True,
    )
    return output


def _pointer(array):
    return ct.c_void_p(array.ctypes.data)


def _rotate(vertices, first, second, degrees):
    sine, cosine = math.sin(math.radians(degrees)), math.cos(math.radians(degrees))
    a, b = vertices[:, first].copy(), vertices[:, second].copy()
    vertices[:, first] = a * cosine - b * sine
    vertices[:, second] = a * sine + b * cosine


def _random16(seed):
    # Luanti PseudoRandom truncates signed division toward zero.
    while True:
        seed = (seed * 1103515245 + 12345) & 0xffffffff
        signed = seed if seed < 2**31 else seed - 2**32
        yield math.trunc(signed / 65536) % 16


def _plant_quads(definition, param2, coordinates, rooted):
    scale = 0.5 * float(definition.get("visual_scale", 1))
    mode = definition.get("paramtype2")
    style, height, rotation = 0, 1.0, 0.0
    offset = np.zeros(3, dtype=float)
    x, y, z = (int(c) % 16 for c in coordinates)
    if rooted:
        y += 1  # The rooted plant's renderer starts in the cell above the root.
    if mode == "meshoptions":
        style = param2 & 7
        if style >= len(PLANT_STYLES):
            return
        if param2 & 0x10:
            scale *= 1.41421
        if param2 & 0x08:
            random = _random16(x << 8 | z | y << 16)
            offset[0] = next(random) / 16 * 0.29 - 0.145
            offset[2] = next(random) / 16 * 0.29 - 0.145
    elif mode == "degrotate":
        rotation = 1.5 * (param2 % 240)
    elif mode == "colordegrotate":
        rotation = 15 * ((param2 & 31) % 24)
    elif mode == "leveled":
        height = param2 / 16.0

    wall = (param2 & 7) if mode in {"wallmounted", "colorwallmounted"} else 1
    if rooted:
        # Rooted nodes have a full root cube and a displaced plant surface.
        if wall == 0:
            offset[1] += 2
        elif wall in (2, 3, 4, 5):
            offset += (-1, 1, 0)

    for face_index, (yaw, shift, top_only) in enumerate(PLANT_STYLES[style]):
        top = -0.5 + 2 * scale * height
        vertices = np.array([(-scale, top, shift), (scale, top, shift),
                             (scale, -0.5, 0 if top_only else shift),
                             (-scale, -0.5, 0 if top_only else shift)])
        _rotate(vertices, 0, 2, yaw + rotation)
        if mode == "meshoptions" and param2 & 0x20:
            offset[1] = -next(_random16(face_index | x << 16 | z << 8 | y << 24)) / 128
        vertices += offset
        for first, second, degrees in WALL_ROTATIONS.get(wall, ()):
            _rotate(vertices, first, second, degrees)
        if rooted:
            vertices[:, 1] += 1
        yield vertices


def is_water_liquid(definition):
    # The water group also includes solid waterlogged mangrove roots.
    return bool(definition.get("water") and (definition.get("liquidtype") in {"source", "flowing"}
                or definition.get("drawtype") in {"liquid", "flowingliquid"}))


def _liquid_top(nodes, params, names, definition, at, flowing):
    """The four Luanti corner levels in the drawLiquidTop vertex order."""
    x, y, z = at
    source = names.get(definition.get("liquid_alternative_source"), -1)
    air = names["air"]
    liquid_range = min(8, max(1, int(definition.get("liquid_range", 8))))

    def read(i, j, k):
        if 0 <= i < SIZE and 0 <= j < SIZE and 0 <= k < SIZE:
            return int(nodes[i, k, j])
        return -2

    def level(dx, dz):
        levels, air_count = [], 0
        for i in (x + dx - 1, x + dx):
            for k in (z + dz - 1, z + dz):
                node = read(i, y, k)
                if node == -2:
                    continue
                if node == source or read(i, y + 1, k) in (source, flowing):
                    return 0.5
                if node == flowing:
                    param2 = int(params[i, k, y])
                    raw_level = max(0, (param2 & 0x07) - (8 - liquid_range))
                    levels.append(-0.5 + (raw_level + 0.5) / liquid_range)
                elif node == air:
                    air_count += 1
        if air_count >= 2:
            return -0.48
        return sum(levels) / len(levels) if levels else 0

    corners = ((0, 1), (1, 1), (1, 0), (0, 0))
    return np.array([(dx - 0.5, level(dx, dz), dz - 0.5) for dx, dz in corners])


def _liquid_quads(nodes, params, names, definition, at, flowing):
    """Native corner levels with an entry surface per voxel, even at shared sides."""
    top = _liquid_top(nodes, params, names, definition, at, flowing)
    bottom = top.copy()
    bottom[:, 1] = -0.5
    yield top
    yield bottom[::-1]
    for i in range(4):
        j = (i + 1) % 4
        yield np.array((top[i], bottom[i], bottom[j], top[j]))


def _flowing_water(definition):
    return definition.get("drawtype") == "flowingliquid" or definition.get("liquidtype") == "flowing"


def _water_tops(nodes, params, definitions):
    tops = np.full((*nodes.shape, 4), -1., dtype=np.float32)
    names = {d["name"]: i for i, d in definitions.items()}
    for node in np.unique(nodes):
        definition = definitions[int(node)]
        if not is_water_liquid(definition):
            continue
        mask = nodes == node
        if not _flowing_water(definition):
            tops[mask] = .5
            continue
        for x, z, y in np.argwhere(mask):
            tops[x, z, y] = _liquid_top(nodes, params, names, definition, (x, y, z), node)[:, 1]
    return tops


def water_contains(nodes, params, definitions, local_xyz):
    """Test the actual Luanti XYZ eye point against the v3 liquid volume."""
    xyz = np.asarray(local_xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return False
    center = np.floor(xyz + .5).astype(int)
    x, y, z = center + SIZE // 2
    if min(x, y, z) < 0 or max(x, y, z) >= SIZE:
        return False
    node = int(nodes[x, z, y])
    definition = definitions[node]
    if not is_water_liquid(definition):
        return False
    dx, dy, dz = xyz - center
    if dy <= -.5:
        return False
    if not _flowing_water(definition):
        return dy < .5
    names = {d["name"]: i for i, d in definitions.items()}
    a, b, c, d = _liquid_top(nodes, params, names, definition, (x, y, z), node)[:, 1].astype(np.float32)
    u, v = dx + .5, dz + .5
    height = a + (b - a) * u + (c - b) * (1 - v) if u + v >= 1 else d + (c - d) * u + (a - d) * v
    return bool(dy < height)


def _fence_boxes(nodes, kinds, at):
    """Luanti posts own their positive-X and positive-Z connecting rails."""
    x, y, z = at
    post, bar = 1 / 8, 1 / 16
    yield (-post, -0.5, -post, post, 0.5, post)
    for h in (-0.25, 0.25):
        if x + 1 < SIZE and kinds[nodes[x + 1, z, y]] == "fencelike":
            yield (post, h - bar, -bar, 1 - post, h + bar, bar)
        if z + 1 < SIZE and kinds[nodes[x, z + 1, y]] == "fencelike":
            yield (-bar, h - bar, post, bar, h + bar, 1 - post)


def _scene_geometry(nodes, params, definitions, boxes, foot_xyz):
    node_ids = np.unique(nodes)
    kinds = {int(i): ("airlike" if definitions[int(i)]["name"] == "air"
                      else definitions[int(i)].get("drawtype", "normal")) for i in node_ids}
    names = {d["name"]: i for i, d in definitions.items()}
    cubes = np.isin(nodes, [i for i, kind in kinds.items() if kind in FULL_DRAW_TYPES]).astype("u1")
    special = [i for i, kind in kinds.items() if kind not in FULL_DRAW_TYPES | {"airlike"}]
    geometry_counts = Counter()
    shapes = []
    foot_xyz = np.asarray(foot_xyz)

    for index in np.flatnonzero(np.isin(nodes, special)):
        x, remainder = divmod(int(index), SIZE * SIZE)
        z, y = divmod(remainder, SIZE)
        node = int(nodes[x, z, y])
        definition = definitions[node]
        drawtype = kinds[node]
        center = np.array((x - 24, y - 24, z - 24))
        boxes_here, quads_here, triangles = (), (), 1
        if drawtype in {"plantlike", "plantlike_rooted"}:
            rooted = drawtype == "plantlike_rooted"
            boxes_here = [(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)] if rooted else ()
            quads_here = _plant_quads(definition, int(params[x, z, y]), foot_xyz + center, rooted)
            geometry_counts[drawtype] += 1
        elif drawtype == "flowingliquid":
            quads_here = _liquid_quads(nodes, params, names, definition, (x, y, z), node)
            triangles = 2
            geometry_counts["flowingliquid_corner_shell"] += 1
            if x in (0, SIZE - 1) or z in (0, SIZE - 1) or y == SIZE - 1:
                geometry_counts["flowingliquid_boundary"] += 1
        elif drawtype == "fencelike":
            boxes_here = _fence_boxes(nodes, kinds, (x, y, z))
            geometry_counts["fencelike"] += 1
            if x == SIZE - 1 or z == SIZE - 1:
                geometry_counts["fencelike_boundary"] += 1
        else:
            boxes_here = boxes.get(int(index)) or ()
            if drawtype == "nodebox":
                if int(index) not in boxes:
                    raise ValueError(f"Missing native nodebox for {definition['name']} at ENU index {index}")
                geometry_counts["native_nodebox"] += 1
            else:
                prefix = "selection_box_approximation" if boxes_here else "missing_selection_box"
                geometry_counts[f"{prefix}:{drawtype}"] += 1

        for bounds in boxes_here:
            bounds = np.asarray(bounds, dtype=float).reshape(2, 3)
            if np.all(bounds[0] < bounds[1]):
                shapes.append([index, 0, *(bounds + center).ravel(), 0, 0, 0])
        for quad in quads_here:
            quad = quad + center
            for corners in ([0, 1, 2], [2, 3, 0]):
                shapes.append([index, triangles, *quad[corners].ravel()])
    return cubes, np.asarray(shapes, dtype=np.float32).reshape(-1, 11), dict(geometry_counts)


class Projector:
    def __init__(self, library_path):
        lib = ct.CDLL(str(library_path))
        pointer = ct.c_void_p
        lib.vox2pix_scene.argtypes = (pointer, pointer, pointer, pointer, ct.c_int)
        lib.vox2pix_scene.restype = pointer
        lib.vox2pix_destroy.argtypes = (pointer,)
        lib.vox2pix_destroy.restype = None
        lib.vox2pix_render.argtypes = (pointer, pointer, pointer, ct.c_double, ct.c_double,
                                     ct.c_int, ct.c_int, pointer, pointer, pointer)
        lib.vox2pix_render.restype = None
        lib.vox2pix_render_v3.argtypes = (pointer, pointer, pointer, pointer, ct.c_double, ct.c_double,
                                        ct.c_int, ct.c_int, pointer, pointer, pointer, pointer)
        lib.vox2pix_render_v3.restype = None
        self.library = lib

    def scene(self, node_id, param2, definitions, boxes, *, foot_xyz=(0, 0, 0)):
        return TraceScene(self.library, node_id, param2, definitions, boxes, foot_xyz)


class TraceScene:
    def __init__(self, library, node_id, param2, definitions, boxes, foot_xyz):
        self.node_id = np.array(node_id, dtype=np.uint16, order="C")
        self.param2 = np.array(param2, dtype=np.uint8, order="C")
        if self.node_id.shape != (SIZE,) * 3 or self.param2.shape != self.node_id.shape:
            raise ValueError("White Model requires 48³ aligned ENU node_id and param2")
        self.cubes, shapes, self.geometry_counts = _scene_geometry(
            self.node_id, self.param2, definitions, boxes, foot_xyz
        )
        self.library = library
        self.definitions = definitions
        self._tops = None
        self.handle = library.vox2pix_scene(
            _pointer(self.node_id), _pointer(self.param2), _pointer(self.cubes), _pointer(shapes), len(shapes)
        )
        if not self.handle:
            raise MemoryError("Could not build the White Model geometry index")
        self._cleanup = weakref.finalize(self, library.vox2pix_destroy, self.handle)

    @staticmethod
    def _camera(cam_xyz_local, direction_xyz, fov_x, fov_y, width, height):
        origin = np.ascontiguousarray(cam_xyz_local, dtype=np.float64)
        direction = np.ascontiguousarray(direction_xyz, dtype=np.float64)
        if (origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(origin).all()
                or not np.isfinite(direction).all() or np.hypot(direction[0], direction[2]) <= 1e-12):
            raise ValueError("Invalid or pole-degenerate Craftium camera pose")
        if not (0 < fov_x < math.pi and 0 < fov_y < math.pi and width > 0 and height > 0):
            raise ValueError("Invalid pinhole intrinsics or resolution")
        return origin, direction

    def render(self, cam_xyz_local, direction_xyz, fov_x, fov_y, width, height):
        origin, direction = self._camera(cam_xyz_local, direction_xyz, fov_x, fov_y, width, height)
        shape = (3, height, width)
        layers = tuple(np.empty(shape, dtype=dtype) for dtype in (np.uint16, np.uint8, np.uint8))
        self.library.vox2pix_render(self.handle, _pointer(origin), _pointer(direction),
                                    fov_x, fov_y, width, height, *map(_pointer, layers))
        return layers

    def water_contains(self, local_xyz):
        return water_contains(self.node_id, self.param2, self.definitions, local_xyz)

    def render_v3(self, cam_xyz_local, direction_xyz, fov_x, fov_y, width, height):
        origin, direction = self._camera(cam_xyz_local, direction_xyz, fov_x, fov_y, width, height)
        if self._tops is None:
            self._tops = _water_tops(self.node_id, self.param2, self.definitions)
        fields = tuple(np.empty((height, width), dtype=dtype)
                       for dtype in (np.uint16, np.uint8, np.uint8, np.uint8))
        self.library.vox2pix_render_v3(self.handle, _pointer(self._tops), _pointer(origin), _pointer(direction),
                                       fov_x, fov_y, width, height, *map(_pointer, fields))
        return fields
