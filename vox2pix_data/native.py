"""Read native Mapgen v7 terrain and pair a stable voxel map with client RGB packets."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import time

import numpy as np

from vae_data.sharded.clock import RAW, verify_clock
from vae_pilot.sampling import biome_group
from voxwm_gt.session import GTSession

from .geometry import Projector
from .protocol import (
    BIOMES, CAMERA_OFFSET_XYZ, SETTINGS, SURVEY_SIDE, WORLD_BOTTOM, WORLD_TOP,
    RejectedScene, camera_angles, direction_for_angles, region_id, sample_views,
    scene_visibility, site_candidates,
)

HERE = Path(__file__).resolve().parent
REUSED_LUA = HERE.parent / "vae_data/sharded"
NOOP = np.zeros(4, dtype=np.int64)
SETUP = dict(SETTINGS, node_highlighting="none",
             performance_tradeoffs=False, mcl_logging_mobs_spawn=False)
WORLDS_PER_SLOT = 8
SITES_PER_WORLD = 4


class RenderSession(GTSession):
    """Bound a lost native socket read without changing older GT users."""

    def step(self, action):
        fd = self.env.unwrapped.mt_chann.connfd
        if getattr(self, "_deadline_fd", None) != fd:
            with socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) as conn:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("@ll", 30, 0))
            self._deadline_fd = fd
        return super().step(action)


def decode_region(payload):
    """Convert native chunk-major raw rows to the foot-centered ENU array."""
    if len(payload) != 48 ** 3 * RAW.itemsize:
        raise ValueError("Incomplete native voxel payload")
    return (np.frombuffer(payload, RAW).reshape(3, 3, 3, 16, 16, 16)
            .transpose(0, 3, 1, 4, 2, 5).reshape(48, 48, 48))


@lru_cache(maxsize=None)
def _builds(craftium_dir):
    game = craftium_dir / "craftium-envs/openworld-creative/games/VoxeLibre"
    return tuple(subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip() for path in (craftium_dir, game))


class NativeScene:
    def __init__(self, session):
        self.session = session
        self.seq = 0
        self.packet = json.loads((session.world / "vox2pix_engine.json").read_text())
        if self.packet["mapgen"] != "v7":
            raise RuntimeError("Vox2Pix requires native Mapgen v7")
        self.definitions = {row["id"]: row for row in self.packet["definitions"]}
        self.vocabulary = sorted(row["name"] for row in self.definitions.values())
        if len(self.vocabulary) >= 65535:
            raise RuntimeError("Node Name vocabulary does not fit in uint16 with the empty sentinel")
        name_ids = {name: index for index, name in enumerate(self.vocabulary)}
        self.lookup = np.full(65536, 65535, dtype=np.uint16)
        self.stable_defs = {}
        for runtime_id, row in self.definitions.items():
            ident = name_ids[row["name"]]
            self.lookup[runtime_id] = ident
            self.stable_defs[ident] = row

    def step(self):
        rgb, _, terminated, truncated, info = self.session.step(NOOP)
        if terminated or truncated:
            raise RejectedScene("The native render session ended")
        return rgb, info

    def command(self, kind, **fields):
        self.seq += 1
        world = self.session.world
        target = world / "vox2pix_command.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(request_id=self.seq, kind=kind, **fields), allow_nan=False))
        temporary.replace(target)
        response = world / "vox2pix_response.json"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            rgb, info = self.step()
            row = json.loads(response.read_text()) if response.is_file() else {}
            if row.get("request_id") == self.seq:
                verify_clock(row, info)
                return row, rgb, info
        raise TimeoutError(f"Native {kind} did not finish within 90 seconds")

    def survey(self, site):
        x, z = site
        low = (x - 24, WORLD_BOTTOM, z - 24)
        high = (low[0] + SURVEY_SIDE - 1, WORLD_TOP, low[2] + SURVEY_SIDE - 1)
        row, _, _ = self.command("survey", low_xyz=low, high_xyz=high)
        if row["status"] != "surveyed":
            raise RejectedScene("Native site emergence failed")
        survey = np.fromfile(self.session.world / "vox2pix_survey.bin", dtype="<u2")
        return survey.reshape(SURVEY_SIDE, SURVEY_SIDE, WORLD_TOP - WORLD_BOTTOM + 1), low

    def geometry_snapshot(self, low, kind, boxes=False):
        row, rgb, info = self.command("snapshot", low_xyz=low, anchor_kind=kind, boxes=boxes)
        if row["status"] != "captured":
            raise RejectedScene(row.get("reason", "Native voxel snapshot failed"))
        state = decode_region((self.session.world / "vox2pix_region.bin").read_bytes())
        ids = self.lookup[state["node"]]
        if (ids == 65535).any():
            raise RuntimeError("Native region contains an unregistered content ID")
        arrays = ids, state["param2"].copy()
        return arrays, row, rgb, info

    def acquire(self, foot, kind, config, rng, library_path):
        origin = tuple(coordinate - 24 for coordinate in foot)
        camera = np.add(foot, CAMERA_OFFSET_XYZ)
        # Neither the initial player position nor the camera position sets the crop.
        self.command("pose", relocate=True, foot_xyz=foot, anchor_kind=kind, yaw=0., pitch=0.)
        for _ in range(72):
            self.step()
        baseline, _, _, _ = self.geometry_snapshot(origin, kind, boxes=True)
        center, full, active = scene_visibility(baseline[0], self.stable_defs)
        box_rows = json.loads((self.session.world / "vox2pix_boxes.json").read_text())["boxes"]
        projector = Projector(library_path).scene(
            *baseline, self.stable_defs, dict(box_rows or []), foot_xyz=foot
        )
        views = sample_views(active, rng)
        rgbs, whites = [], []
        width, height = config["width"], config["height"]
        for view in views:
            requested = view["requested_yaw_deg"], view["requested_pitch_deg"]
            expected_direction = direction_for_angles(*requested)
            self.command("pose", relocate=False, yaw=requested[0], pitch=requested[1])
            # pyConnStep reads the previous completed render and its camera.
            # Allow force-move, the client mesh draw list and the framebuffer to settle.
            for _ in range(12):
                self.step()
            previous, prior, _, _ = self.geometry_snapshot(origin, kind)
            current, capture, rgb, info = self.geometry_snapshot(origin, kind)
            tick = int(capture["native_clock"]["server_tick"])
            if tick != int(prior["native_clock"]["server_tick"]) + 1:
                raise RejectedScene("RGB was not bracketed by consecutive native voxel captures")
            if any(not np.array_equal(field, expected)
                   for state in (previous, current) for field, expected in zip(state, baseline)):
                raise RejectedScene("Node Name or raw param2 changed between native views")
            position = np.asarray(info["cam_pos"], dtype=np.float64)
            direction = np.asarray(info["cam_dir"], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if not np.all(np.isfinite(position)) or not np.isfinite(norm) or norm < 1e-10:
                raise RejectedScene("The client camera packet is invalid")
            direction /= norm
            yaw, pitch = camera_angles(direction)
            if (not np.allclose(position, camera, atol=.002, rtol=0)
                    or direction @ expected_direction < math.cos(math.radians(.04))
                    or region_id(yaw, pitch) != view["region_id"]):
                raise RejectedScene("The native render pose did not settle at the requested view")
            fov_x, fov_y = float(info["cam_fov_x"]), float(info["cam_fov_y"])
            if not (0 < fov_x < math.pi and 0 < fov_y < math.pi
                    and math.isclose(math.tan(fov_x / 2) / math.tan(fov_y / 2),
                                     width / height, rel_tol=1e-5)):
                raise RejectedScene("Actual camera FOV does not match the RGB aspect ratio")
            if capture["weather"] != "none" or not math.isclose(capture["world_time"], .5, abs_tol=1e-5):
                raise RejectedScene("Native start-day and clear-weather settings did not hold")
            if capture["gamma"] != SETUP["display_gamma"]:
                raise RejectedScene("Native render gamma differs from the pilot setting")
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError("Native RGB packet has the wrong size or dtype")
            rgbs.append(rgb.copy())
            whites.append(projector.render(position - foot, direction, fov_x, fov_y, width, height))
            view.update(cam_pos_xyz=position.tolist(), cam_dir_xyz=direction.tolist(),
                        yaw_deg=yaw, pitch_deg=pitch, fov_x_rad=fov_x, fov_y_rad=fov_y,
                        resolution=[width, height], gamma=capture["gamma"],
                        world_time=capture["world_time"], weather=capture["weather"],
                        simulation_tick=tick, render_simulation_tick=tick - 1)
        arrays = dict(scene_node_id=baseline[0], scene_param2=baseline[1])
        for name, fields in zip(("white_node_id", "white_param2", "white_face"), zip(*whites)):
            arrays[name] = np.stack(fields)
        meta = dict(
            data_source="native",
            foot_anchor_xyz=list(foot), region_origin_xyz=list(origin),
            region_origin_enu=[origin[0], origin[2], origin[1]], anchor_kind=kind,
            center_free_ratio=center, full_free_ratio=full, active_view_regions=active,
            views=views, render_settings=dict(SETUP, renderer=config["renderer"],
                fixed24=True, rgb_observations=True, hud=False),
            gamma=views[0]["gamma"], world_time=views[0]["world_time"], weather=views[0]["weather"],
            engine_version=self.packet["version"], geometry_counts=projector.geometry_counts,
            timing_patch=capture["native_clock"]["patch_version"],
            region_capture_patch=capture["native_clock"]["region_capture_version"],
            water_nodes=sorted(d["name"] for d in self.definitions.values() if d["water"]),
            glass_nodes=sorted(d["name"] for d in self.definitions.values() if d["glass"]),
        )
        return meta, self.vocabulary, arrays, rgbs


def _search_world(native, slot, config, rng, library_path, rejected):
    visited = set()
    for _ in range(SITES_PER_WORLD):
        x, z = map(int, rng.integers(-24000, 24001, size=2))
        hit, _, _ = native.command("search", x=x, z=z, biome_groups=BIOMES,
            biome_group=slot["biome"], resolution=64, max_checks=16384)
        if hit["status"] != "found":
            rejected[hit["status"]] += 1
            continue
        site = tuple(hit["site"])
        if site in visited:
            continue
        visited.add(site)
        try:
            survey, low = native.survey(site)
            anchors = site_candidates(survey, native.definitions, slot["height"], rng)
            for relative, surface, kind in anchors:
                tx, ty, tz = relative
                surveyed_region = survey[tx - 24:tx + 24, tz - 24:tz + 24, ty - 24:ty + 24]
                try:
                    scene_visibility(surveyed_region, native.definitions)
                except RejectedScene as exc:
                    rejected[str(exc)] += 1
                    continue
                foot = tuple(int(a + b) for a, b in zip(relative, low, strict=True))
                check, _, _ = native.command("anchor", foot_xyz=foot,
                    surface_y=surface, anchor_kind=kind)
                if not check["valid"] or biome_group(check["biome"], BIOMES) != slot["biome"]:
                    rejected["native anchor or biome mismatch"] += 1
                    continue
                try:
                    meta, vocabulary, arrays, rgbs = native.acquire(foot, kind, config, rng, library_path)
                except RejectedScene as exc:
                    rejected[str(exc)] += 1
                    continue
                meta.update(surface_y=surface, native_biome=check["biome"])
                return meta, vocabulary, arrays, rgbs
            rejected["no accepted anchor at site"] += 1
        except RejectedScene as exc:
            rejected[str(exc)] += 1
    return None


def capture_scene(slot, config, craftium_dir, library_path):
    """Allocate disjoint deterministic seed candidates and never substitute a quota."""
    craftium_dir = Path(craftium_dir).resolve()
    engine_build, game_build = _builds(craftium_dir)
    rejected = Counter()
    for attempt in range(WORLDS_PER_SLOT):
        ordinal = slot["scene_id"] * WORLDS_PER_SLOT + attempt
        # Odd multiplier permutes the 31-bit seed domain, so slots cannot share worlds.
        seed = (2654435761 * ordinal + config["seed"]) & 0x7fffffff
        rng = np.random.default_rng(seed)
        with RenderSession(
            "procedural", seed, config["renderer"], craftium_dir,
            scene_script=HERE / "scene.lua",
            scene_assets=(REUSED_LUA / "biome_search.lua", REUSED_LUA / "scene_geometry.lua"),
            native_settings=SETUP, image_size=(config["width"], config["height"]),
            fixed_step=True, voxel_observation=False, render_rgb=True,
        ) as session:
            native = NativeScene(session)
            try:
                result = _search_world(native, slot, config, rng, library_path, rejected)
            except (ConnectionError, TimeoutError) as exc:
                # A late response belongs to the discarded world, not the next request.
                rejected[str(exc)] += 1
                continue
            if result is not None:
                meta, vocabulary, arrays, rgbs = result
                meta.update(slot, world_seed=seed, engine_build=engine_build, game_build=game_build)
                return meta, vocabulary, arrays, rgbs
    raise RejectedScene(f"Scene {slot['scene_id']} ({slot['biome']}/{slot['height']}) exhausted its search: {dict(rejected)}")
