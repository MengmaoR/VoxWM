"""Read native Mapgen v7 terrain and pair a stable voxel map with client RGB packets."""
from collections import Counter, deque
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
    BIOMES, NOMINAL_CAMERA_HEIGHT, SETTINGS, SURVEY_SIDE, VISIBILITY_LOOKUP_VERSION,
    VISIBILITY_RADIUS, V3_RGB, WORLD_BOTTOM, WORLD_TOP, RejectedScene, camera_angles,
    camera_local_xyz, direction_for_angles, region_id, require_air_camera, sample_views,
    scene_visibility, site_candidates, v3_lighting_images, v3_rgb,
)

HERE = Path(__file__).resolve().parent
REUSED_LUA = HERE.parent / "vae_data/sharded"
NOOP = np.zeros(4, dtype=np.int64)
SETUP = dict(SETTINGS, node_highlighting="none",
             performance_tradeoffs=False, mcl_logging_mobs_spawn=False)
WORLDS_PER_SLOT = 8
SITES_PER_WORLD = 4
PILOT_SCENES = 200
SEED_MULTIPLIER = 2654435761
SEED_SPACE = 2**31
SEED_MULTIPLIER_INVERSE = pow(SEED_MULTIPLIER, -1, SEED_SPACE)


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
        self.definitions = {}
        for row in self.packet["definitions"]:
            value = row["id"]
            # Luanti's JSON encoder writes Lua numbers as doubles, including IDs.
            if type(value) not in (int, float) or not 0 <= value <= 65535 or value != int(value):
                raise ValueError(f"Invalid native content ID for {row['name']}: {value!r}")
            runtime_id = int(value)
            if runtime_id in self.definitions:
                raise ValueError(f"Duplicate native content ID {runtime_id}")
            self.definitions[runtime_id] = dict(row, id=runtime_id)
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
        deadline = min(time.monotonic() + 90, getattr(self, "search_deadline", math.inf))
        while time.monotonic() < deadline:
            rgb, info = self.step()
            row = json.loads(response.read_text()) if response.is_file() else {}
            if row.get("request_id") == self.seq:
                try:
                    verify_clock(row, info)
                except ValueError as exc:
                    if not str(exc).startswith("Server/client/Lua clock mismatch:"):
                        raise
                    raise ConnectionError(f"Unaligned native tick during {kind}") from exc
                return row, rgb, info
        raise TimeoutError(f"Native {kind} exceeded its command or scene deadline")

    def survey(self, site, survey_columns=16, biome=None):
        x, z = site
        margin = 24 + (survey_columns - 16) // 2
        side = SURVEY_SIDE + survey_columns - 16
        low = (x - margin, WORLD_BOTTOM, z - margin)
        high = (low[0] + side - 1, WORLD_TOP, low[2] + side - 1)
        row, _, _ = self.command("survey", low_xyz=low, high_xyz=high, biome_group=biome,
                                  biome_groups=BIOMES if biome else None)
        if row["status"] != "surveyed":
            raise RejectedScene("Native site emergence failed")
        survey = np.fromfile(self.session.world / "vox2pix_survey.bin", dtype="<u2")
        return survey.reshape(side, side, WORLD_TOP - WORLD_BOTTOM + 1), low, row.get("cave_hints") or []

    def surface_biomes(self, positions):
        if not positions:
            return {}
        row, _, _ = self.command("surface_biomes", positions=positions)
        return dict(zip(positions, row["names"], strict=True))

    def geometry_snapshot(self, low, kind, boxes=False):
        row, rgb, info = self.command("snapshot", low_xyz=low, anchor_kind=kind, boxes=boxes)
        if row["status"] != "captured":
            raise RejectedScene(row.get("reason", "Native voxel snapshot failed"))
        if not row.get("rgb_enabled"):
            raise RuntimeError("Native RGB rendering is off at a saved capture")
        state = decode_region((self.session.world / "vox2pix_region.bin").read_bytes())
        ids = self.lookup[state["node"]]
        if (ids == 65535).any():
            raise RuntimeError("Native region contains an unregistered content ID")
        if "native_render_settings" in row:
            self.region_light = state["light"].copy()
        arrays = ids, state["param2"].copy()
        return arrays, row, rgb, info

    def sink_to_floor(self, spawn):
        """Observe the natural descent without positioning or inspecting a crop."""
        recent = deque(maxlen=12)
        while time.monotonic() < self.search_deadline:
            _, info = self.step()
            player = np.asarray(info["player_pos"], dtype=np.float64)
            recent.append(player)
            if (len(recent) == recent.maxlen and player[1] < spawn[1] - .005
                    and np.max(np.ptp(recent, axis=0)) <= .003):
                return player, int(info["native_client_tick"])
        raise TimeoutError("Underwater observer did not settle before the scene deadline")

    def renderer_lighting(self):
        path = self.session.world / "client_clock.txt.lighting"
        try:
            fields = path.read_text().split()
        except FileNotFoundError as exc:
            raise RuntimeError("The native renderer did not publish lighting-floor evidence") from exc
        tick, gamma, floor, zero, maximum, pending, version = fields
        return dict(render_tick=int(tick), display_gamma=float(gamma), lighting_floor=float(floor),
                    lut_zero=int(zero), lut_max=int(maximum), pending_meshes=int(pending), version=version)

    def renderer_material_lighting(self):
        path = self.session.world / "client_clock.txt.material_lighting"
        try:
            tick, gamma, floor, ambient, zero, maximum, pending, bound, version = path.read_text().split()
        except (OSError, ValueError) as exc:
            raise RuntimeError("The native renderer did not publish a valid material-lighting probe") from exc
        return dict(render_tick=int(tick), display_gamma=float(gamma), lighting_floor=float(floor),
                    solid_ambient_floor=float(ambient), lut_zero=int(zero), lut_max=int(maximum),
                    pending_meshes=int(pending), shader_bound=bound == "1", version=version)

    def check_light_frame(self, capture, floor):
        lighting = self.renderer_lighting()
        tick = int(capture["native_clock"]["server_tick"])
        expected_zero = math.floor(floor * 255 + .5)
        if (lighting["render_tick"] != tick - 1 or lighting["lighting_floor"] != floor
                or lighting["display_gamma"] != V3_RGB["display_gamma"]
                or lighting["lut_zero"] != expected_zero or lighting["lut_max"] != 255
                or lighting["pending_meshes"] != 0 or lighting["version"] != "voxwm-light-floor-v1"):
            raise RejectedScene(f"Client lighting floor or mesh did not settle for render tick {tick - 1}: {lighting}")
        if capture["wielded_item"] or not capture["hud_wielditem_hidden"]:
            raise RejectedScene("Native observer has a visible wield model or an item selected")
        return lighting

    def check_material_light_frame(self, light, ambient):
        probe = self.renderer_material_lighting()
        frame_keys = ("render_tick", "display_gamma", "lighting_floor", "lut_zero", "lut_max", "pending_meshes")
        if (probe["version"] != "voxwm-material-ambient-v1" or not probe["shader_bound"]
                or probe["solid_ambient_floor"] != ambient
                or any(probe[key] != light[key] for key in frame_keys)):
            raise RejectedScene(f"Material lighting does not match the captured RGB frame: {probe}")
        return probe

    def request_floor(self, floor, *, ambient=None):
        request = self.session.world / "client_clock.txt.lighting_request"
        pending = request.with_suffix(".tmp")
        pending.write_text(f"{floor}\n" if ambient is None else f"{floor} {ambient}\n")
        pending.replace(request)

    def wait_lighting(self, floor, *, ambient=None):
        while time.monotonic() < self.search_deadline:
            self.step()
            light = self.renderer_lighting()
            if light["lighting_floor"] != floor or light["pending_meshes"]:
                continue
            if ambient is not None:
                probe = self.renderer_material_lighting()
                if probe["solid_ambient_floor"] != ambient or probe["pending_meshes"]:
                    continue
            return
        raise TimeoutError("Native lighting remesh exceeded the scene deadline")

    def acquire(self, foot, kind, config, rng, library_path, scene_kind=None):
        rgb_settings = v3_rgb(config) if scene_kind else V3_RGB
        floor = rgb_settings["lighting_floor"]
        ambient = rgb_settings.get("solid_ambient_floor")
        initial = self.command("pose", relocate=True, foot_xyz=foot, anchor_kind=kind, yaw=0., pitch=0.)
        underwater_descent = None
        if scene_kind == "underwater":
            from .v3_sampling import settled_anchor

            spawn = foot
            settled, final_tick = self.sink_to_floor(spawn)
            foot = settled_anchor(settled)
            origin = tuple(coordinate - 24 for coordinate in foot)
            emerged, _, _ = self.command("emerge_capture", low_xyz=origin,
                                         high_xyz=tuple(coordinate + 47 for coordinate in origin))
            if emerged["status"] != "capture_emerged":
                raise RejectedScene("Final underwater 48³ region could not emerge")
            underwater_descent = dict(spawn_player_xyz=list(spawn), settled_player_xyz=settled.tolist(),
                                      ticks=final_tick - int(initial[2]["native_client_tick"]))
        else:
            origin = tuple(coordinate - 24 for coordinate in foot)
        for _ in range(72):
            self.step()
        if scene_kind:
            self.wait_lighting(floor, ambient=ambient)
        baseline, base_capture, _, base_info = self.geometry_snapshot(origin, kind, boxes=True)
        if scene_kind:
            from . import v3_sampling

            eye = tuple(np.asarray(base_info["cam_pos"]) - foot)
            if underwater_descent:
                support = base_capture["underwater_support"]
                native_xyz = support["support_xyz"]
                support_xyz = [int(coordinate) for coordinate in native_xyz]
                if support_xyz != native_xyz:
                    raise RejectedScene("Native support coordinate is outside the integer lattice")
                underwater_descent.update(support, support_xyz=support_xyz)
                center, full, active = v3_sampling.scene_visibility(baseline[0], self.stable_defs, scene_kind, eye)
            else:
                center, full, active = v3_sampling.scene_visibility(baseline[0], self.stable_defs, scene_kind)
            cave = v3_sampling.cave_evidence(baseline[0], self.stable_defs) if scene_kind == "cave" else None
        else:
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
            if scene_kind:
                self.wait_lighting(floor, ambient=ambient)
            previous, prior, _, _ = self.geometry_snapshot(origin, kind)
            current, capture, rgb, info = self.geometry_snapshot(origin, kind)
            tick = int(capture["native_clock"]["server_tick"])
            if tick != int(prior["native_clock"]["server_tick"]) + 1:
                raise RejectedScene("RGB was not bracketed by consecutive native voxel captures")
            if any(not np.array_equal(field, expected)
                   for state in (previous, current) for field, expected in zip(state, baseline)):
                raise RejectedScene("Node Name or raw param2 changed between native views")
            position = np.asarray(info["cam_pos"], dtype=np.float64)
            real_foot = np.asarray(info["player_pos"], dtype=np.float64) if scene_kind else foot
            eye_height = float(camera_local_xyz(position, real_foot if underwater_descent else foot)[1])
            projection_origin = position - foot
            if scene_kind:
                v3_sampling.require_camera(current[0], self.stable_defs, projection_origin,
                                           scene_kind, projector.water_contains)
                if capture["native_render_settings"] != rgb_settings:
                    raise RejectedScene("V3 native render settings differ from protocol")
                if underwater_descent and (np.max(np.abs(real_foot - settled)) > .003
                                           or np.max(np.abs(projection_origin - eye)) > .003):
                    raise RejectedScene("Underwater observer moved after settling")
                if kind == "surface_water" and np.max(np.abs(real_foot - foot)) > .003:
                    raise RejectedScene("Surface observer moved from its water-surface anchor")
                lighting = self.check_light_frame(capture, floor)
                view.update(renderer_lighting=lighting, lighting_floor=lighting["lighting_floor"],
                            display_gamma=lighting["display_gamma"], actual_player_world_xyz=real_foot.tolist(),
                            hud_wielditem_hidden=capture["hud_wielditem_hidden"], wielded_item=capture["wielded_item"],
                            camera_medium="water" if scene_kind == "underwater" else "air")
                if ambient is not None:
                    view.update(solid_ambient_floor=ambient,
                                material_renderer_lighting=self.check_material_light_frame(lighting, ambient))
                if rgb_settings.get("voxwm_exact_draw_range"):
                    from .fog_runtime import read_fog, validate_fog, validate_fog_absolute

                    probe = read_fog(self)
                    if "fog_end_distance" in rgb_settings:
                        view["renderer_fog"] = validate_fog_absolute(probe, tick=lighting["render_tick"])
                    else:
                        view["renderer_fog"] = validate_fog(
                            probe, ratio=rgb_settings["fog_start"], tick=lighting["render_tick"])
            else:
                require_air_camera(current[0], self.stable_defs, projection_origin)
            direction = np.asarray(info["cam_dir"], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm < 1e-10:
                raise RejectedScene("The client camera packet is invalid")
            direction /= norm
            yaw, pitch = camera_angles(direction)
            if (direction @ expected_direction < math.cos(math.radians(.04))
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
            render = projector.render_v3 if scene_kind else projector.render
            whites.append(render(projection_origin, direction, fov_x, fov_y, width, height))
            view.update(foot_anchor_world_xyz=list(foot), actual_camera_world_xyz=position.tolist(),
                        actual_camera_height_above_foot=eye_height,
                        nominal_camera_height=NOMINAL_CAMERA_HEIGHT,
                        projection_origin_local_xyz=projection_origin.tolist(),
                        visibility_radius=VISIBILITY_RADIUS, visibility_lookup_version=VISIBILITY_LOOKUP_VERSION,
                        cam_dir_xyz=direction.tolist(),
                        yaw_deg=yaw, pitch_deg=pitch, fov_x_rad=fov_x, fov_y_rad=fov_y,
                        resolution=[width, height], gamma=capture["gamma"],
                        world_time=capture["world_time"], weather=capture["weather"],
                        simulation_tick=tick, render_simulation_tick=tick - 1)
        arrays = dict(scene_node_id=baseline[0], scene_param2=baseline[1])
        fields = (("surface_node_id", "surface_param2", "surface_face_dir", "water_count") if scene_kind
                  else ("white_node_id", "white_param2", "white_face"))
        for name, values in zip(fields, zip(*whites), strict=True):
            arrays[name] = np.stack(values)
        meta = dict(
            data_source="native",
            foot_anchor_xyz=list(foot), region_origin_xyz=list(origin),
            region_origin_enu=[origin[0], origin[2], origin[1]], anchor_kind=kind,
            nominal_camera_height=NOMINAL_CAMERA_HEIGHT, visibility_radius=VISIBILITY_RADIUS,
            visibility_lookup_version=VISIBILITY_LOOKUP_VERSION,
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
        if not scene_kind:
            return meta, self.vocabulary, arrays, rgbs
        meta.update(render_settings=dict(meta["render_settings"], **rgb_settings),
                    native_render_settings=capture["native_render_settings"], cave_evidence=cave,
                    lighting_floor=floor, display_gamma=rgb_settings["display_gamma"],
                    settled_player_xyz=(settled.tolist() if underwater_descent else base_info["player_pos"].tolist()),
                    terrain_nodes=sorted(d["name"] for d in self.definitions.values() if d["terrain"] and d["walkable"]),
                    water_definitions={d["name"]: d for d in self.definitions.values() if d["water"]})
        if ambient is not None:
            meta.update(solid_ambient_floor=ambient, lighting_profile=config["lighting_profile"])
        if rgb_settings.get("voxwm_exact_draw_range"):
            from .protocol import v3_fog_geometry

            meta["fog_geometry"] = v3_fog_geometry(config)
        if underwater_descent:
            meta.update(underwater_descent=underwater_descent, visibility_camera_local_xyz=list(eye))
        meta["lighting_comparison"], auxiliary = self.lighting_comparison(
            origin, kind, baseline, views[-1], rgbs[-1], info, len(views) - 1,
            projector, whites[-1], config)
        return meta, self.vocabulary, arrays, rgbs, auxiliary

    def lighting_comparison(self, origin, kind, baseline, view, proposed_rgb, proposed_camera,
                            index, projector, proposed_white, config):
        settings = v3_rgb(config)
        ambient = settings.get("solid_ambient_floor")
        control_ambient = 0. if ambient is not None else None
        native_light = self.region_light.copy()
        camera_keys = ("cam_pos", "cam_dir", "cam_fov_x", "cam_fov_y")
        proposed_camera = {key: np.asarray(proposed_camera[key]).copy() for key in camera_keys}
        self.request_floor(0., ambient=control_ambient)
        try:
            self.wait_lighting(0., ambient=control_ambient)
            previous, prior, _, _ = self.geometry_snapshot(origin, kind)
            prior_light = self.region_light
            current, capture, control_rgb, control_camera = self.geometry_snapshot(origin, kind)
            tick = int(capture["native_clock"]["server_tick"]) - 1
            evidence = self.check_light_frame(capture, 0.)
            if control_ambient is not None:
                material_control = self.check_material_light_frame(evidence, control_ambient)
            if tick != int(prior["native_clock"]["server_tick"]):
                raise RejectedScene("Lighting control render was not bracketed by consecutive voxel snapshots")
            if any(not np.array_equal(data, reference)
                   for state in (previous, current) for data, reference in zip(state, baseline, strict=True)):
                raise RejectedScene("Node ID or param2 changed during lighting comparison")
            if not np.array_equal(native_light, self.region_light) or not np.array_equal(native_light, prior_light):
                raise RejectedScene("Native voxel param1 changed during lighting comparison")
            if any(not np.allclose(control_camera[key], proposed_camera[key], atol=1e-6, rtol=0)
                   for key in camera_keys):
                raise RejectedScene("Lighting control camera pose differs from the proposed frame")
            height, width, _ = control_rgb.shape
            control_white = projector.render_v3(view["projection_origin_local_xyz"], view["cam_dir_xyz"],
                                                view["fov_x_rad"], view["fov_y_rad"], width, height)
            if any(not np.array_equal(a, b) for a, b in zip(control_white, proposed_white, strict=True)):
                raise RejectedScene("White model changed during lighting comparison")
            control_rgb = control_rgb.copy()
        finally:
            self.request_floor(settings["lighting_floor"], ambient=ambient)
        keys = ("actual_camera_world_xyz", "cam_dir_xyz", "fov_x_rad", "fov_y_rad", "display_gamma",
                "lighting_floor", "renderer_lighting", "render_simulation_tick")
        proposed = {key: view[key] for key in keys}
        x, y, z = (math.floor(value + 24.5) for value in view["projection_origin_local_xyz"])
        proposed["native_eye_param1"] = int(native_light[x, z, y])
        control = dict(proposed, lighting_floor=0., renderer_lighting=evidence, render_simulation_tick=tick)
        if ambient is not None:
            control.update(solid_ambient_floor=control_ambient, material_renderer_lighting=material_control)
            proposed.update(solid_ambient_floor=ambient, material_renderer_lighting=view["material_renderer_lighting"])
        names = v3_lighting_images(config)
        return (dict(view_index=index, voxel_identity_unchanged=True, light_identity_unchanged=True,
                     white_identity_unchanged=True, control=control, proposed=proposed),
                dict(zip(names, (control_rgb, proposed_rgb), strict=True)))


def _search_world(native, slot, config, rng, library_path, rejected, sites_per_world,
                  survey_columns, land_priority, cave_hints, committed_scene):
    visited = set()
    hints = deque()
    for site_attempt in range(1, sites_per_world + 1):
        if committed_scene is not None and committed_scene.is_dir():
            return None
        while hints and hints[0] in visited:
            hints.popleft()
        from_cave = bool(hints) and site_attempt % 3 != 0
        if from_cave:
            site = hints.popleft()
        else:
            x, z = map(int, rng.integers(-24000, 24001, size=2))
            sites = {}
            if land_priority and slot["height"] != "surface" and site_attempt % 4 != 0:
                sites["origins"] = [[x, z], *rng.integers(-24000, 24001, size=(3, 2)).tolist()]
            hit, _, _ = native.command("search", x=x, z=z, biome_groups=BIOMES,
                biome_group=slot["biome"], resolution=64, max_checks=16384, **sites)
            if hit["status"] != "found":
                rejected[hit["status"]] += 1
                continue
            site = tuple(hit["site"])
        if site in visited:
            rejected["duplicate native site for seed"] += 1
            continue
        visited.add(site)
        try:
            survey, low, nearby = native.survey(site, survey_columns, biome=slot["biome"] if cave_hints else None)
            hints.extend(tuple(row) for row in nearby)
            anchors = site_candidates(survey, native.definitions, slot["height"], rng, rejected)
            biomes = None
            if land_priority and slot["height"] != "surface":
                anchors = list(anchors)
                columns = list(dict.fromkeys((int(x + low[0]), surface, int(z + low[2]))
                                             for (x, _, z), surface, _ in anchors))
                biomes = native.surface_biomes(columns)
            for relative, surface, kind in anchors:
                tx, ty, tz = relative
                foot = tuple(int(a + b) for a, b in zip(relative, low, strict=True))
                if biomes is not None and biome_group(biomes[foot[0], surface, foot[2]], BIOMES) != slot["biome"]:
                    rejected["native surface biome mismatch"] += 1
                    continue
                surveyed_region = survey[tx - 24:tx + 24, tz - 24:tz + 24, ty - 24:ty + 24]
                try:
                    scene_visibility(surveyed_region, native.definitions)
                except RejectedScene as exc:
                    rejected[str(exc)] += 1
                    continue
                check, _, _ = native.command("anchor", foot_xyz=foot,
                    surface_y=surface, anchor_kind=kind)
                if biome_group(check["biome"], BIOMES) != slot["biome"]:
                    rejected["native surface biome mismatch"] += 1
                    continue
                if not check["valid"]:
                    rejected["native standing anchor mismatch"] += 1
                    continue
                try:
                    meta, vocabulary, arrays, rgbs = native.acquire(foot, kind, config, rng, library_path)
                except RejectedScene as exc:
                    rejected[str(exc)] += 1
                    continue
                meta.update(surface_y=surface, native_biome=check["biome"], site_attempt=site_attempt,
                            land_search_source="native_cave_hint" if from_cave else "biome_search")
                return meta, vocabulary, arrays, rgbs
            rejected["no accepted anchor at site"] += 1
        except RejectedScene as exc:
            rejected[str(exc)] += 1
    return None


def scene_world_seed(scene_id, pilot_seed, attempt):
    # Round zero retains the original mapping. Each later round owns a fresh
    # range of the same permutation, with eight candidates for every slot.
    search_round, index = divmod(attempt, WORLDS_PER_SLOT)
    ordinal = (search_round * PILOT_SCENES + scene_id) * WORLDS_PER_SLOT + index
    return (SEED_MULTIPLIER * ordinal + pilot_seed) % SEED_SPACE


def scene_id_for_world_seed(world_seed, pilot_seed):
    ordinal = ((world_seed - pilot_seed) * SEED_MULTIPLIER_INVERSE) % SEED_SPACE
    return (ordinal // WORLDS_PER_SLOT) % PILOT_SCENES


def capture_scene(slot, config, craftium_dir, library_path, search_round=0, search_rounds=1,
                  sites_per_world=SITES_PER_WORLD, survey_columns=16, land_priority=False,
                  cave_hints=False, committed_scene=None):
    """Allocate disjoint deterministic seed candidates and never substitute a quota."""
    craftium_dir = Path(craftium_dir).resolve()
    engine_build, game_build = _builds(craftium_dir)
    rejected = Counter()
    first = search_round * WORLDS_PER_SLOT
    limit = first + search_rounds * WORLDS_PER_SLOT
    for attempt in range(first, limit):
        if committed_scene is not None and committed_scene.is_dir():
            return None
        current_round = attempt // WORLDS_PER_SLOT
        seed = scene_world_seed(slot["scene_id"], config["seed"], attempt)
        rng = np.random.default_rng(seed)
        try:
            with RenderSession(
                "procedural", seed, config["renderer"], craftium_dir,
                scene_script=HERE / "scene.lua",
                scene_assets=(REUSED_LUA / "biome_search.lua", REUSED_LUA / "scene_geometry.lua"),
                native_settings=dict(SETUP, voxwm_skip_rgb=True, vox2pix_cave_hints=cave_hints),
                image_size=(config["width"], config["height"]),
                fixed_step=True, voxel_observation=False, render_rgb=True,
            ) as session:
                native = NativeScene(session)
                result = _search_world(native, slot, config, rng, library_path, rejected,
                                       sites_per_world, survey_columns, land_priority, cave_hints, committed_scene)
        except (ConnectionError, TimeoutError) as exc:
            # A late response belongs to the discarded world, not the next request.
            rejected[str(exc)] += 1
            result = None
        if result is not None:
            meta, vocabulary, arrays, rgbs = result
            meta.update(slot, world_seed=seed, engine_build=engine_build, game_build=game_build,
                        search_round=current_round, world_attempt=attempt + 1,
                        survey_columns=survey_columns, land_priority=land_priority,
                        search_rejections=dict(rejected))
            return meta, vocabulary, arrays, rgbs
        if (attempt + 1) % WORLDS_PER_SLOT == 0 and attempt + 1 < limit:
            print(f"Search round {current_round} exhausted for scene {slot['scene_id']}: {dict(rejected)}", flush=True)
    if committed_scene is not None and committed_scene.is_dir():
        return None
    raise RejectedScene(f"Scene {slot['scene_id']} ({slot['biome']}/{slot['height']}) exhausted its search: {dict(rejected)}")
