"""Native GT capture helpers extracted from the completed Test A environment."""
import os
import subprocess
import sys
import numpy as np
from .runtime import InfrastructureError,stop_process

VOXEL_RADIUS = 24
FPS = 24

class VideoWriter:
    def __init__(self, path, fps=FPS):
        import av
        self.av = av
        self.container = av.open(str(path), mode='w')
        self.stream = self.container.add_stream('libx264', rate=fps)
        self.stream.width, self.stream.height = 640, 360
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '18', 'preset': 'veryfast'}

    def append(self, rgb):
        frame = self.av.VideoFrame.from_ndarray(np.asarray(rgb, np.uint8), format='rgb24')
        for packet in self.stream.encode(frame): self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode(): self.container.mux(packet)
        self.container.close()


def parse_nodes(raw):
    """Accept Lua JSON array rows or an id→record object; Encoder only needs content_id→name."""
    if isinstance(raw, list):
        nodes = {}
        for row in raw:
            ident, name, walkable, water, liquid = row[:5]
            nodes[str(int(ident))] = dict(name=str(name), walkable=bool(walkable),
                                          water=bool(water), liquidtype=str(liquid))
        return nodes
    if not isinstance(raw, dict):
        raise ValueError(f'Engine node dump is {type(raw).__name__}, expected object or array')
    nodes = {}
    for key, rec in raw.items():
        if not isinstance(rec, dict):
            raise ValueError(f'Engine node dump[{key}] is {type(rec).__name__}, expected object')
        nodes[str(int(key))] = dict(name=str(rec.get('name', '')), walkable=bool(rec.get('walkable')),
                                    water=bool(rec.get('water')),
                                    liquidtype=str(rec.get('liquidtype') or 'none'))
    return nodes


def camera_record(info):
    """Use the center paired with this voxel observation, already ENU via the wrapper."""
    center = np.asarray(info['voxel_obs_center'], np.float64)
    origin = center - VOXEL_RADIUS - .5
    local = (np.asarray(info['cam_pos'], np.float64) - (center-.5)) / 48.
    forward = np.asarray(info['cam_dir'], np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(-np.array([0., 0., 1.]), forward)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    norms = np.linalg.norm(rotation, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-8).any():
        raise ValueError('Degenerate native camera direction')
    rotation /= norms
    camera = np.r_[rotation[:2].reshape(6), local, info['cam_fov_x']].astype(np.float32)
    return camera, origin


def renderer_config(mode, platform=None):
    platform = platform or sys.platform
    renderer = ('software' if platform.startswith('linux') else 'window') if mode == 'auto' else mode
    if renderer == 'software' and not platform.startswith('linux'):
        raise ValueError('Mesa software rendering requires Linux; use --renderer window on macOS')
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ''
    if renderer == 'software':
        environment.update(SDL_VIDEODRIVER='x11', LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe')
        environment.setdefault('LP_NUM_THREADS','1')
    else:
        environment.pop('SDL_VIDEODRIVER', None)
        environment.pop('LIBGL_ALWAYS_SOFTWARE', None)
        environment.pop('GALLIUM_DRIVER', None)
        # XQuartz DISPLAY gives Irrlicht an OpenGL 1.x context without GLSL.
        if platform == 'darwin':
            environment.pop('DISPLAY', None)
            environment['SDL_VIDEODRIVER'] = 'cocoa'
            # Luanti still constructs OpenAL; dummy/null avoids macOS alcDestroyContext hang on exit.
            environment['SDL_AUDIODRIVER'] = 'dummy'
            environment['ALSOFT_DRIVERS'] = 'null'
    return renderer, environment


def connect_engine(channel,engine,opener):
    """Short connection polls notice an exited engine instead of waiting 60s."""
    import time
    original=channel.listen_timeout;deadline=time.monotonic()+original/1000
    try:
        while True:
            if engine.proc is not None and engine.proc.poll()is not None:raise RuntimeError('Native engine exited during startup')
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Native engine connection timed out')
            channel.listen_timeout=max(1,min(250,int(remaining*1000)))
            try:return opener()
            except ConnectionError as exc:
                if 'listen timeout'not in str(exc).lower():raise
    finally:channel.listen_timeout=original


def startup_error(engine,exc):
    """Classify observed display failures, never silently retry a full display."""
    from pathlib import Path
    path=Path(engine.run_dir)/'stderr.txt'
    text=path.read_text(errors='replace')[-8000:]if path.exists()else''
    signals=('maximum number of clients reached','unable to initialize sdl','x11 not available','invalid window','could not open display','cannot open display')
    if any(part in text.lower()for part in signals):return InfrastructureError('Native display startup failed: '+text[-3000:])
    return exc


def close_env(env):
    """Close sockets/pipes and the process even if the initial connection failed."""
    mt=env.unwrapped.mt;channel=env.unwrapped.mt_chann;proc=mt.proc;connected=channel.is_open()
    try:
        if connected:
            try:channel.send_kill()
            except OSError:pass
        try:channel.close()
        except OSError:
            try:os.close(channel.sockfd)
            except OSError:pass
        if connected and proc is not None and proc.poll()is None:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:pass
    finally:
        try:stop_process(proc)
        finally:
            mt.close_pipes()
            try:mt.clear()
            except OSError:pass


def water_signature(voxel, origin, classes):
    water = np.asarray([c['water'] for c in classes], bool)[voxel]
    points = np.argwhere(water)
    # Fixed physical water cells, unaffected by B's local crop shifts. Include native liquid param2.
    world = np.rint(origin + .5).astype(np.int64) + points
    rows = np.column_stack((world, voxel[water]))
    if len(rows): rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    return rows.tobytes(), len(points)
