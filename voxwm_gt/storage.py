"""Categorical voxel streams: full first frame followed by lossless sparse changes."""
import json
import struct
from pathlib import Path
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class DeltaWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.initial = self.previous = None
        self.pointers = [0]
        self.indices, self.values = [], []

    def append(self, voxel):
        voxel = np.asarray(voxel)
        if voxel.ndim != 3 or voxel.dtype.kind not in 'iu' or voxel.min() < 0 or voxel.max() > 65535:
            raise ValueError('Expected nonnegative categorical 3D voxel, fitting uint16')
        if self.initial is None:
            self.initial = voxel.astype('<u2', copy=True)
        else:
            if voxel.shape != self.initial.shape:
                raise ValueError('Voxel shape changed')
            changed = np.flatnonzero(voxel.ravel() != self.previous.ravel()).astype('<u4')
            self.indices.append(changed)
            self.values.append(voxel.ravel()[changed].astype('<u2'))
        self.pointers.append(self.pointers[-1] + (len(self.indices[-1]) if self.previous is not None else 0))
        self.previous = voxel.astype('<u2', copy=True)

    def close(self):
        if self.initial is None:
            return
        frames = len(self.pointers) - 1
        with self.path.open('wb') as f:
            f.write(struct.pack('<4s4I', b'ABV1', frames, *self.initial.shape))
            self.initial.tofile(f)
            np.asarray(self.pointers, dtype='<u4').tofile(f)
            for a in self.indices: a.tofile(f)
            for a in self.values: a.tofile(f)


class DeltaReader:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            magic, self.frames, *shape = struct.unpack('<4s4I', f.read(20))
        if magic != b'ABV1' or not self.frames:
            raise ValueError('Not an ABV1 voxel stream')
        self.shape = tuple(shape)
        size = int(np.prod(shape))
        self.initial = np.memmap(path, mode='r', dtype='<u2', offset=20, shape=self.shape)
        offset = 20 + size * 2
        self.pointers = np.memmap(path, mode='r', dtype='<u4', offset=offset, shape=(self.frames + 1,))
        count = int(self.pointers[-1]); offset += (self.frames + 1) * 4
        if Path(path).stat().st_size != offset + count * 6:
            raise ValueError('Incomplete sparse voxel stream')
        self.indices = np.memmap(path, mode='r', dtype='<u4', offset=offset, shape=(count,)) if count else np.empty(0, np.uint32)
        self.values = np.memmap(path, mode='r', dtype='<u2', offset=offset+count*4, shape=(count,)) if count else np.empty(0, np.uint16)
        self.current, self.t = np.array(self.initial), 0

    def frame(self, t):
        if not 0 <= t < self.frames:
            raise IndexError(t)
        if t < self.t:
            self.current = np.array(self.initial); self.t = 0
        while self.t < t:
            self.t += 1
            a, b = self.pointers[self.t:self.t+2]
            self.current.ravel()[self.indices[a:b]] = self.values[a:b]
        return self.current.copy()

    def __iter__(self):
        for t in range(self.frames):
            yield self.frame(t)


def class_metadata(classdict, nodes):
    result = []
    for key, (node, param) in sorted(classdict['node_classes'].items(), key=lambda p: int(p[0])):
        definition = nodes.get(str(node), {})
        name = definition.get('name', f'node_{node}')
        water = definition.get('water', False)
        color = '#9a9386'
        if 'stone' in name or 'cobble' in name: color = '#8a9199'
        if 'dirt' in name: color = '#87633f'
        if 'grass' in name: color = '#699443'
        if 'tree' in name or 'wood' in name: color = '#926038'
        if 'leaves' in name: color = '#35784b'
        if 'sand' in name: color = '#d5c18c'
        if water: color = '#348cce'
        result.append(dict(id=int(key), node=node, param2=param, name=name, color=color,
                           water=bool(water), air=node == 126, ignore=node == 127,
                           solid=bool(definition.get('walkable', False)) and not water))
    return result
