"""Node/param2 encoding. Known classes retain their historical ids; new ones append."""
from pathlib import Path
import numpy as np
from .storage import class_metadata, read_json

PALETTE = Path(__file__).parent / 'assets/voxel_classes.json'

class Encoder:
    def __init__(self, nodes):
        self.nodes = nodes
        self.classdict = read_json(PALETTE)
        entries = self.classdict['node_classes']
        size = max(max(v[0] for v in entries.values()), max(map(int, nodes))) + 1
        self.lookup = np.full((size, 256), -1, np.int32)
        for key, (node, param) in entries.items():
            self.lookup[node, param] = int(key)
        self.classes = class_metadata(self.classdict, nodes)

    def __call__(self, raw):
        pairs = np.asarray(raw)[:48, :48, :48]
        if pairs.shape != (48,48,48,2):
            raise ValueError(f'Expected 49^3 native observation right-cropped to 48^3, got {pairs.shape}')
        node, param = pairs[...,0].astype(int), pairs[...,1].astype(int)
        if node.min() < 0 or node.max() >= len(self.lookup) or param.min() < 0 or param.max() > 255:
            raise ValueError('Unregistered native node/param2')
        result = self.lookup[node, param]
        if (result < 0).any():
            for n,p in np.unique(pairs[result < 0].reshape(-1,2), axis=0):
                ident = len(self.classdict['node_classes'])
                if ident > 65535: raise ValueError('Voxel class palette exceeds uint16')
                self.classdict['node_classes'][str(ident)] = [int(n),int(p)]
                self.lookup[n,p] = ident
            self.classes = class_metadata(self.classdict, self.nodes)
            result = self.lookup[node,param]
        return result.astype(np.uint16)
