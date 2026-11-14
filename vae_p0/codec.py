"""One append-only ID for (stable Node Name, raw param2). Light is not a token input."""
import json
from pathlib import Path
import numpy as np

BUILD=dict(craftium='ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49',voxelibre='178c71b1e7d6ec35c0d9c481c55eb306d6f330bc')

class Vocabulary:
    def __init__(self,path=None):
        self.entries=[];self.index={};self.checked_voxels=0;self.limit=65536
        if path:
            source=json.loads(Path(path).read_text())
            for e in source['tokens']:
                if e['id']!=len(self.entries):raise ValueError('Noncontiguous vocabulary')
                self.add(e['node_name'],e['param2'])

    def add(self,name,param2):
        pair=(str(name),int(param2))
        if not 0<=pair[1]<=255:raise ValueError('param2 must retain a raw uint8 value')
        if pair not in self.index:
            if len(self.entries)>=self.limit:raise OverflowError('Combined vocabulary exceeds uint16')
            self.index[pair]=len(self.entries);self.entries.append(pair)
        return self.index[pair]

    def encode(self,nodes,param2,native_names):
        nodes=np.asarray(nodes);param2=np.asarray(param2)
        if nodes.shape!=(16,16,16) or param2.shape!=nodes.shape:raise ValueError('Expected aligned 16-cubed arrays')
        pairs, inverse=np.unique(np.stack((nodes.ravel(),param2.ravel()),axis=1),axis=0,return_inverse=True)
        ids=[]
        for node,state in pairs:
            if int(node) not in native_names:raise KeyError(f'Unresolved native content ID {node}')
            ids.append(self.add(native_names[int(node)],int(state)))
        result=np.asarray(ids,np.uint16)[inverse].reshape(nodes.shape)
        # Every observed voxel is checked against the original native pair, not a predicted value.
        decoded_names,decoded_param=self.decode(result)
        expected=np.asarray([native_names[int(i)] for i in nodes.ravel()],dtype=object).reshape(nodes.shape)
        if not np.array_equal(decoded_names,expected) or not np.array_equal(decoded_param,param2):raise ValueError('Combined-token round trip failed')
        self.checked_voxels+=nodes.size
        return result

    def decode(self,tokens):
        a=np.asarray(tokens)
        if a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=len(self.entries):raise ValueError('Unknown Combined Token')
        names=np.asarray([e[0] for e in self.entries],object)
        params=np.asarray([e[1] for e in self.entries],np.uint8)
        return names[a],params[a]

    def mask(self,tokens,kind):
        names=np.asarray([e[0] for e in self.entries],object)
        if kind=='D1':v=np.array(['water' in n and ('source' in n or 'flowing' in n) for n in names])
        elif kind=='D2':v=np.array(['lava' in n and ('source' in n or 'flowing' in n) for n in names])
        elif kind=='D3':v=np.array([n.startswith('mcl_fire:') and 'fire' in n for n in names])
        elif kind=='D4':v=np.array([n=='mcl_core:vine' or n.startswith('mcl_bamboo:bamboo') and not any(x in n for x in ('block','plank','mosaic','shoot')) for n in names])
        else:raise ValueError(kind)
        return v[tokens]

    def save(self,path):
        Path(path).write_text(json.dumps(dict(version=1,build=BUILD,representation='VocabularyID(Node Name, raw param2)',
            light_in_token=False,K=len(self.entries),tokens=[dict(id=i,node_name=n,param2=p) for i,(n,p) in enumerate(self.entries)]),indent=2)+'\n')
