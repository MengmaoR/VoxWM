"""Fixed24 clock contract and bulk read-only native encoding; no wall-clock pacing."""
import time
import numpy as np
from vae_p1.backend import Probe
from vae_p0.sampling import State
from .protocol import SIMULATION_HZ

PATCH_VERSION='voxwm-fixed24-v1'
RAW=np.dtype([('node','<u2'),('light','u1'),('param2','u1')])

class RegionEncoder:
    """Persistent content-ID/param2 lookup into the unchanged append-only vocabulary."""
    def __init__(self,vocabulary,names):
        self.v=vocabulary;self.names=names
        self.lookup=np.full((max(names)+1,256),65536,np.uint32)
    def encode(self,cells):
        node=cells['node'];param=cells['param2'];tokens=self.lookup[node,param]
        missing=tokens==65536
        if missing.any():
            keys=np.unique(node[missing].astype(np.uint32)*256+param[missing])
            for key in keys:
                n,p=divmod(int(key),256);self.lookup[n,p]=self.v.add(self.names[n],p)
            tokens=self.lookup[node,param]
        return tokens.astype(np.uint16)

class TickProbe(Probe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.encoder=RegionEncoder(self.v,self.names)
        self.communication_seconds=0.;self.step_wall_seconds=0.
    def command(self,kind,**fields):
        # Same command/response protocol; the completion barrier makes a capture
        # request finish in exactly one native tick. Async map emergence is setup only.
        import json
        self.check();self.seq+=1;path=self.s.world/'pilot_command.json';tmp=path.with_suffix('.tmp')
        start=time.perf_counter();tmp.write_text(json.dumps(dict(request_id=self.seq,kind=kind,**fields)));tmp.replace(path)
        self.communication_seconds+=time.perf_counter()-start
        for count in range(24000):
            start=time.perf_counter();self.step();self.step_wall_seconds+=time.perf_counter()-start
            start=time.perf_counter();response=self.s.world/'pilot_response.json'
            result=json.loads(response.read_text())if response.is_file()else{}
            self.communication_seconds+=time.perf_counter()-start
            if result.get('request_id')==self.seq:
                if kind=='advance'and count:raise ValueError('Native advance used multiple physical ticks')
                result['command_ticks']=count+1;return result
        raise RuntimeError('Native setup request did not complete')
    def decode_capture(self,result):
        raw=(self.s.world/'native48.bin').read_bytes();chunks=result['chunks']
        if len(raw)!=27*4096*RAW.itemsize:raise ValueError('Incomplete native region')
        cells=np.frombuffer(raw,RAW).reshape(27,16,16,16);tokens=self.encoder.encode(cells);states={}
        for i,info in enumerate(chunks):
            if info['missing']or info['mismatches']:raise ValueError('Incomplete/mismatched independent VoxelManip reference')
            raw_origin=np.asarray(info['origin'])
            if raw_origin.shape!=(3,)or raw_origin.dtype.kind not in'iuf'or not np.all(np.isfinite(raw_origin)&(raw_origin==np.floor(raw_origin))):raise ValueError('Native sampling origin is not integer-valued')
            origin=tuple(raw_origin.astype(int).tolist())
            meta=dict(world_seed=self.seed,chunk_origin=list(origin),surface_biome=info['surface_biome'],stratum=info['stratum'],
                data_source='native',sim_time=result['sim_time'],simulation_step=result['simulation_step'],engine_us=result['engine_us'])
            states[origin]=State(tokens[i],cells[i]['light'].copy(),meta)
        return states

def verify_clock(result,client_info,previous=None):
    c=result['native_clock'];tick=int(c['server_tick'])
    if c['patch_version']!=PATCH_VERSION:raise ValueError('Rebuild the fixed24 Craftium patch')
    if tick!=int(c['client_request_tick'])or tick!=client_info['native_client_tick']or tick!=int(result['lua_tick']):
        raise ValueError('Server/client/Lua clock mismatch: '+str(dict(server=tick,request=c['client_request_tick'],client=client_info['native_client_tick'],lua=result['lua_tick'])))
    if abs(c['server_dt']-1/SIMULATION_HZ)>1e-8 or abs(client_info['native_client_dt']-1/SIMULATION_HZ)>1e-8:raise ValueError('Non-fixed native dt')
    if abs(result['sim_time']-tick/SIMULATION_HZ)>1e-6:raise ValueError('Native absolute clock drift')
    if previous is not None and tick!=previous+1:raise ValueError('Missing/extra native tick between captures')
    return tick
