"""Round-trip export/time/coordinate contracts and finite native dispatch, no engine."""
import base64,json,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from vae_data.sharded.planning.fixtures import fixture
from vae_data.sharded.trace import Trace,TraceReader
from vae_data.sharded.protocol import Frame
from vae_data.sharded.export_trace import one_trace
from voxwm_gt.storage import DeltaReader

class ViewerTests(unittest.TestCase):
    def test_export_preserves_all_observed_indices_voxels_and_real_times(self):
        with tempfile.TemporaryDirectory()as tmp:
            f,v=fixture();f.meta['data_source']='native'
            trace=Trace(Path(tmp)/'raw',v,'D1');trace.observe(f,'before_intervention')
            second=Frame(f.tokens.copy(),f.light.copy(),dict(f.meta,simulation_step=17,sim_time=2.75));second.tokens[20,20,20]=v.add('mcl_core:water_source',0)
            trace.observe(second,'after_intervention');trace.finish(status={'status':'fixture_export_test'})
            target=Path(tmp)/'view';summary=one_trace(trace.root,target);reader=DeltaReader(target/'voxels.bin')
            self.assertEqual(reader.frames,2);np.testing.assert_array_equal(reader.frame(1),second.tokens)
            self.assertEqual([r['metadata']['sim_time']for r in summary['frames']],[0.,2.75])
            source=Path(__file__).parents[1]/'viewer/stream.js'
            # Exercise the actual browser decoder and timeline module, with no renderer.
            module=base64.b64encode(source.read_bytes()).decode()
            code="""import fs from 'node:fs';const {VoxelStream,timeIndex,worldToLocal}=await import('data:text/javascript;base64,'+process.argv[1]);
const buffer=fs.readFileSync(process.argv[2]);const stream=new VoxelStream(buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength));
const rows=[{metadata:{sim_time:0}},{metadata:{sim_time:2.75}}];
if(timeIndex(rows,2.5)!==0||timeIndex(rows,2.75)!==1)throw Error('Real-time indexing');
if(stream.frame(1)[(20*48+20)*48+20]!==Number(process.argv[3]))throw Error('ABV1 frame');
if(JSON.stringify(worldToLocal([4,4,4],[-16,-16,-16]))!==JSON.stringify([-3.5,-3.5,3.5]))throw Error('ENU mapping');
"""
            subprocess.run(['node','--input-type=module','-e',code,module,str(target/'voxels.bin'),str(int(second.tokens[20,20,20]))],check=True,capture_output=True)
    def test_native_six_stops_each_type_after_one_success_and_three_attempts(self):
        from vae_data.sharded.native_six import run,TYPES
        class World:
            def __init__(self,*args):self.v=args[1];self.center=[0,0,0];self.surface_y=0
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def prepare_at(self,*args):return None,None
        class Trace:
            def __init__(self,*a):self.stream=SimpleNamespace(closed=False)
            def finish(self,*a,**k):self.stream.closed=True
        for success in(True,False):
            with tempfile.TemporaryDirectory()as tmp:
                args=SimpleNamespace(dataset=Path(tmp)/'six',wall_seconds=900,attempts_per_type=3,seed=42,craftium_dir='unused',renderer='auto')
                def collect(*a,**kw):return(object()if success else None,dict(status='success'if success else'rejected'))
                with patch('vae_data.sharded.native_six.Native',World),patch('vae_data.sharded.native_six.Trace',Trace),patch('vae_data.sharded.native_six.prepare_dynamic',return_value=(None,None)),patch('vae_data.sharded.native_six.collect',side_effect=collect)as collect_mock,patch('vae_data.sharded.native_six.save_example',return_value='fixture_dispatch'),patch('builtins.print'):
                    r=run(args)
                self.assertEqual(collect_mock.call_count,6 if success else 18)
                self.assertEqual(set(r['successes']),set(TYPES));self.assertTrue(all(n==(1 if success else 0)for n in r['successes'].values()))

if __name__=='__main__':unittest.main()
