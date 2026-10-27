"""Numerical comparison of compiled graphs and the original TFLite files."""
from pathlib import Path
import ctypes,json,argparse,sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,default=ROOT/'eva.tflite');args=parser.parse_args()
OUT=ROOT/'analysis/device-build'
report=json.loads((OUT/'models.json').read_text())
lib=ctypes.CDLL(str(OUT/('models.dll' if sys.platform=='win32' else 'libmodels.so')))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
arena=np.empty(report['shared_arena_bytes']//4,np.float32)
rng=np.random.default_rng(42)
results=[]
for name,filename in [('mel',ROOT/'.cache/eva-engine/melspectrogram.tflite'),('embedding',ROOT/'.cache/eva-engine/embedding_model.tflite'),('eva',args.model.resolve())]:
    info=report['models'][name]
    model=Interpreter(model_path=str(ROOT/filename),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    input_index=model.get_input_details()[0]['index']
    model.resize_tensor_input(input_index,info['input_shape']);model.allocate_tensors()
    fn=getattr(lib,name+'_eval');fn.argtypes=[fp,fp,fp];fn.restype=None
    for kind in ['zeros','random','large']:
        scale={'zeros':0,'random':(1000 if name=='mel' else 1),'large':(20000 if name=='mel' else 3)}[kind]
        data=(rng.normal(size=info['input_shape'])*scale).astype(np.float32)
        model.set_tensor(input_index,data);model.invoke()
        expected=model.get_tensor(model.get_output_details()[0]['index'])
        actual=np.empty(info['output_shape'],np.float32);fn(data,actual,arena)
        error=float(np.max(np.abs(actual-expected)))
        passed=bool(np.allclose(actual,expected,atol=0.002,rtol=0.002))
        row={'model':name,'case':kind,'max_abs_error':error,'passed':passed}
        results.append(row);print(row,flush=True)
summary={'checks':results,'all_checks_passed':all(r['passed'] for r in results)}
(OUT/'numerical-verification.json').write_text(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['all_checks_passed'] else 1)
