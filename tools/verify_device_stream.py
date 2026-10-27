"""Compare the overlap cache with full embedding inference across rolling windows and resets."""
from pathlib import Path
import ctypes,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/device-build'
report=json.loads((OUT/'models.json').read_text())
lib=ctypes.CDLL(str(OUT/('models.dll' if sys.platform=='win32' else 'libmodels.so')))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
for name in ['embedding_eval','embedding_stream_eval']:
    fn=getattr(lib,name);fn.argtypes=[fp,fp,fp];fn.restype=None
lib.embedding_stream_reset.argtypes=[];lib.embedding_stream_reset.restype=None
arena=np.empty(report['shared_arena_bytes']//4,np.float32)
actual=np.empty((1,1,1,96),np.float32);expected=np.empty_like(actual)
rng=np.random.default_rng(173);rows=[]
for scale in [0,1,3]:
    window=(rng.normal(size=(1,76,32,1))*scale).astype(np.float32)
    for frame in range(32):
        reset=frame in [0,17]
        if reset:
            window=(rng.normal(size=window.shape)*scale).astype(np.float32)
            lib.embedding_stream_reset()
        else:
            window[:,:68]=window[:,8:].copy()
            window[:,68:]=(rng.normal(size=(1,8,32,1))*scale).astype(np.float32)
        lib.embedding_eval(window,expected,arena)
        lib.embedding_stream_eval(window,actual,arena)
        error=float(np.max(np.abs(actual-expected)))
        passed=bool(np.allclose(actual,expected,atol=0.0001,rtol=0.0001))
        rows.append({'scale':scale,'frame':frame,'reset':reset,'max_abs_error':error,'passed':passed})
summary={'all_checks_passed':all(r['passed'] for r in rows),'comparisons':len(rows),
         'max_abs_error':max(r['max_abs_error'] for r in rows),'checks':rows,
         'cache_bytes':report['models']['embedding']['stream_cache_bytes']}
(OUT/'stream-verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
raise SystemExit(0 if summary['all_checks_passed'] else 1)
