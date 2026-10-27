"""Ahead-of-time export of the three pinned float32 graphs; no runtime converter on device."""
from pathlib import Path
import json,hashlib,math,argparse
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from ai_edge_litert import schema_py_generated as schema
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Compile a compatible float32 openWakeWord classifier into native C.')
parser.add_argument('--model',type=Path,default=ROOT/'eva.tflite')
parser.add_argument('--threshold',type=float,default=0.5)
args=parser.parse_args()
if not math.isfinite(args.threshold) or not 0<args.threshold<=1:parser.error('Threshold must be in (0,1].')
if not args.model.is_file():parser.error('Model file not found: '+str(args.model))
check=Interpreter(model_path=str(args.model),num_threads=1)
model_input=check.get_input_details();model_output=check.get_output_details()
if len(model_input)!=1 or list(model_input[0]['shape'])!=[1,16,96] or model_input[0]['dtype']!=np.float32:
    parser.error('Expected one float32 input [1,16,96], matching the openWakeWord embedding frontend.')
if len(model_output)!=1 or list(model_output[0]['shape'])!=[1,1] or model_output[0]['dtype']!=np.float32:
    parser.error('Expected one float32 probability output [1,1].')
OUT=ROOT/'analysis/device-build'
OUT.mkdir(parents=True,exist_ok=True)
weights=bytearray()
cache={}
reports={}

def array(values): return '(const int[]){'+','.join(map(str,values))+'}'
def d4(shape): return [1]*(4-len(shape))+list(shape)
def strides(shape): return [math.prod(shape[i+1:]) for i in range(len(shape))]
def options(op,kind):
    table=op.BuiltinOptions(); obj=getattr(schema,kind)(); obj.Init(table.Bytes,table.Pos); return obj

def compile_model(name,path,input_shape):
    blob=path.read_bytes()
    model=schema.Model.GetRootAsModel(blob,0)
    graph=model.Subgraphs(0)
    input_index=int(graph.Inputs(0))
    runtime=Interpreter(model_path=str(path),experimental_preserve_all_tensors=True,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,num_threads=1)
    runtime.resize_tensor_input(input_index,input_shape)
    runtime.allocate_tensors()
    runtime.set_tensor(input_index,np.zeros(input_shape,np.float32));runtime.invoke()
    details={t['index']:t for t in runtime.get_tensor_details()}
    constants={i for i in range(graph.TensorsLength()) if model.Buffers(graph.Tensors(i).Buffer()).DataLength()>0}
    aliases={}
    def base(i):
        while i in aliases:i=aliases[i]
        return i
    nodes=[]
    for index in range(graph.OperatorsLength()):
        op=graph.Operators(index)
        code=model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        inputs=[int(v) for v in op.InputsAsNumpy()]
        output=int(op.Outputs(0))
        if op.OutputsLength()!=1: raise ValueError('Multiple outputs unsupported')
        if code==77 or all(i<0 or base(i) in constants for i in inputs): constants.add(output);continue
        if code in (22,43,70):
            assert math.prod(details[output]['shape'])==math.prod(details[inputs[0]]['shape'])
            aliases[output]=base(inputs[0]);continue
        assert details[output]['dtype']==np.float32
        nodes.append((index,code,inputs,output,op))
    # Liveness allocation; views share an allocation and constants are read-only.
    end={input_index:-1}
    for position,(_,_,inputs,output,_) in enumerate(nodes):
        end[output]=position
        for i in inputs:
            if i>=0 and base(i) not in constants:end[base(i)]=position
    final=base(int(graph.Outputs(0)));end[final]=len(nodes)
    offsets={};active={};free=[];top=0
    def allocate(tensor,position):
        nonlocal top
        for old in list(active):
            if end[old]<position:
                free.append((offsets[old],active.pop(old)))
        # Coalesce adjacent released blocks.
        free.sort();merged=[]
        for offset,size in free:
            if merged and sum(merged[-1])==offset:merged[-1]=(merged[-1][0],merged[-1][1]+size)
            else:merged.append((offset,size))
        free[:]=merged
        size=(int(math.prod(details[tensor]['shape']))+3)&~3
        candidates=[(length,index) for index,(_,length) in enumerate(free) if length>=size]
        if candidates:
            _,j=min(candidates);offset,length=free.pop(j)
            if length>size:free.append((offset+size,length-size))
        else:offset=top;top+=size
        offsets[tensor]=offset;active[tensor]=size
    allocate(input_index,-1)
    for pos,(_,_,_,output,_) in enumerate(nodes):allocate(output,pos)
    def pointer(i):
        if i<0:return 'NULL'
        i=base(i)
        if i in constants:
            value=np.asarray(runtime.get_tensor(i))
            assert value.dtype==np.float32, (name,i,value.dtype)
            data=value.astype('<f4').tobytes()
            if data not in cache:
                weights.extend(b'\0'*(-len(weights)%16));cache[data]=len(weights);weights.extend(data)
            return f'(const float*)(model_weights+{cache[data]})'
        return f'(arena+{offsets[i]})'
    def shape(i):return list(map(int,details[i]['shape']))
    def const(i):assert base(i) in constants;return runtime.get_tensor(base(i))
    lines=[f'MODEL_API void {name}_eval(const float *input,float *output,float *arena) {{',f'memcpy(arena+{offsets[input_index]},input,{math.prod(input_shape)*4});']
    stream_calls=[];stream_caches=[];time_shift=8
    for _,code,inputs,out,op in nodes:
        a=inputs[0];s=shape(a);o=shape(out);n=math.prod(o)
        ap=pointer(a);yp=pointer(out)
        if code in (0,18,41,55,57,99):
            b=inputs[1];bs=d4(shape(b));sa=d4(s);od=d4(o)
            astr=[0 if dim==1 else st for dim,st in zip(sa,strides(sa))]
            bstr=[0 if dim==1 else st for dim,st in zip(bs,strides(bs))]
            act=0
            kinds={0:'AddOptions',18:'MulOptions',41:'SubOptions'}
            if code in kinds:act=options(op,kinds[code]).FusedActivationFunction()
            call=f'binary({ap},{pointer(b)},{yp},{code},{math.prod(s)},{math.prod(shape(b))},{n},{array(od)},{array(astr)},{array(bstr)},{act});'
        elif code in (14,73,76,98):
            alpha=options(op,'LeakyReluOptions').Alpha() if code==98 else 0
            call=f'unary({ap},{yp},{n},{code},{float(alpha).hex()}f);'
        elif code==9:
            cfg=options(op,'FullyConnectedOptions');w=shape(inputs[1]);assert len(w)==2
            assert not cfg.KeepNumDims() or math.prod(s)//w[1]==math.prod(o)//w[0]
            call=f'dense({ap},{pointer(inputs[1])},{pointer(inputs[2])},{yp},{n//w[0]},{w[1]},{w[0]},{cfg.FusedActivationFunction()});'
        elif code==3:
            cfg=options(op,'Conv2DOptions');w=shape(inputs[1]);assert len(s)==len(o)==len(w)==4 and w[3]==s[3]
            sh,sw=cfg.StrideH(),cfg.StrideW();dh,dw=cfg.DilationHFactor(),cfg.DilationWFactor()
            ph=max(0,(o[1]-1)*sh+(w[1]-1)*dh+1-s[1])//2 if cfg.Padding()==0 else 0
            pw=max(0,(o[2]-1)*sw+(w[2]-1)*dw+1-s[2])//2 if cfg.Padding()==0 else 0
            call=f'conv({ap},{pointer(inputs[1])},{pointer(inputs[2])},{yp},{array(s)},{array(o)},{w[1]},{w[2]},{sh},{sw},{dh},{dw},{ph},{pw},{cfg.FusedActivationFunction()});'
        elif code==17:
            cfg=options(op,'Pool2DOptions');kh,kw,sh,sw=cfg.FilterHeight(),cfg.FilterWidth(),cfg.StrideH(),cfg.StrideW()
            ph=max(0,(o[1]-1)*sh+kh-s[1])//2 if cfg.Padding()==0 else 0
            pw=max(0,(o[2]-1)*sw+kw-s[2])//2 if cfg.Padding()==0 else 0
            call=f'pool({ap},{yp},{array(s)},{array(o)},{kh},{kw},{sh},{sw},{ph},{pw},{cfg.FusedActivationFunction()});'
        elif code==39:
            perm=list(map(int,const(inputs[1]).reshape(-1)));steps=[strides(s)[i] for i in perm]
            call=f'transpose({ap},{yp},{n},{array(d4(o))},{array([0]*(4-len(steps))+steps)});'
        elif code in (40,82):
            axes={int(v)%len(s) for v in const(inputs[1]).reshape(-1)}
            reduced=[1 if i in axes else dim for i,dim in enumerate(s)]
            steps=[0 if i in axes else st for i,st in enumerate(strides(reduced))]
            call=f'reduce({ap},{yp},{math.prod(s)},{n},{array(d4(s))},{array([0]*(4-len(steps))+steps)},{int(code==82)});'
        elif code==34:
            padding=const(inputs[1]);assert len(s)==4
            call=f'pad({ap},{yp},{array(s)},{array(o)},{array(list(map(int,padding[:,0])))});'
        elif code==126:
            cfg=options(op,'BatchMatMulOptions');bs=shape(inputs[1])
            assert not cfg.AdjX() and not cfg.AdjY() and math.prod(bs[:-2])==1
            assert s[-1]==bs[-2]
            call=f'matmul({ap},{pointer(inputs[1])},{yp},{math.prod(s[:-2])},{s[-2]},{s[-1]},{bs[-1]});'
        else:raise ValueError(f'Unsupported runtime op {code} in {name}')
        stream_call=call
        if name=='embedding' and code==3:
            assert sh==1 and ph==0 and o[0]==1
            if o[1]>time_shift:
                cache_name=f'embedding_cache_{len(stream_caches)}'
                stream_caches.append((cache_name,n))
                stream_call=call.replace('conv(','conv_cached(',1)[:-2]+f',{cache_name},{time_shift},embedding_stream_valid);'
        if name=='embedding' and code==17:
            assert time_shift%sh==0 and ph==0
            time_shift//=sh
        stream_calls.append(stream_call)
        lines.append(call)
    lines += [f'memcpy(output,{pointer(final)},{math.prod(shape(final))*4});','}']
    reports[name]={'sha256':hashlib.sha256(blob).hexdigest(),'input_shape':input_shape,'output_shape':shape(final),'runtime_ops':len(nodes),'arena_bytes':top*4}
    if name=='embedding':
        reports[name]['stream_cache_bytes']=sum(n*4 for _,n in stream_caches)
        lines += ['static int embedding_stream_valid;']+[f'static float {key}[{n}];' for key,n in stream_caches]
        lines += ['MODEL_API void embedding_stream_reset(void) {embedding_stream_valid=0;}',
                  'MODEL_API void embedding_stream_eval(const float *input,float *output,float *arena) {',
                  f'memcpy(arena+{offsets[input_index]},input,{math.prod(input_shape)*4});']
        lines += stream_calls+[f'memcpy(output,{pointer(final)},{math.prod(shape(final))*4});','embedding_stream_valid=1;','}']
    return '\n'.join(lines)

generated=[]
for name,path,shape in [('mel',ROOT/'.cache/eva-engine/melspectrogram.tflite',[1,1760]),('embedding',ROOT/'.cache/eva-engine/embedding_model.tflite',[1,76,32,1]),('eva',args.model,[1,16,96])]:
    generated.append(compile_model(name,path,shape))
chunks=['"'+''.join('\\x%02x'%v for v in weights[i:i+256])+'"' for i in range(0,len(weights),256)]
header='#include "kernels.h"\n#ifdef _WIN32\n#define MODEL_API __declspec(dllexport)\n#else\n#define MODEL_API\n#endif\nstatic const unsigned char model_weights[] __attribute__((aligned(16))) =\n'+'\n'.join(chunks)+';\n'
(OUT/'models.c').write_text(header+'\n'.join(generated),encoding='ascii')
arena=max(r['arena_bytes'] for r in reports.values())
(OUT/'models.h').write_text('#pragma once\n#define MODEL_ARENA_BYTES '+str(arena)+f'\n#define MODEL_THRESHOLD {args.threshold.hex()}f\n#define MODEL_RELEASE_THRESHOLD {min(0.2,args.threshold*0.4).hex()}f\n'+'\n'.join(f'void {name}_eval(const float*,float*,float*);' for name in reports)+'\nvoid embedding_stream_reset(void);\nvoid embedding_stream_eval(const float*,float*,float*);\n',encoding='ascii')
report={'models':reports,'threshold':args.threshold,'weights_bytes':len(weights),'shared_arena_bytes':arena,'float32_graphs':True,'runtime':'generated C; no Python/TFLite needed on target'}
(OUT/'models.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
