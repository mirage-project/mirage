"""Test production QK scores against native BF16 matmul on saved and random inputs."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import torch
from test_batch_model_gpu import compare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--production-only',action='store_true')
    p.add_argument('--all-short-batches',action='store_true')
    p.add_argument('--dispatch-probes',action='store_true')
    p.add_argument('--tile-probes',action='store_true')
    p.add_argument('--gemv-probes',action='store_true')
    args=p.parse_args()
    if args.all_short_batches: args.dispatch_probes = True
    args.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
    saved=torch.load(args.snapshot,weights_only=True,map_location='cpu')
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    assert uuid.lower().removeprefix('gpu-')==os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32=False
    repo=Path(__file__).resolve().parents[2]
    library=args.output/'scores.so'
    command=[str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_100a',
             '-DMIRAGE_BACKEND_USE_CUDA','--shared','-Xcompiler','-fPIC',
             *['-I'+str(repo/n) for n in ('include','include/mirage/persistent_kernel','deps/cutlass/include','deps/json/include')],
             str(repo/'tests/qwen3_5/blackwell_scores.cu'),'-o',str(library)]
    with (args.output/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    lib=ctypes.CDLL(str(library));lib.launch_scores.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2
    checks=[];probability_checks=[];profiles=[]
    def check(label,q,k,captured=None):
        batch,length=k.shape[:2]
        expected=(q.reshape(batch,24,1,256)@k.transpose(1,2).repeat_interleave(6,1).transpose(-1,-2))*.0625
        if not args.production_only and (((args.gemv_probes or args.tile_probes) and label=='layer3') or args.dispatch_probes):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                expected=(q.reshape(batch,24,1,256)@k.transpose(1,2).repeat_interleave(6,1).transpose(-1,-2))*.0625
                torch.cuda.synchronize()
            names=sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
            (args.output/'native-kernels.json').write_text(json.dumps(names,indent=2)+'\n')
            prof.export_chrome_trace(str(args.output/'native-trace.json'))
            trace=json.loads((args.output/'native-trace.json').read_text())
            profiles.append(dict(case=label,batch=batch,length=length,kernels=[dict(name=e['name'],grid=e['args'].get('grid'),block=e['args'].get('block')) for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'gemv' in e['name']]))
        cache=torch.full((batch,1024,4,256),float('nan'),dtype=torch.bfloat16,device='cuda');cache[:,:length].copy_(k)
        pos=torch.tensor([length-1],device='cuda');heads=torch.arange(24,device='cuda')
        actual=torch.empty(batch,24,1024,device='cuda')
        modes=(0,8,16,108,116,132,172,180,196,308,324,364,372,388,428,436,452) if args.gemv_probes else (0,8,16)
        if args.tile_probes:
            modes = [0]+[100+(1<<15)+parts+(chunk<<6)+(order<<9)
                         for parts in (1,2,4,8,16,32) for chunk in range(6)
                         if parts*(1<<chunk)<=256 for order in (0,1,2)]
        if args.dispatch_probes:
            modes = [0]+[100+(1<<15)+parts+(chunk<<6)
                         for parts in (4,8,16,32) for chunk in (0,2,3,4,5) if parts*(1<<chunk)<=256]
        if args.production_only: modes = [0]
        for mode in modes:
            torch.cuda.synchronize()
            assert lib.launch_scores(*[t.data_ptr() for t in (q,pos,cache,heads,actual)],batch,mode)==0
            torch.cuda.synchronize()
            value=actual[:,:,:length].reshape(expected.shape)
            row=dict(case=label,batch=batch,length=length,mode=mode,gpu_uuid=uuid,
                     snapshot_gpu_uuid=saved['gpu_uuid'],different=int((value!=expected).sum()),**compare(value,expected))
            checks.append(row);print(json.dumps(row),flush=True)
            if mode == 0 and captured is not None:
                native = expected.float().softmax(-1).bfloat16()
                from_scores = value.float().softmax(-1).bfloat16()
                probability_checks.append(dict(case=label,
                    mpk_vs_native=compare(captured.reshape(native.shape).cuda(),native),
                    mpk_vs_from_scores=compare(captured.reshape(native.shape).cuda(),from_scores),
                    from_scores_vs_native=compare(from_scores,native)))
    for i,a in enumerate(saved['layers']):
        if 'query' in a:check(f'layer{i}',a['query'].cuda().contiguous(),a['key_cache'].cuda(),a['attention_weights'])
    torch.manual_seed(20261116)
    for batch in (range(1,33) if args.all_short_batches else (1,2,3,4,8,16,32) if args.dispatch_probes else (1,2,3)):
        lengths = (1,2,3,7,8,16,24,25,26,27,31,32,35,63,64,65,66,70,127,128,129,512,1023,1024)
        if args.dispatch_probes: lengths = sorted({*range(1,81),126,127,128,129,255,256,257,511,512,1023,1024})
        if args.all_short_batches: lengths = range(1,81)
        for length in lengths:
            q=torch.randn(batch,24,256,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(batch,length,4,256,device='cuda',dtype=torch.bfloat16)
            if args.dispatch_probes:
                q *= torch.exp2(torch.randint(-8,9,q.shape,device='cuda').float()).bfloat16()
                k *= torch.exp2(torch.randint(-8,9,k.shape,device='cuda').float()).bfloat16()
            check('random',q,k)
    (args.output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    (args.output/'probabilities.json').write_text(json.dumps(probability_checks,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(checks,indent=2)+'\n')
    if args.all_short_batches:
        targeted = [c for c in checks if c['mode'] == 0 and c['length'] % 8
                    and c['batch'] < 8 and c['length'] <= (59 if c['batch'] < 4 else 31)]
        assert targeted and all(c['exact'] for c in targeted), 'Short QK schedule differs from native HF'


if __name__=='__main__':main()
