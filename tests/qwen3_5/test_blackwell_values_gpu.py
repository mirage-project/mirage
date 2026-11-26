"""Compare Blackwell PV reduction variants using saved and synthetic native inputs."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from test_batch_model_gpu import compare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aligned-short', action='store_true', help='Profile and verify short aligned GEMV dispatch at every batch size')
    p.add_argument('--ungated', action='store_true', help='Use an exact 0.5 gate to isolate the matrix product')
    p.add_argument('--gemv-probes', action='store_true')
    p.add_argument('--gemv-sweep', action='store_true', help='Test candidate schedules at every cache length')
    p.add_argument('--gemv-boundaries', action='store_true', help='Cross-batch checks of the measured small-GEMV ranges')
    p.add_argument('--short-batches', action='store_true', help='Locate short-GEMV dispatch changes over batch 1..32')
    p.add_argument('--gemv2n-probes', action='store_true', help='Test two-level GEMV2N accumulators')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    assert uuid.lower().removeprefix('gpu-')==os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32=False
    repo=Path(__file__).resolve().parents[2];library=args.output/'values.so'
    command=[str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_100a',
             '-DMIRAGE_BACKEND_USE_CUDA','--shared','-Xcompiler','-fPIC',
             *['-I'+str(repo/n) for n in ('include','include/mirage/persistent_kernel','deps/cutlass/include','deps/json/include')],
             str(repo/'tests/qwen3_5/blackwell_values.cu'),'-o',str(library)]
    with (args.output/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    lib=ctypes.CDLL(str(library));lib.launch_values.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2
    checks=[];profiles={}
    def check(label,p,v,g):
        batch,length=v.shape[:2]
        if args.ungated: g = torch.zeros_like(g)
        values=v.transpose(1,2).repeat_interleave(6,1)
        expected=(p[:,:,None]@values)[:,:,0]*g.sigmoid()
        trace = (args.gemv_sweep or args.aligned_short) and label == 'random'
        if (label=='random' and batch==3) or trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                p[:,:,None]@values;torch.cuda.synchronize()
            profiles[length]=sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
            if trace:
                trace_path=args.output/'native-pv-last.json'
                prof.export_chrome_trace(str(trace_path))
                events=json.loads(trace_path.read_text())['traceEvents']
                profiles[f'{batch}.{length}']=[dict(name=e['name'],grid=e['args'].get('grid'),block=e['args'].get('block'))
                                  for e in events if e.get('cat')=='kernel']
        cache=torch.full((batch,1024,4,256),float('nan'),dtype=torch.bfloat16,device='cuda');cache[:,:length].copy_(v)
        padded=F.pad(p,(0,1024-length)).contiguous()
        pos=torch.tensor([length-1],device='cuda');heads=torch.arange(24,device='cuda')
        actual=torch.empty_like(g)
        modes = [0,8,16,32,64]
        if args.gemv_probes:
            modes += [100 + parts + (chunk << 7) + (order << 11)
                      for parts in (1,2,4,8,16,32,64) for chunk in range(6) for order in range(3)]
        if args.gemv_sweep:
            modes = [0,16,64,101,102,104,108,116,230,232,236,244,260]
            modes += [100+parts+(chunk<<7) for parts in (2,4,8,16) for chunk in (6,7,8)]
        if args.gemv_boundaries: modes = [0,104,230,232,236,2280,2284,4328,4332]
        if args.short_batches: modes = [0,230]
        if args.gemv2n_probes:
            modes = [0,101,102,104,230,232,236] + [100+parts+(chunk<<7)+(outer<<11)+(inner<<13)+(order<<15)
                           for parts in (2,4,8) for chunk in (0,1,2,3)
                           for inner in (1,2) for outer in (0,1,2) for order in (0,1,2)]
            modes += [100+parts+(outer<<11)+(inner<<13)+(order<<15)+(1<<17)
                      for parts in (2,4,8) for inner in (1,2,3)
                      for outer in (0,1,2) for order in (0,1,2)]
        if args.aligned_short: modes = [0,101,102,104,230,232,236,516]
        for mode in modes:
            torch.cuda.synchronize();assert lib.launch_values(*[t.data_ptr() for t in (padded,g,pos,cache,heads,actual)],batch,mode)==0
            torch.cuda.synchronize()
            row=dict(case=label,batch=batch,length=length,mode=mode,different=int((actual!=expected).sum()),**compare(actual,expected))
            checks.append(row);print(json.dumps(row),flush=True)
    saved=torch.load(args.snapshot,weights_only=True,map_location='cpu')
    for i,a in enumerate(saved['layers']):
        if 'query' in a:
            if args.gemv2n_probes or args.aligned_short:
                for length in (24,32):
                    check(f'layer{i}.prefix{length}',a['attention_weights'][...,:length].cuda(),a['value_cache'][:,:length].cuda(),a['gate'].cuda())
            else:check(f'layer{i}',a['attention_weights'].cuda(),a['value_cache'].cuda(),a['gate'].cuda())
    torch.manual_seed(20261116)
    batches = range(1,33) if args.short_batches or args.aligned_short else (8,) if args.gemv_probes or args.gemv_sweep or args.gemv2n_probes else (1,2,3,4,8,16,32)
    for batch in batches:
        lengths = (range(1,1025) if args.gemv_sweep else (63,65,67,69,70,127,129,1023) if args.gemv_probes else
                   (1,2,3,7,8,16,60,62,63,64,65,66,67,69,70,126,127,128,129,130,132,134,512,1023,1024))
        if args.gemv_boundaries:
            lengths = sorted({8,16,24,32,40,48,56,64,68,70,72,74,76,78,80,82,84,86,
                              88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,
                              122,124,128,136,144,152,*range(11,39),*range(39,68,2),*range(121,136,2)})
        if args.short_batches: lengths = [n for n in range(11,39) if n%8]
        if args.aligned_short: lengths = (8,16,24,32,40)
        if args.gemv2n_probes: lengths = (8,16,24,32,40,218,222,254,510,1022)
        for length in lengths:
            p = torch.randn(batch,24,length,device='cuda')
            v = torch.randn(batch,length,4,256,device='cuda',dtype=torch.bfloat16)
            if args.gemv_sweep or args.gemv_boundaries or args.short_batches or args.gemv2n_probes or args.aligned_short:
                p *= 4
                v *= torch.exp2(torch.randint(-8,9,v.shape,device='cuda').float()).bfloat16()
            check('random',p.softmax(-1).bfloat16(),v,
                  torch.randn(batch,24,256,device='cuda',dtype=torch.bfloat16))
    wmma = [c for c in checks if c['mode'] == 0 and c['length'] in (60,62,66,126,130,132,134)]
    report = dict(gpu_uuid=uuid,snapshot_gpu_uuid=saved['gpu_uuid'],checks=checks,
                  profiles=profiles,wmma_exact=all(c['exact'] for c in wmma))
    if args.gemv_boundaries:
        def targeted(c):
            n=c['length']
            return ((n%2 and (59<=n<=67 or 121<=n<=135)) or n in (68,70) or
                    (c['batch']<16 and 11<=n<=38 and n%8) or
                    (48<=n<=80 and n%8==0) or (72<=n<=124 and n%2==0 and n%8))
        selected = [c for c in checks if c['mode'] == 0 and targeted(c)]
        report['gemv_exact'] = all(c['exact'] for c in selected)
    if args.short_batches:
        selected = [c for c in checks if c['case']=='random' and c['mode']==0 and c['batch']<16]
        report['short_gemv_exact'] = all(c['exact'] for c in selected)
    if args.aligned_short:
        selected = [c for c in checks if c['mode'] == 0]
        report['aligned_short_exact'] = all(c['exact'] for c in selected)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['wmma_exact'], 'Production WMMA path differs from native BF16 BMM'
    if args.gemv_boundaries:
        assert report['gemv_exact'], 'Production GEMV boundary path differs from native BF16 BMM'
    if args.short_batches:
        assert report['short_gemv_exact'], 'Production short GEMV differs from native BF16 BMM'
    if args.aligned_short:
        assert report['aligned_short_exact'], 'Production aligned short GEMV differs from native BF16 BMM'


if __name__=='__main__':main()
