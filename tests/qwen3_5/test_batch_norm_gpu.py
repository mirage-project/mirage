"""Require exact native RMSNorm/QK-RoPE parity for saved failures and batch 1..32."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import torch
from safetensors import safe_open
from mirage.mpk.models.qwen3_5.reference import rms_norm
from mirage.mpk.models.qwen3_5.attention_reference import text_rope,partial_rope
from mirage.mpk.models.qwen3_5.config import Qwen35TextConfig
from test_batch_model_gpu import compare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,nargs='+',required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(10,0)
    assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32=False
    repo=Path(__file__).resolve().parents[2];library=args.output/'norm.so'
    command=[str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_100a',
             '-DMIRAGE_BACKEND_USE_CUDA','--shared','-Xcompiler','-fPIC',
             *['-I'+str(repo/n) for n in ('include','include/mirage/persistent_kernel','deps/cutlass/include','deps/json/include')],
             str(repo/'tests/qwen3_5/batch_norm.cu'),'-o',str(library)]
    with (args.output/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    lib=ctypes.CDLL(str(library));lib.launch_norm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
    lib.launch_qk.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]
    checks=[]
    def record(name,actual,expected):
        row=dict(case=name,different=int((actual!=expected).sum()),**compare(actual,expected))
        checks.append(row)
        if not row['exact']:print('DIFFERENCE',row,flush=True)
    def norm(name,x,w):
        out=torch.empty_like(x)
        torch.cuda.synchronize();assert lib.launch_norm(x.data_ptr(),w.data_ptr(),out.data_ptr(),x.shape[0],x.shape[-1])==0
        torch.cuda.synchronize();record(name,out,rms_norm(x,w))
    config=Qwen35TextConfig.from_json(args.model/'config.json')
    mapping=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(args.model/mapping[name],framework='pt') as f:return f.get_tensor(name).bfloat16().float().cuda()
    def qk(name,qg,k,qw,kw,past):
        batch=qg.shape[0]
        qg=qg.reshape(batch,24,512);k=k.reshape(batch,4,256)
        cos,sin=text_rope(config,torch.tensor([past],device='cuda'))
        rope=torch.stack((cos[0,0],sin[0,0])).contiguous()
        qo=torch.empty(batch,24,256,device='cuda',dtype=torch.bfloat16);ko=torch.empty_like(k);gate=torch.empty_like(qo)
        torch.cuda.synchronize();assert lib.launch_qk(*[t.data_ptr() for t in (qg,k,qw,kw,rope,qo,ko,gate)],batch)==0
        torch.cuda.synchronize()
        expected_q,expected_k=partial_rope(rms_norm(qg[...,:256],qw).unsqueeze(2),rms_norm(k,kw).unsqueeze(2),cos,sin)
        record(name+'.query',qo,expected_q[:,:,0]);record(name+'.key',ko,expected_k[:,:,0]);record(name+'.gate',gate,qg[...,256:])
    for path in args.snapshots:
        saved=torch.load(path,weights_only=True,map_location='cpu')
        # Saved inputs are replayable on another B200: both candidate and
        # native reference below execute on the currently guarded device.
        for i,a in enumerate(saved['layers']):
            prefix=f'model.language_model.layers.{i}.'
            for inp,key in (('input','input_layernorm.weight'),('residual','post_attention_layernorm.weight')):
                norm(f'b{saved["batch"]}.layer{i}.{inp}',a[inp].cuda(),weight(prefix+key))
            if 'query' in a:qk(f'b{saved["batch"]}.layer{i}',a['qg'].cuda(),a['k'].cuda(),weight(prefix+'self_attn.q_norm.weight'),weight(prefix+'self_attn.k_norm.weight'),saved['past'])
    torch.manual_seed(20261116)
    for batch in range(1,33):
        for width in (1024,5120):
            x=torch.randn(batch,width,dtype=torch.bfloat16,device='cuda');w=torch.randn(width,dtype=torch.bfloat16,device='cuda').float()
            norm(f'random.b{batch}.h{width}',x,w)
        qk(f'random.b{batch}',torch.randn(batch,24,512,dtype=torch.bfloat16,device='cuda'),torch.randn(batch,4,256,dtype=torch.bfloat16,device='cuda'),
           torch.randn(256,dtype=torch.bfloat16,device='cuda').float(),torch.randn(256,dtype=torch.bfloat16,device='cuda').float(),batch*31)
    report=dict(passed=all(r['exact'] for r in checks),checks=checks)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EXACT',sum(r['exact'] for r in checks),'/',len(checks),flush=True)
    assert report['passed']


if __name__=='__main__':main()
