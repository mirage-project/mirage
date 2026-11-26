"""27B production kernels vs independent native HF operators on an allocated idle GPU."""
import argparse,ctypes,json,os,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule,torch_recurrent_gated_delta_rule
from mirage.mpk.models.qwen3_5.reference import gated_rms_norm,rms_norm
from mirage.mpk.models.qwen3_5.mpk_prefill import flash_split_config,eager_mma_config
from qwen38_test_utils import metric

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--norm-fixture',type=Path)
 p.add_argument('--projection-fixture',type=Path)
 p.add_argument('--down-model',type=Path);args=p.parse_args()
 args.output_dir.mkdir(parents=True,exist_ok=False)
 assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability() in ((9,0),(10,0))
 assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
 torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(38271116)
 repo=Path(__file__).resolve().parents[2];library=args.output_dir/'qwen38.so'
 arch='sm_90a' if torch.cuda.get_device_capability()==(9,0) else 'sm_100a'
 metadata=dict(gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),gpu_name=torch.cuda.get_device_name(),
               compute_capability=list(torch.cuda.get_device_capability()),architecture=arch,
               gpu_sms=torch.cuda.get_device_properties(0).multi_processor_count,torch=str(torch.__version__))
 cmd=[str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'-std=c++17','-O3','-arch='+arch,'--shared','-Xcompiler','-fPIC','-DMIRAGE_BACKEND_USE_CUDA',*['-I'+str(repo/n) for n in ('include','include/mirage/persistent_kernel','deps/cutlass/include','deps/json/include')],str(repo/'tests/qwen3_5/qwen38_isolated.cu'),'-o',str(library)]
 with (args.output_dir/'compile.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 lib=ctypes.CDLL(str(library))
 lib.launch_decode.argtypes=[ctypes.c_void_p]*10;lib.launch_prefill.argtypes=[ctypes.c_void_p]*11
 lib.launch_attention.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*2
 lib.launch_norm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]
 lib.launch_qk.argtypes=[ctypes.c_void_p]*8
 lib.launch_projection.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
 lib.launch_down.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]
 rows=[]
 def rand(*shape):return torch.randn(shape,device='cuda')
 def record(name,actual,expected):
  torch.cuda.synchronize();m=metric(actual,expected);rows.append(dict(name=name,**m));print(name,m,flush=True)
  (args.output_dir/'report.json').write_text(json.dumps({'metadata':metadata,'checks':rows,'completed':False,'passed':False},indent=2))
  if name.startswith(('saved_norm.','saved_projection.','projection.','down.')):
   assert m['exact'], 'Pinned operator must match exactly: '+name
 ids=torch.arange(48,dtype=torch.int64,device='cuda')
 for length in (1,2,7,63,64):
  for cached in (False,True):
   conv=rand(64,10240).bfloat16();z=rand(64,6144).bfloat16();ab=rand(64,128).bfloat16()
   params=rand(2,48);params[0].sub_(2);weight=rand(128).bfloat16()
   state=rand(48,128,128)*.1 if cached else torch.zeros(48,128,128,device='cuda')
   q,k,v=conv[:length].split((2048,2048,6144),-1)
   q,k=[a.reshape(1,length,16,128).repeat_interleave(3,2) for a in (q,k)]
   v=v.reshape(1,length,48,128)
   g=(-params[0].exp()*F.softplus(ab[:length,:48].float()+params[1])).unsqueeze(0)
   beta=ab[:length,48:96].sigmoid().unsqueeze(0)
   expected,next_state=torch_chunk_gated_delta_rule(q,k,v,g,beta,initial_state=state[None],output_final_state=True,use_qk_l2norm_in_kernel=True)
   expected=gated_rms_norm(expected.reshape(-1,128),weight,z[:length].reshape(-1,128)).reshape(length,48,128)
   out=torch.empty(64,48,128,dtype=torch.bfloat16,device='cuda');new=torch.empty_like(state)
   scratch=torch.empty(48,9*8192+3*4096+64+64*48*128//2,device='cuda')
   count=torch.tensor([length,length],device='cuda',dtype=torch.int64)
   assert lib.launch_prefill(*[a.data_ptr() for a in (conv,z,ab,params,weight,state,count,ids,out,new,scratch)])==0
   record(f'gdn.prefill.{length}.{cached}',out[:length],expected)
   record(f'gdn.state.{length}.{cached}',new,next_state[0])
   if length==1:
    state_copy=state.clone();raw=torch.empty(48,128,dtype=torch.bfloat16,device='cuda');dec=torch.empty_like(raw)
    pos=torch.tensor([1 if cached else 0],device='cuda',dtype=torch.int64)
    assert lib.launch_decode(*[a.data_ptr() for a in (conv,z,ab,params,weight,state_copy,pos,ids,raw,dec)])==0
    if cached:
     er,es=torch_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=state[None],output_final_state=True,use_qk_l2norm_in_kernel=True)
     eg=gated_rms_norm(er.reshape(-1,128),weight,z[:1].reshape(-1,128)).reshape(48,128)
    else:eg,es=expected[0],next_state
    record(f'gdn.decode.{cached}',dec,eg);record(f'gdn.decode_state.{cached}',state_copy,es[0])
 for length in (1,2,7,63,64):
  for past in (0,65,513,960):
   total=past+length
   q=rand(length,24,256).bfloat16();gate=rand(length,24,256).bfloat16()
   k=rand(1024,4,256).bfloat16();v=rand(1024,4,256).bfloat16()
   positions=torch.arange(past,total,device='cuda',dtype=torch.int64).repeat_interleave(24)
   heads=torch.arange(24,device='cuda',dtype=torch.int64).repeat(length)
   mask=torch.arange(total,device='cuda')[None,:]<=torch.arange(past,total,device='cuda')[:,None]
   qref=q.transpose(0,1)[None];kr=k[:total].transpose(0,1).repeat_interleave(6,0)[None];vr=v[:total].transpose(0,1).repeat_interleave(6,0)[None]
   for sdpa in (False,True):
    if sdpa:ref=F.scaled_dot_product_attention(qref,kr,vr,attn_mask=mask[None,None],dropout_p=0.)
    else:
     scores=(qref@kr.transpose(-1,-2))*.0625
     scores=scores.masked_fill(~mask,torch.finfo(q.dtype).min)
     ref=F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)@vr
    expected=ref[0].transpose(0,1).contiguous()*gate.sigmoid()
    split=(flash_split_config(length,total,torch.cuda.get_device_properties(0).multi_processor_count,24) if sdpa else eager_mma_config(length,total))
    cfg=torch.tensor(split,device='cuda',dtype=torch.int64)
    out=torch.empty_like(q);scratch=torch.empty(length*24,4112,device='cuda')
    assert lib.launch_attention(*[a.data_ptr() for a in (q,gate,positions,k,v,heads,cfg,out,scratch)],length,int(sdpa))==0
    record(f'attention.{length}.{past}.{sdpa}',out,expected)
 for length in (1,2,7,64):
  x=rand(length,5120).bfloat16();weight=rand(5120)*.1;out=torch.empty_like(x)
  assert lib.launch_norm(x.data_ptr(),weight.data_ptr(),out.data_ptr(),length)==0
  record(f'norm.{length}',out,rms_norm(x,weight))
 qg=rand(24,512).bfloat16();k=rand(4,256).bfloat16();qw=rand(256)*.1;kw=rand(256)*.1
 angle=rand(32);rope=torch.stack((angle.cos().repeat(2),angle.sin().repeat(2))).bfloat16()
 qo=torch.empty(24,256,device='cuda',dtype=torch.bfloat16);ko=torch.empty_like(k);gate=torch.empty_like(qo)
 assert lib.launch_qk(*[a.data_ptr() for a in (qg,k,qw,kw,rope,qo,ko,gate)])==0
 for name,out,x,w in (('q',qo,qg[:,:256],qw),('k',ko,k,kw)):
  expected=rms_norm(x,w);rotated=torch.cat((-expected[:,32:64],expected[:,:32]),-1)
  expected[:,:64]=expected[:,:64]*rope[0]+rotated*rope[1]
  record('qk_rope.'+name,out,expected)
 record('qk_rope.gate',gate,qg[:,256:])
 if args.norm_fixture:
  for sample in torch.load(args.norm_fixture,map_location='cpu',weights_only=True):
   x,weight=[sample[k].cuda().contiguous() for k in ('x','w')];out=torch.empty_like(x)
   assert lib.launch_norm(x.data_ptr(),weight.data_ptr(),out.data_ptr(),len(x))==0
   record(f"saved_norm.{sample['case']}.{sample['layer']}.{sample['name']}",out,sample['expected'].cuda())
 if args.projection_fixture:
  samples=torch.load(args.projection_fixture,map_location='cpu',weights_only=True)
  for sample in samples:
   x,w=[sample[k].cuda().contiguous() for k in ('x','w')];n=len(w);length=len(x)
   padded=F.pad(x,(0,0,0,64-length));out=torch.empty(64,n,device='cuda',dtype=torch.bfloat16)
   assert lib.launch_projection(padded.data_ptr(),w.data_ptr(),out.data_ptr(),n,length)==0
   record(f"saved_projection.{sample['case']}.{sample['layer']}.{sample['name']}",out[:length],sample['expected'].cuda())
  # An independent checkpoint layer and every supported length verify the
  # shape-dependent reduction, including full-call lengths beyond one tile.
  for sample in samples:
   if sample['case']!='prefill' or (sample['layer'],sample['name']) not in ((1,'ab'),(7,'k')):continue
   w=sample['w'].cuda().contiguous();n=len(w)
   for length in range(1,1025):
    x=rand(length,5120).bfloat16()
    expected=(torch.cat((F.linear(x,w[:48]),F.linear(x,w[48:96]),torch.zeros(length,32,device='cuda',dtype=x.dtype)),-1)
              if n==128 else F.linear(x,w))
    tile=F.pad(x[:64],(0,0,0,max(0,64-length))).contiguous();out=torch.empty(64,n,device='cuda',dtype=x.dtype)
    assert lib.launch_projection(tile.data_ptr(),w.data_ptr(),out.data_ptr(),n,length)==0
    record(f"projection.{n}.{length}",out[:min(64,length)],expected[:64])
 if args.down_model:
  from safetensors import safe_open
  mapping=json.loads((args.down_model/'model.safetensors.index.json').read_text())['weight_map']
  key='model.language_model.layers.7.mlp.down_proj.weight'
  with safe_open(args.down_model/mapping[key],framework='pt') as handle:w=handle.get_tensor(key).cuda()
  assert w.shape==(5120,17408) and w.dtype==torch.bfloat16
  for length in range(1,1025):
   x=rand(length,17408).bfloat16();expected=F.linear(x,w)[:64]
   tile=F.pad(x[:64],(0,0,0,max(0,64-length))).contiguous();out=torch.empty(64,5120,device='cuda',dtype=x.dtype)
   assert lib.launch_down(tile.data_ptr(),w.data_ptr(),out.data_ptr(),length)==0
   record(f'down.{length}',out[:min(64,length)],expected)
 assert all(row['accepted'] for row in rows),'27B isolated numerical acceptance failed'
 (args.output_dir/'report.json').write_text(json.dumps({'metadata':metadata,'checks':rows,'completed':True,'passed':True},indent=2))
 print('PASS',len(rows),'checks',flush=True)
if __name__=='__main__':main()
