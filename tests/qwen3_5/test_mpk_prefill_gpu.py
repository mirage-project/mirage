"""MPK chunk boundaries, cache continuation, failure atomicity and no fallback.

Numerical gates are the existing 2% relative RMS / 5% reference peak + 1e-5.
Bitwise equality is reported independently; greedy next tokens must match HF.
Use test_backend_generation_gpu.py for the original 32 independent generations.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen3_5ForConditionalGeneration
from mirage.mpk.models.qwen3_5.model import Qwen35Model
from test_backend_generation_gpu import metrics
from test_prefill_gpu import state_hashes
from prefill_test_utils import load_compiled_pair


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--compiled-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    producer=json.loads((args.compiled_run/'report.json').read_text())
    backend=producer['attn_implementation']
    assert torch.cuda.device_count()==1
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    assert uuid==producer['gpu_uuid']
    assert uuid.lower().removeprefix('gpu-')==os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,passed=False,attn_implementation=backend,gpu_uuid=uuid,
                producer_sha256=hashlib.sha256((args.compiled_run/'report.json').read_bytes()).hexdigest(),
                thresholds=dict(relative_rms=.02,reference_peak=.05,absolute_floor=1e-5),checks=[],boundaries=[],chunks=[])
    def save(): (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    mpk=None;handles=[];last={}
    try:
        mpk=Qwen35Model.from_pretrained(args.model,attn_implementation=backend)
        load_compiled_pair(mpk,args.compiled_run)
        hf=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
            attn_implementation=backend,local_files_only=True,device_map='cuda').eval()
        for i,layer in enumerate(hf.model.language_model.layers):
            def capture(module,inputs,output,index=i): last[index]=output[:,-1].clone()
            handles.append(layer.register_forward_hook(capture))
        def hf_step(ids,cache):
            out=hf(input_ids=torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
            return out.logits[:,-1],out.past_key_values
        def compare(name,actual,expected,cache):
            pairs=[(name+'.logits',actual,expected)]
            for i,layer in enumerate(mpk.layers):
                pairs.append((f'{name}.layer{i}',layer.tensors['decoder_output'],last[i]))
                if i in mpk.config.gdn_layers:
                    fields=[('state',cache.layers[i].recurrent_states[0][0]),
                            ('conv_state',cache.layers[i].conv_states[0][0,:,-3:])]
                else:
                    fields=[('key_cache',cache.layers[i].keys[0].transpose(0,1)),
                            ('value_cache',cache.layers[i].values[0].transpose(0,1))]
                for field,value in fields:
                    t=layer.tensors[field]
                    if field.endswith('_cache'):t=t[:mpk.position]
                    pairs.append((f'{name}.layer{i}.{field}',t,value))
            for key,a,b in pairs:
                # The generic metrics helper includes argmax only for logits.
                report['checks'].append(dict(name=key,**metrics(a,b)))
            equal=int(actual.argmax())==int(expected.argmax())==int(mpk.tensors['next_token'])
            print(name,'next_token',equal,'exact',sum(torch.equal(a,b) for _,a,b in pairs),'/',len(pairs),flush=True)
            save()
            return equal
        def prefill(ids):
            # Fail loudly if arithmetic is silently delegated to PyTorch/HF.
            with ExitStack() as stack:
                for name in ('linear','conv1d','scaled_dot_product_attention','softmax','embedding'):
                    stack.enter_context(patch('torch.nn.functional.'+name,side_effect=AssertionError('PyTorch fallback: '+name)))
                return mpk.prefill(ids,mode='mpk')
        generator=torch.Generator().manual_seed(351115)
        for length in ([63,65] if args.quick else [1,2,63,64,65,127,128,129,1023,1024]):
            ids=torch.randint(0,mpk.config.vocab_size,(length,),generator=generator).tolist()
            mpk.reset();expected,cache=hf_step(ids,None)
            equal=compare(f'length{length}',prefill(ids),expected,cache)
            for j in range(min(3,1024-length)):
                token=int(expected.argmax());expected,cache=hf_step([token],cache)
                equal=compare(f'length{length}.decode{j}',mpk.step(token),expected,cache) and equal
            report['boundaries'].append(dict(length=length,next_tokens_exact=equal))
        plans=[[63,2,1,64]] if args.quick else [[1,1,2,65],[63,2,1,64],[65,63,1],[1018,4,2]]
        for plan in plans:
            ids=torch.randint(0,mpk.config.vocab_size,(sum(plan),),generator=generator).tolist()
            mpk.reset();cache=None;offset=0;equal=True
            for length in plan:
                chunk=ids[offset:offset+length];expected,cache=hf_step(chunk,cache)
                equal=compare(f'chunks{plan}.{offset}',prefill(chunk),expected,cache) and equal
                offset+=length
            report['chunks'].append(dict(plan=plan,next_tokens_exact=equal))
        mpk.reset();prefill([7,8,9]);before=state_hashes(mpk)
        original=mpk._prefill_graph.launch;calls=0
        def fail_second():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('injected second tile failure')
            original()
        with patch.object(mpk._prefill_graph,'launch',side_effect=fail_second):
            try:prefill([7]*65)
            except RuntimeError as e:assert 'injected' in str(e)
            else:raise AssertionError('Failure injection missed')
        assert state_hashes(mpk)==before
        for invalid in ([],[1,-1],[1,True],[1,248320],[0]*1025):
            try:prefill(invalid)
            except ValueError:assert state_hashes(mpk)==before
            else:raise AssertionError('Invalid input accepted')
        # Retrying after a failed tile must copy committed state back to scratch.
        retried=prefill([7]*65).clone();after=state_hashes(mpk)
        mpk.reset();prefill([7,8,9]);fresh=prefill([7]*65)
        assert torch.equal(retried,fresh) and state_hashes(mpk)==after
        report.update(failed_computation_atomic=True,invalid_input_atomic=True,
                      retry_reset_exact=True,no_torch_fallback=True,completed=True)
        checks=report['checks']
        report['summary']=dict(total=len(checks),exact=sum(c['exact'] for c in checks),
            accepted=sum(c['accepted'] for c in checks),max_relative_rms=max(c['relative_rms'] for c in checks))
        report['passed']=(all(c['accepted'] for c in checks)
                          and all(c['next_tokens_exact'] for c in report['boundaries']+report['chunks']))
        print(json.dumps(report['summary']),flush=True)
        assert report['passed'],'MPK prefill boundary acceptance failed'
    finally:
        for handle in handles:handle.remove()
        if mpk is not None:mpk.close()
        save()


if __name__=='__main__':main()
