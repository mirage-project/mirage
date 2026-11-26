"""Sequential HF/MPK acceptance for the 27B checkpoint on one 80-GB GPU.

Freeze the native HF oracle in one process, then run MPK in another: the two
models need not fit in memory together. Thresholds are fixed before execution:
2% logits relative RMS, 5% reference peak + 1e-5, and exact greedy tokens.
"""
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module
from mirage.mpk.models.qwen3_5.model import Qwen35Model
from qwen38_test_utils import metric

REVISION = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_hashes():
    repo = Path(__file__).resolve().parents[2]
    files = list((repo/'python/mirage/mpk/models/qwen3_5').glob('*.py'))
    files += list((repo/'include/mirage/persistent_kernel/tasks/common').glob('qwen35*.cuh'))
    files += [repo/'src/kernel/graph.cc', repo/'src/kernel/task_register.cc', Path(__file__).resolve()]
    return {str(p.relative_to(repo)): sha(p) for p in files}


def selected_state(model, *, hf=False, cache=None):
    result = {}
    for i in (0, 3, 62, 63):
        if hf:
            layer = cache.layers[i]
            if i in (0, 62):
                fields = dict(state=layer.recurrent_states[0][0], conv_state=layer.conv_states[0][0, :, -3:])
            else:
                fields = dict(key_cache=layer.keys[0].transpose(0, 1), value_cache=layer.values[0].transpose(0, 1))
        else:
            fields = {k: model.layers[i].tensors[k] for k in
                      (('state', 'conv_state') if i in (0, 62) else ('key_cache', 'value_cache'))}
            fields = {k: t[:model.position] if k.endswith('_cache') else t for k, t in fields.items()}
        result.update({f'{i}.{k}': t.cpu().clone() for k, t in fields.items()})
    return result


@torch.inference_mode()
def freeze(args, metadata):
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
        local_files_only=True, device_map='cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    fixture = dict(metadata=metadata, completed=False, cases=[], generations=[])
    last = {}
    handles = []
    for i, layer in enumerate(hf.model.language_model.layers):
        def capture(module, inputs, output, index=i):
            last[index] = output[:, -1].clone()
        handles.append(layer.register_forward_hook(capture))
    def hidden():
        return torch.stack([last[i][0] for i in range(64)]).cpu()
    generator = torch.Generator().manual_seed(38271116)
    plans = [[n] for n in (1, 2, 63, 64, 65, 127, 128, 129, 1023, 1024)]
    plans += [[63, 2, 1, 64], [1]*7]
    for plan in plans:
        ids = torch.randint(0, hf.config.text_config.vocab_size, (sum(plan),), generator=generator).tolist()
        cache = None
        offset = 0
        steps = []
        for n in plan:
            chunk = ids[offset:offset+n]
            out = hf(input_ids=torch.tensor([chunk], device='cuda'), past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            steps.append(dict(inputs=chunk, logits=out.logits[:, -1].cpu(), hidden=hidden(),
                              states=selected_state(hf, hf=True, cache=cache)))
            offset += n
        for _ in range(min(2, 1024-offset)):
            token = int(out.logits.argmax())
            out = hf(input_ids=torch.tensor([[token]], device='cuda'), past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            steps.append(dict(inputs=[token], logits=out.logits[:, -1].cpu(), hidden=hidden(),
                              states=selected_state(hf, hf=True, cache=cache)))
        fixture['cases'].append(dict(name='chunks.'+'.'.join(map(str, plan)), steps=steps))
        print('oracle', fixture['cases'][-1]['name'], flush=True)
        del out, cache
    torch.save(fixture, args.output_dir/'oracle-boundaries.pt')
    for handle in handles:
        handle.remove()
    for name, prompt in (
        ('english', 'Explain why the sky looks blue in two sentences.'),
        ('chinese', '请用两句话介绍北京。'),
        ('math', 'Compute 17 * 23. Give the answer and one short calculation.'),
        ('code', 'Write a Python function that returns the larger of two numbers.'),
    ):
        rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist()
        inputs = torch.tensor([ids], device='cuda')
        result = hf.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, max_new_tokens=16, return_dict_in_generate=True, output_logits=True)
        fixture['generations'].append(dict(name=name, prompt_ids=ids,
            tokens=result.sequences[0, len(ids):].tolist(), logits=[t.cpu() for t in result.logits]))
        print('oracle generation', name, fixture['generations'][-1]['tokens'], flush=True)
        del result
    fixture['completed'] = True
    torch.save(fixture, args.output_dir/'oracle.pt')
    (args.output_dir/'oracle-boundaries.pt').unlink()
    (args.output_dir/'oracle.json').write_text(json.dumps(metadata, indent=2)+'\n')


@torch.inference_mode()
def freeze_stream(args, metadata):
    """Freeze native tokenwise HF separately; never replace the block oracle."""
    assert args.reference is not None
    original = torch.load(args.reference, map_location='cpu', weights_only=True)
    assert original['completed'] and original['metadata'] == metadata
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
        local_files_only=True, device_map='cuda').eval()
    fixture = dict(metadata=metadata, completed=False, generations=[], prefill_mode='stream',
                   block_reference_sha256=sha(args.reference), freezer_sha256=sha(__file__))
    for case in original['generations']:
        ids = case['prompt_ids']
        hf.model.rope_deltas = None
        inputs = torch.tensor([ids], device='cuda')
        block = hf.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, max_new_tokens=16, return_dict_in_generate=True, output_logits=True)
        assert block.sequences[0, len(ids):].tolist() == case['tokens']
        assert len(block.logits) == len(case['logits'])
        assert all(torch.equal(a.cpu(), b) for a, b in zip(block.logits, case['logits']))
        del block
        hf.model.rope_deltas = None
        cache = None
        for pos, token in enumerate(ids, 1):
            out = hf(input_ids=torch.tensor([[token]], device='cuda'),
                     attention_mask=torch.ones(1, pos, device='cuda', dtype=torch.int64),
                     past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        tokens, logits = [], []
        eos = hf.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for step in range(16):
            logits.append(out.logits[:, -1].cpu().clone())
            token = int(out.logits[:, -1].argmax())
            tokens.append(token)
            if token in eos or step == 15:
                break
            pos = len(ids) + step + 1
            out = hf(input_ids=torch.tensor([[token]], device='cuda'),
                     attention_mask=torch.ones(1, pos, device='cuda', dtype=torch.int64),
                     past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        fixture['generations'].append(dict(name=case['name'], prompt_ids=ids,
            tokens=tokens, logits=logits, block_replay_exact=True,
            block_first_logits=metric(logits[0], case['logits'][0]),
            block_tokens_exact=tokens == case['tokens']))
        print('stream oracle', case['name'], tokens, flush=True)
        del out, cache
    fixture['completed'] = True
    torch.save(fixture, args.output_dir/'oracle.pt')
    (args.output_dir/'oracle.json').write_text(json.dumps(
        {**fixture, 'generations': [{k: v for k, v in c.items() if k != 'logits'}
                                   for c in fixture['generations']]}, indent=2)+'\n')


@torch.inference_mode()
def verify(args, metadata):
    assert args.reference is not None
    fixture = torch.load(args.reference, map_location='cpu', weights_only=True)
    assert fixture['completed'] and fixture['metadata'] == metadata, 'Incomplete oracle or environment/checkpoint drift'
    report = dict(completed=False, passed=False, metadata=metadata, checks=[], cases=[], generations=[],
        reference_sha256=sha(args.reference), source_sha256=source_hashes(),
        attn_implementation=args.attn_implementation,
        prefill_backend='mpk_chunk64' if args.prefill_mode == 'mpk' else 'pytorch_block',
        thresholds=dict(relative_rms=.02, reference_peak=.05, absolute_floor=1e-5, exact_tokens=True))
    stream_fixture = None
    if args.stream_reference:
        assert args.prefill_mode == 'block', 'Stream oracle applies to the stream/block suite'
        stream_fixture = torch.load(args.stream_reference, map_location='cpu', weights_only=True)
        assert stream_fixture['completed'] and stream_fixture['metadata'] == metadata
        assert stream_fixture['prefill_mode'] == 'stream'
        assert stream_fixture['block_reference_sha256'] == report['reference_sha256']
        assert [(c['name'], c['prompt_ids']) for c in stream_fixture['generations']] == [
            (c['name'], c['prompt_ids']) for c in fixture['generations']]
        report.update(stream_reference_sha256=sha(args.stream_reference),
                      stream_checks=[], stream_generations=[], same_mode_passed=False)
    def save():
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def compare(name, actual, expected):
        result = dict(name=name, **metric(actual.detach().cpu(), expected))
        report['checks'].append(result)
        return result['accepted']
    model = None
    try:
        model = Qwen35Model.from_pretrained(args.model, attn_implementation=args.attn_implementation,
                                          shard_roots=args.shard_root)
        if args.compiled_run:
            from prefill_test_utils import load_compiled_pair, load_compiled_decode
            if args.prefill_mode == 'mpk':
                load_compiled_pair(model, args.compiled_run)
            else:
                load_compiled_decode(model, args.compiled_run,
                                     allow_other_b200=args.allow_other_b200_compiled)
            report['compiled_report_sha256'] = sha(args.compiled_run/'report.json')
            report['compiled_gpu_uuid'] = json.loads((args.compiled_run/'report.json').read_text()).get('gpu_uuid')
        else:
            model.compile(args.output_dir/'kernel', prefill_chunk_size=64 if args.prefill_mode == 'mpk' else None)
        def prefill(ids):
            if args.prefill_mode == 'block' and len(ids) > 1:
                return model.prefill(ids, mode='block')
            with ExitStack() as stack:
                for op in ('linear', 'conv1d', 'scaled_dot_product_attention', 'softmax', 'embedding'):
                    stack.enter_context(patch('torch.nn.functional.'+op,
                                        side_effect=AssertionError('Unexpected PyTorch fallback: '+op)))
                return model.prefill(ids, mode=args.prefill_mode)
        for case in fixture['cases']:
            model.reset()
            tokens_equal = True
            for i, step in enumerate(case['steps']):
                logits = prefill(step['inputs'])
                compare(f"{case['name']}.{i}.logits", logits, step['logits'])
                hidden = torch.stack([layer.tensors['decoder_output'][0] for layer in model.layers]).cpu()
                for layer, (actual, expected) in enumerate(zip(hidden, step['hidden'])):
                    compare(f"{case['name']}.{i}.hidden{layer}", actual, expected)
                if not report['checks'][-65]['accepted'] and 'first_logits_failure' not in report:
                    report['first_logits_failure'] = dict(case=case['name'], step=i)
                    torch.save(dict(actual=logits.cpu(), expected=step['logits'], hidden=hidden,
                                    expected_hidden=step['hidden']), args.output_dir/'first-logits-failure.pt')
                tokens_equal &= int(logits.argmax()) == int(step['logits'].argmax()) == int(model.tensors['next_token'])
                for key, actual in selected_state(model).items():
                    compare(f"{case['name']}.{i}.{key}", actual, step['states'][key])
            report['cases'].append(dict(name=case['name'], next_tokens_exact=tokens_equal))
            save()
            print(case['name'], 'tokens', tokens_equal, flush=True)
        for mode in (('mpk', 'block') if args.prefill_mode == 'mpk' else ('stream', 'block')):
            for case in fixture['generations']:
                model.reset()
                actual = model.generate(case['prompt_ids'], 16, prefill_mode=mode)
                row = dict(name=case['name'], prefill_mode=mode, actual=actual, expected=case['tokens'], tokens_exact=actual==case['tokens'])
                report['generations'].append(row)
                model.reset()
                logits = (prefill(case['prompt_ids']) if mode == 'mpk'
                              else model.prefill(case['prompt_ids'], mode=mode))
                for i, expected in enumerate(case['logits']):
                    if i:
                        logits = model.step(case['tokens'][i-1])
                    compare(f"generation.{mode}.{case['name']}.{i}.logits", logits, expected)
                print('generation', mode, case['name'], row['tokens_exact'], actual, flush=True)
                save()
        if stream_fixture is not None:
            for case in stream_fixture['generations']:
                model.reset()
                actual = model.generate(case['prompt_ids'], 16, prefill_mode='stream')
                row = dict(name=case['name'], prefill_mode='stream', actual=actual,
                           expected=case['tokens'], tokens_exact=actual == case['tokens'])
                report['stream_generations'].append(row)
                model.reset()
                logits = model.prefill(case['prompt_ids'], mode='stream')
                for i, expected in enumerate(case['logits']):
                    if i:
                        logits = model.step(case['tokens'][i-1])
                    report['stream_checks'].append(dict(
                        name=f"generation.stream.{case['name']}.{i}.logits",
                        **metric(logits.detach().cpu(), expected)))
                print('same-mode stream', case['name'], row['tokens_exact'], actual, flush=True)
                save()
        # Exact reset/replay using every decoder's recurrent/KV state.
        from test_prefill_gpu import state_hashes
        model.reset(); prefill([7, 8, 9]); first = state_hashes(model)
        model.reset(); prefill([5, 6]); model.reset(); prefill([7, 8, 9])
        assert state_hashes(model) == first
        for invalid in ([], [-1], [True], [model.config.vocab_size], [0]*1025):
            try:
                prefill(invalid)
            except ValueError:
                assert state_hashes(model) == first
            else:
                raise AssertionError('Invalid input accepted')
        report.update(completed=True, reset_replay_exact=True, invalid_input_atomic=True,
                      decode_no_torch_fallback=True, no_torch_fallback=args.prefill_mode == 'mpk')
        report['summary'] = dict(total=len(report['checks']), accepted=sum(c['accepted'] for c in report['checks']),
            exact=sum(c['exact'] for c in report['checks']),
            max_relative_rms=max(c['relative_rms'] for c in report['checks']))
        report['passed'] = (all(c['accepted'] for c in report['checks'])
            and all(c['next_tokens_exact'] for c in report['cases'])
            and all(c['tokens_exact'] for c in report['generations']))
        print(report['summary'], flush=True)
        if stream_fixture is not None:
            same_mode = [c for c in report['checks'] if not c['name'].startswith('generation.stream.')]
            same_mode += report['stream_checks']
            report['same_mode_summary'] = dict(total=len(same_mode),
                accepted=sum(c['accepted'] for c in same_mode), exact=sum(c['exact'] for c in same_mode),
                max_relative_rms=max(c['relative_rms'] for c in same_mode))
            report['same_mode_passed'] = (all(c['accepted'] for c in same_mode)
                and all(c['next_tokens_exact'] for c in report['cases'])
                and all(c['tokens_exact'] for c in report['generations'] if c['prefill_mode'] == 'block')
                and all(c['tokens_exact'] for c in report['stream_generations']))
            report['acceptance_basis'] = 'Same prefill mode in native HF and MPK; legacy cross-mode results retained'
            print('same-mode', report['same_mode_summary'], flush=True)
            assert report['same_mode_passed'], 'Same-mode 27B acceptance failed; see report.json'
        else:
            assert report['passed'], 'Full-model 27B acceptance failed; see report.json'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if model is not None:
            model.close()
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('oracle', 'stream-oracle', 'verify'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--shard-root', action='append', type=Path, default=[])
    p.add_argument('--attn-implementation', choices=('eager', 'sdpa'), default='eager')
    p.add_argument('--prefill-mode', choices=('mpk', 'block'), default='mpk',
                   help='Block prefill plus MPK decode can be validated on SM90 or SM100; MPK chunks require SM90')
    p.add_argument('--reference', type=Path)
    p.add_argument('--stream-reference', type=Path,
                   help='Additional native tokenwise HF oracle; retains all original cross-mode results')
    p.add_argument('--compiled-run', type=Path)
    p.add_argument('--allow-other-b200-compiled', action='store_true',
                   help='Reuse compatible SM100 code from another B200; HF oracle must still match the current GPU')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.allow_other_b200_compiled:
        assert args.phase == 'verify' and args.compiled_run and args.prefill_mode == 'block'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() in ((9, 0), (10, 0))
    if args.prefill_mode == 'mpk':
        assert torch.cuda.get_device_capability() == (9, 0), 'MPK chunk prefill currently requires SM90'
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert uuid.lower().removeprefix('gpu-') == os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    assert str(torch.__version__) == '2.7.0+cu128' and transformers.__version__ == '5.14.1'
    manifest = json.loads(args.manifest.read_text())
    assert manifest['revision'] == REVISION and manifest['repo'] == 'Qwen/Qwen3.8-27B'
    for name, item in manifest['files'].items():
        assert sha(args.model/name) == item['sha256'], name
    metadata = dict(gpu_uuid=uuid, torch=str(torch.__version__), transformers=transformers.__version__,
        compute_capability=list(torch.cuda.get_device_capability()),
        gpu_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        attn_implementation=args.attn_implementation, revision=REVISION,
        checkpoint_sha256={name: item['sha256'] for name, item in manifest['files'].items()},
        hf_source_sha256=sha(inspect.getfile(hf_module)))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    {'oracle': freeze, 'stream-oracle': freeze_stream, 'verify': verify}[args.phase](args, metadata)


if __name__ == '__main__':
    main()
