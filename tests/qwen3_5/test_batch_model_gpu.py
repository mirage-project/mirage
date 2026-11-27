"""Freeze independent native HF batch oracles; verify MPK batches on one B200.

Numerical acceptance keeps the existing 2% RMS / 5% peak model thresholds.
Execution/isolation checks are reported separately from HF numeric parity.
"""
import argparse
from contextlib import ExitStack
import gc
import json
import inspect
import transformers
import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module
import os
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from mirage.mpk.models.qwen3_5.model import Qwen35Model
from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader
from benchmark_qwen38_gpu import prompts, sha
from test_qwen38_model_gpu import source_hashes


def snapshot(model, *, cache=None):
    result = {}
    if cache is not None:
        for i in (0, 3, 62, 63):
            layer = cache.layers[i]
            if i in (0, 62):
                result[f'{i}.state'] = layer.recurrent_states[0].cpu().clone()
                result[f'{i}.conv_state'] = layer.conv_states[0][..., -3:].cpu().clone()
            else:
                result[f'{i}.key_cache'] = layer.keys.transpose(1, 2).cpu().clone()
                result[f'{i}.value_cache'] = layer.values.transpose(1, 2).cpu().clone()
    else:
        for i in (0, 3, 62, 63):
            for name in (('state', 'conv_state') if i in (0, 62) else ('key_cache', 'value_cache')):
                t = model.layers[i].tensors[name]
                if model.batch_size == 1: t = t.unsqueeze(0)
                if name.endswith('_cache'): t = t[:, :model.position]
                result[f'{i}.{name}'] = t.cpu().clone()
    return result


def compare(actual, expected):
    a, b = actual.float(), expected.float()
    assert a.shape == b.shape, (a.shape, b.shape)
    delta = a-b
    rms = float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))
    peak = float(delta.abs().max())
    return dict(exact=bool(torch.equal(a,b)), relative_rms=rms, max_abs=peak,
                accepted=bool(torch.isfinite(a).all()) and rms <= .02 and peak <= .05*float(b.abs().max())+1e-5)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('freeze', 'verify'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--oracle-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 3, 4, 8, 16, 32])
    p.add_argument('--append-output', action='store_true',
                   help='Add new batch directories to an existing output root; never overwrite a batch')
    p.add_argument('--stop-on-numerical-failure', action='store_true',
                   help='Save the failing report and stop before compiling another batch')
    args = p.parse_args()
    if args.append_output:
        assert args.phase == 'verify'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert uuid.lower().removeprefix('gpu-') == os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    metadata = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                    hf_source_sha256=sha(inspect.getfile(hf_module)),
                    checkpoint_manifest_sha256=sha(args.model/'download_manifest.json'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = None
    strict_failed = False
    if args.phase == 'freeze':
        args.oracle_dir.mkdir(parents=True, exist_ok=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation='eager', local_files_only=True, device_map='cuda').eval()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=args.append_output)
        loader = Qwen35WeightLoader(args.model)
        weights = loader.load(weight_policy='hf_bf16')
    try:
        for batch in args.batch_sizes:
            path = args.oracle_dir/f'b{batch}.pt'
            plans = [(3, [1, 1, 1]), (70, [65, 1, 1, 2, 1])]
            if batch == 2: plans.append((1024, [1023, 1]))
            if args.phase == 'freeze':
                oracle = dict(batch_size=batch, gpu_uuid=uuid, metadata=metadata, cases=[])
            else:
                oracle = torch.load(path, map_location='cpu', weights_only=True)
                assert oracle['batch_size'] == batch and oracle['gpu_uuid'] == uuid
                assert oracle['metadata'] == metadata
                run = args.output_dir/f'b{batch}'
                run.mkdir()
                report = dict(completed=False, passed=False, batch_size=batch, gpu_uuid=uuid,
                    compute_capability=list(torch.cuda.get_device_capability()),
                    attn_implementation='eager', prefill_backend='pytorch_block',
                    source_sha256=source_hashes(), verifier_sha256=sha(__file__),
                    oracle_sha256=sha(path), metadata=metadata, checks=[],
                    thresholds=dict(relative_rms=.02, peak_fraction=.05, absolute_floor=1e-5))
                model = Qwen35Model(loader.config, weights, batch_size=batch)
                model.compile(run/'kernel')
                print('COMPILED', batch, flush=True)
            for index, (length, plan) in enumerate(plans):
                rows = prompts(tokenizer, length, batch) if length > 3 else [[101+row, 231+row, 355+row] for row in range(batch)]
                if args.phase == 'freeze':
                    case = dict(rows=rows, plan=plan, snapshots=[])
                    oracle['cases'].append(case)
                    cache = None
                    model.model.rope_deltas = None
                else:
                    case = oracle['cases'][index]
                    assert case['rows'] == rows and case['plan'] == plan
                    model.reset()
                offset = 0
                for chunk_index, chunk in enumerate(plan):
                    tokens = [row[offset:offset+chunk] for row in rows]
                    offset += chunk
                    if args.phase == 'freeze':
                        out = model(input_ids=torch.tensor(tokens, device='cuda'),
                            attention_mask=torch.ones(batch, offset, dtype=torch.int64, device='cuda'),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                        cache = out.past_key_values
                        snap = snapshot(model, cache=cache)
                        snap['logits'] = out.logits[:, -1].cpu().clone()
                        case['snapshots'].append(snap)
                    else:
                        with ExitStack() as stack:
                            if chunk == 1:
                                for name in ('linear', 'embedding', 'conv1d', 'scaled_dot_product_attention'):
                                    stack.enter_context(patch('torch.nn.functional.'+name, side_effect=AssertionError('Torch decode fallback: '+name)))
                            logits = model.prefill(tokens[0] if batch == 1 else tokens)
                        snap = snapshot(model)
                        snap['logits'] = logits.cpu().clone()
                        ref = case['snapshots'][chunk_index]
                        for name in snap:
                            for row in range(batch):
                                check = compare(snap[name][row], ref[name][row])
                                report['checks'].append(dict(case=index, position=offset, row=row, tensor=name, **check))
                                assert bool(torch.isfinite(snap[name][row]).all()), (index, offset, row, name)
                        report.setdefault('greedy_matches', []).append(dict(case=index, position=offset,
                            matches=(snap['logits'].argmax(-1) == ref['logits'].argmax(-1)).tolist()))
                if args.phase == 'verify':
                    # Permuting complete request histories must permute all output and state rows exactly.
                    saved = snap
                    model.reset()
                    order = list(reversed(range(batch)))
                    reordered = [rows[i] for i in order]
                    offset = 0
                    for chunk in plan:
                        tokens = [row[offset:offset+chunk] for row in reordered]
                        offset += chunk
                        model.prefill(tokens[0] if batch == 1 else tokens)
                    again = snapshot(model)
                    again['logits'] = model.tensors['logits'].cpu()
                    for name in again:
                        assert torch.equal(again[name], saved[name][order]), ('permutation/replay',batch,index,name)
                    assert model.position == length
                    print('PASS execution/isolation',batch,plan,flush=True)
            if args.phase == 'freeze':
                torch.save(oracle, path)
                print('FROZEN', batch, flush=True)
            else:
                report.update(completed=True, passed=True, numerical_acceptance=all(c['accepted'] for c in report['checks']),
                    greedy_token_acceptance=all(all(c['matches']) for c in report['greedy_matches']),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
                report['strict_acceptance'] = report['numerical_acceptance'] and report['greedy_token_acceptance']
                (run/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print('RESULT',batch,'numerical',report['numerical_acceptance'],
                      'passed',sum(c['accepted'] for c in report['checks']),'/',len(report['checks']),
                      'greedy',report['greedy_token_acceptance'],flush=True)
                model.close(); model = None
                gc.collect(); torch.cuda.empty_cache()
                if not report['strict_acceptance']:
                    strict_failed = True
                    if args.stop_on_numerical_failure:
                        raise SystemExit(f'Batch {batch} failed strict HF acceptance; further compilation stopped')
    finally:
        if args.phase == 'verify' and model is not None:
            model.close()
            (run/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if strict_failed:
        raise SystemExit('Strict HF acceptance failed; see the per-batch reports')


if __name__ == '__main__':
    main()
