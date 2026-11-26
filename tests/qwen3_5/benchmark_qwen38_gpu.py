"""Single-B200 true fixed-batch text benchmark for HF and MPK.

Measure synchronous greedy forward loops with a fresh cache per sample and
CPU token delivery at every step. A batch is one model call with B distinct
rows; independent single-request calls are never reported as a larger batch.
Loading, compilation, tokenization, warmup and result checks are not timed.
"""
import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module

REVISION = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def distribution(values):
    values = sorted(values)
    return dict(n=len(values), median=statistics.median(values),
                p90=values[math.ceil(.9*len(values))-1], min=min(values),
                max=max(values), mean=statistics.mean(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.)


def telemetry(uuid):
    return subprocess.check_output([
        'nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,driver_version,'
        'temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,'
        'memory.used,utilization.gpu', '--format=csv'], text=True).strip()


def gpu_compute_pids(uuid):
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    pids = set()
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 2:
            raise RuntimeError(f'Unexpected nvidia-smi compute-process row: {line!r}')
        if fields[0].lower() == uuid.lower():
            pids.add(int(fields[1]))
    return pids


def assert_exclusive_gpu(uuid, baseline_pids=frozenset()):
    others = sorted(gpu_compute_pids(uuid) - set(baseline_pids) - {os.getpid()})
    if others:
        raise RuntimeError(f'New compute process entered the allocated GPU; '
                           f'stop this benchmark: {others}')


def prompts(tokenizer, length, batch_size):
    body = ('A research team is comparing language model inference on one GPU. '
            'They record latency, throughput, memory usage and reproducibility. '
            'Requests have separate histories and must never share their cache. '
            'Explain the tradeoffs clearly and give a practical example. ')
    tail = tokenizer(body, add_special_tokens=False)['input_ids']
    result = []
    for row in range(batch_size):
        head = tokenizer(f'Benchmark request {row + 1}. ', add_special_tokens=False)['input_ids']
        result.append((head + tail*math.ceil(length/len(tail)))[:length])
    assert all(len(row) == length for row in result)
    assert len({tuple(row) for row in result}) == batch_size
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('hf', 'mpk'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--compiled-run', type=Path)
    p.add_argument('--compiled-root', type=Path, help='Validated runs b1/, b2/, ...')
    p.add_argument('--hf-reference', type=Path)
    p.add_argument('--validation-report', type=Path, required=True)
    p.add_argument('--gpu-baseline', type=Path,
                   help='Guard admission JSON for pre-existing idle GPU processes')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 960])
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--prefill-mode', choices=('block', 'stream'), default='block')
    args = p.parse_args()
    assert args.warmup >= 1 and args.repeats >= 2 and args.new_tokens >= 2
    assert len(set(args.batch_sizes)) == len(args.batch_sizes)
    assert all(b >= 1 for b in args.batch_sizes)
    assert all(n >= 2 and n+args.new_tokens-1 <= 1024 for n in args.lengths)
    if args.backend == 'mpk':
        assert all(b <= 32 for b in args.batch_sizes)
        assert args.hf_reference
        assert args.compiled_run is None or len(args.batch_sizes) == 1
        assert not (args.compiled_run and args.compiled_root)
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
    uuid = os.environ['QWEN35_EXPECTED_GPU_UUID']
    properties = torch.cuda.get_device_properties(0)
    assert str(properties.uuid).lower().removeprefix('gpu-') == uuid.lower().removeprefix('gpu-')
    gpu_baseline = None
    baseline_pids = frozenset()
    if args.gpu_baseline is not None:
        gpu_baseline = json.loads(args.gpu_baseline.read_text())
        assert gpu_baseline['gpu_uuid'].lower().removeprefix('gpu-') == \
            uuid.lower().removeprefix('gpu-')
        baseline_list = gpu_baseline['baseline_pids']
        assert isinstance(baseline_list, list)
        assert all(type(pid) is int and pid > 0 for pid in baseline_list)
        baseline_pids = frozenset(baseline_list)
        assert len(baseline_pids) == len(baseline_list)
        assert os.getpid() not in baseline_pids
        memory_mib = gpu_baseline['memory_used_mib']
        utilization_percent = gpu_baseline['utilization_gpu_percent']
        assert type(memory_mib) in (int, float) and math.isfinite(memory_mib)
        assert type(utilization_percent) in (int, float) and math.isfinite(utilization_percent)
        assert 0 <= memory_mib <= 10 * 1024
        assert 0 <= utilization_percent <= 5
    assert_exclusive_gpu(uuid, baseline_pids)
    assert str(torch.__version__) == '2.7.0+cu128' and transformers.__version__ == '5.14.1'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(38271116)
    manifest = json.loads(args.manifest.read_text())
    assert (manifest['repo'], manifest['revision']) == ('Qwen/Qwen3.8-27B', REVISION)
    for name, item in manifest['files'].items():
        assert sha(args.model/name) == item['sha256'], name
    validation = json.loads(args.validation_report.read_text())
    assert validation['completed'] and validation['passed']
    if validation.get('phase') == 'mpk':
        # A completed B32 continuation replay is the acceptance gate for this
        # fixed-batch benchmark. Check its model, settings and production code
        # rather than treating a report from another build as current evidence.
        assert validation['checkpoint_manifest_sha256'] == sha(args.manifest)
        assert validation['settings'] == dict(batch_size=32,
            prompt_lengths=[128, 512, 960], new_tokens=64, prefill='block',
            attention='eager', dtype='bfloat16', tf32=False)
        assert args.batch_sizes == [32] and args.lengths == [128, 512, 960]
        assert args.new_tokens == 64 and args.prefill_mode == 'block'
        source_root = Path(__file__).resolve().parents[2]
        source_digests = validation['source_sha256']
        assert len(source_digests) >= 20
        for name, digest in source_digests.items():
            assert sha(source_root / name) == digest, name
        assert validation['verifier_sha256'] == sha(
            source_root / 'tests/qwen3_5/verify_benchmark_continuation_gpu.py')
        replay_path = args.validation_report.parent.parent / 'hf-replay/report.json'
        replay = json.loads(replay_path.read_text())
        assert sha(replay_path) == validation['hf_replay_report_sha256']
        assert replay['completed'] and replay['passed'] and replay['phase'] == 'hf-replay'
        assert replay['gpu_uuid'].lower().removeprefix('gpu-') == \
            validation['gpu_uuid'].lower().removeprefix('gpu-')
        assert replay['checkpoint_manifest_sha256'] == sha(args.manifest)
        assert replay['hf_source_sha256'] == sha(inspect.getfile(hf_module))
        assert replay['verifier_sha256'] == validation['verifier_sha256']
        assert replay['settings'] == dict(batch_size=32,
            prompt_lengths=[128, 512, 960], new_tokens=64, attention='eager',
            dtype='bfloat16', tf32=False, prefill='native_hf')
        assert len(replay['cases']) == 3
        assert {case['prompt_length'] for case in replay['cases']} == {128, 512, 960}
        assert all(case['all_exact'] and case['exact_rows'] == 32
                   for case in replay['cases'])
        assert len(validation['cases']) == 3
        assert {case['prompt_length'] for case in validation['cases']} == {128, 512, 960}
        for case in validation['cases']:
            assert case['all_exact'] and case['exact_rows'] == 32
            assert len(case['actual_tokens']) == len(case['rows']) == 32
            assert all(len(tokens) == 64 for tokens in case['actual_tokens'])
            assert all(row['exact'] and row['matching_tokens'] == 64
                       for row in case['rows'])
            assert sum(row['matching_tokens'] for row in case['rows']) == 2048
            matched_replay = next(item for item in replay['cases']
                                  if item['prompt_length'] == case['prompt_length'])
            assert case['actual_tokens'] == matched_replay['actual_tokens']
        prior_acceptance_basis = 'B32 same-B200 native-HF and MPK greedy-token replay'
        prior_strict_numerical_acceptance = None
        prior_same_prefill_mode_acceptance = None
        prior_token_acceptance = True
    else:
        assert validation['metadata']['revision'] == REVISION
        assert validation['metadata']['checkpoint_sha256'] == {
            name: item['sha256'] for name, item in manifest['files'].items()}
        prior_acceptance_basis = validation.get('acceptance_basis', 'Original full-model suite')
        prior_strict_numerical_acceptance = validation['passed']
        prior_same_prefill_mode_acceptance = validation.get('same_mode_passed')
        prior_token_acceptance = None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, backend=args.backend, gpu_uuid=uuid,
        gpu_name=properties.name, gpu_sms=properties.multi_processor_count,
        torch=str(torch.__version__), transformers=transformers.__version__,
        cuda=torch.version.cuda, hf_source_sha256=sha(inspect.getfile(hf_module)),
        benchmark_sha256=sha(__file__), checkpoint=manifest,
        validation_report_sha256=sha(args.validation_report),
        gpu_baseline_sha256=sha(args.gpu_baseline) if args.gpu_baseline else None,
        gpu_baseline=gpu_baseline,
        prior_strict_numerical_acceptance=prior_strict_numerical_acceptance,
        prior_same_prefill_mode_acceptance=prior_same_prefill_mode_acceptance,
        prior_token_acceptance=prior_token_acceptance,
        prior_acceptance_basis=prior_acceptance_basis,
        cpu_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        settings=dict(batch_sizes=args.batch_sizes, prompt_lengths=args.lengths,
            new_tokens=args.new_tokens, warmup=args.warmup, repeats=args.repeats,
            attention='eager', dtype='bfloat16', tf32=False, eos_stopping=False,
            torch_compile=False, cuda_graphs=False, padding=False,
            prefill=args.prefill_mode if args.backend == 'mpk' else 'native_hf',
            timing='synchronized perf_counter; greedy forward loop with CPU tokens each step',
            ttft='fresh-cache/reset + prefill + argmax and CPU delivery of first B tokens',
            tpot='decode wall time / (new_tokens - 1); latency per sequence',
            decode_tps='batch_size * (new_tokens - 1) / decode wall seconds',
            output_tps='batch_size * new_tokens / end-to-end wall seconds',
            excluded='loading, compilation, tokenization, warmup, correctness checks',
            memory='one model resident; PyTorch peak allocated/reserved; HF includes unused vision weights; MPK is text-only; not total CUDA process memory',
            inputs='distinct equal-length text token rows; ready on GPU for HF, host IDs for MPK',
            order='ascending batch size, then ascending prompt length'),
        telemetry_start=telemetry(uuid), cases=[])

    def save():
        temporary = args.output_dir/'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output_dir/'report.json')

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = None
    reference = None
    try:
        if args.backend == 'hf':
            model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model,
                dtype=torch.bfloat16, attn_implementation='eager',
                local_files_only=True, device_map='cuda').eval()
            # Preserve native HF device placement: moving the first registered
            # submodule to CPU changes model.device and generation input routing.
            # Text forward does not execute this unused vision submodule.
            visual = getattr(model.model, 'visual', None)
            report['unused_vision_weights_gpu_bytes'] = (sum(
                t.numel()*t.element_size() for t in list(visual.parameters())+list(visual.buffers()))
                if visual is not None else 0)
            gdn = model.model.language_model.layers[0].linear_attn
            report['hf_gdn'] = {name: getattr(gdn, name).__name__ for name in
                ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule', 'causal_conv1d_update')}
            report['hf_fast_path_available'] = hf_module.is_fast_path_available
            report['unused_vision_weights_on_gpu'] = visual is not None
        else:
            from mirage.mpk.models.qwen3_5.model import Qwen35Model
            from prefill_test_utils import load_compiled_decode
            reference = json.loads(args.hf_reference.read_text())
            assert reference['completed'] and reference['backend'] == 'hf'
            assert reference['checkpoint'] == manifest and reference['gpu_uuid'] == uuid
            assert reference['settings']['attention'] == 'eager'
            assert reference['settings']['new_tokens'] == args.new_tokens
            from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader
            from test_qwen38_model_gpu import source_hashes
            loader = Qwen35WeightLoader(args.model)
            weights = loader.load(weight_policy='hf_bf16')
            report['source_sha256'] = source_hashes()
            report['hf_reference_sha256'] = sha(args.hf_reference)

        def run_hf(ids, masks):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.model.rope_deltas = None
            output = model(input_ids=ids, attention_mask=masks[0],
                           use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            next_ids = output.logits[:, -1].argmax(-1).unsqueeze(1)
            tokens = [next_ids[:, 0].tolist()]
            torch.cuda.synchronize()
            first = time.perf_counter()
            for step in range(1, args.new_tokens):
                output = model(input_ids=next_ids, attention_mask=masks[step],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                next_ids = output.logits[:, -1].argmax(-1).unsqueeze(1)
                tokens.append(next_ids[:, 0].tolist())
            torch.cuda.synchronize()
            end = time.perf_counter()
            assert bool(torch.isfinite(output.logits).all()), 'Nonfinite final logits'
            return start, first, end, [list(row) for row in zip(*tokens)]

        def run_mpk(rows):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.reset()
            model.prefill(rows[0] if model.batch_size == 1 else rows, mode=args.prefill_mode)
            tokens = [model.tensors['next_token'].tolist()]
            torch.cuda.synchronize()
            first = time.perf_counter()
            for _ in range(1, args.new_tokens):
                model.step(tokens[-1][0] if model.batch_size == 1 else tokens[-1])
                tokens.append(model.tensors['next_token'].tolist())
            torch.cuda.synchronize()
            end = time.perf_counter()
            assert bool(torch.isfinite(model.tensors['logits']).all()), 'Nonfinite final logits'
            return start, first, end, [list(row) for row in zip(*tokens)]

        for batch in args.batch_sizes:
            if args.backend == 'mpk':
                if model is not None:
                    model.close()
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                model = Qwen35Model(loader.config, weights, attn_implementation='eager', batch_size=batch)
                compiled = args.compiled_root/f'b{batch}' if args.compiled_root else args.compiled_run
                if compiled:
                    load_compiled_decode(model, compiled)
                    report.setdefault('compiled_report_sha256', {})[str(batch)] = sha(compiled/'report.json')
                else:
                    model.compile(args.output_dir/f'kernel-b{batch}')
                print('compiled batch', batch, flush=True)
            for length in args.lengths:
                assert_exclusive_gpu(uuid, baseline_pids)
                case = dict(batch_size=batch, prompt_length=length, status='running', samples=[])
                report['cases'].append(case)
                save()
                rows = prompts(tokenizer, length, batch)
                case['prompt_ids'] = rows
                expected = None
                if reference is not None:
                    expected_case = next(c for c in reference['cases']
                        if c['batch_size'] == batch and c['prompt_length'] == length)
                    assert expected_case['status'] == 'ok' and expected_case['prompt_ids'] == rows
                    expected = expected_case['reference_tokens']
                ids = masks = None
                try:
                    if args.backend == 'hf':
                        ids = torch.tensor(rows, dtype=torch.int64, device='cuda')
                        masks = [torch.ones((batch, length+i), dtype=torch.int64, device='cuda')
                                 for i in range(args.new_tokens)]
                    baseline = None
                    case['repeat_tokens_exact'] = True
                    case['matches_hf_tokens'] = True if expected is not None else None
                    for iteration in range(-args.warmup, args.repeats):
                        torch.cuda.reset_peak_memory_stats()
                        start, first, end, tokens = (run_hf(ids, masks)
                            if args.backend == 'hf' else run_mpk(rows))
                        peak_allocated = torch.cuda.max_memory_allocated()
                        peak_reserved = torch.cuda.max_memory_reserved()
                        assert len(tokens) == batch and all(len(row) == args.new_tokens for row in tokens)
                        if baseline is None:
                            baseline = tokens
                            case['reference_tokens'] = tokens
                            if args.backend == 'hf':
                                # Check the measured loop against the native
                                # generation API outside the timed region.
                                model.model.rope_deltas = None
                                native = model.generate(input_ids=ids, attention_mask=masks[0],
                                    do_sample=False, max_new_tokens=args.new_tokens,
                                    eos_token_id=None, forced_eos_token_id=None,
                                    use_cache=True, disable_compile=True, logits_to_keep=1)
                                case['matches_native_generate'] = native[:, length:].tolist() == tokens
                                del native
                                assert case['matches_native_generate'], 'Forward loop differs from native generate'
                        exact = tokens == baseline
                        case['repeat_tokens_exact'] &= exact
                        matches = tokens == expected if expected is not None else None
                        if expected is not None:
                            case['matches_hf_tokens'] &= matches
                        if not exact or matches is False:
                            case.setdefault('token_mismatches', []).append(dict(iteration=iteration,tokens=tokens))
                        if iteration >= 0:
                            case['samples'].append(dict(iteration=iteration, ttft_ms=(first-start)*1000,
                                decode_ms=(end-first)*1000, tpot_ms=(end-first)*1000/(args.new_tokens-1),
                                total_ms=(end-start)*1000, decode_tps=batch*(args.new_tokens-1)/(end-first),
                                output_tps=batch*args.new_tokens/(end-start),
                                prefill_tps=batch*length/(first-start), peak_allocated_bytes=peak_allocated,
                                peak_reserved_bytes=peak_reserved, repeat_tokens_exact=exact,
                                matches_hf_tokens=matches))
                    keys=('ttft_ms','decode_ms','tpot_ms','total_ms','decode_tps','output_tps',
                          'prefill_tps','peak_allocated_bytes','peak_reserved_bytes')
                    case['summary'] = {key: distribution([s[key] for s in case['samples']]) for key in keys}
                    if not case['repeat_tokens_exact'] or \
                            (expected is not None and not case['matches_hf_tokens']):
                        case['status'] = 'token_mismatch'
                        raise AssertionError('Timed or warmup tokens differ; discard benchmark timings')
                    assert_exclusive_gpu(uuid, baseline_pids)
                    case['status'] = 'ok'
                    case['telemetry'] = telemetry(uuid)
                    print(args.backend, 'batch', batch, 'length', length,
                          {k: round(case['summary'][k]['median'],3) for k in
                           ('ttft_ms','tpot_ms','decode_tps','output_tps')},
                          'repeat_exact',case['repeat_tokens_exact'],
                          'matches_hf',case['matches_hf_tokens'],flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    case.update(status='oom', error=str(exc))
                    print(args.backend, 'batch', batch, 'length', length, 'OOM', flush=True)
                finally:
                    ids = masks = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
        report['completed'] = True
        report['telemetry_end'] = telemetry(uuid)
        report['all_repetitions_deterministic'] = all(c.get('repeat_tokens_exact',False)
            for c in report['cases'] if c['status'] == 'ok')
        columns = ['backend','batch_size','prompt_length','iteration','ttft_ms','decode_ms',
                   'tpot_ms','total_ms','decode_tps','output_tps','prefill_tps',
                   'peak_allocated_bytes','peak_reserved_bytes','repeat_tokens_exact','matches_hf_tokens']
        with (args.output_dir/'samples.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for case in report['cases']:
                for sample in case['samples']:
                    writer.writerow(dict(backend=args.backend, batch_size=case['batch_size'],
                                         prompt_length=case['prompt_length'], **sample))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if args.backend == 'mpk' and model is not None:
            model.close()
        save()


if __name__ == '__main__':
    main()
