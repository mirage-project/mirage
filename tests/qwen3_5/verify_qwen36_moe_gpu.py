"""Independent SM100 probe of Qwen3.6 BF16 expert reduction and shared gate."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from mirage.mpk.models.qwen3_5.moe_reference import sum_expert_slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--oracle-layers', type=Path,
                        help='Replay saved full-model HF router logits through standalone top-k')
    parser.add_argument('--moe-snapshot', type=Path,
                        help='Replay one captured full-model expert reduction')
    parser.add_argument('--hf-oracle', type=Path,
                        help='Replay a frozen HF prefill router probe')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError('Expose exactly one B200 GPU')
    repo = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    library = args.output_dir / 'qwen36_moe_isolated.so'
    command = [str(Path('/usr/local/cuda-12.8/bin/nvcc')), '-std=c++17', '-O3',
               '-arch=sm_100a', '--shared', '-Xcompiler', '-fPIC',
               '-DMIRAGE_BACKEND_USE_CUDA',
               *('-I' + str(repo / path) for path in
                 ('include', 'include/mirage/persistent_kernel',
                  'deps/cutlass/include', 'deps/json/include')),
               str(repo / 'tests/qwen3_5/qwen36_moe_isolated.cu'), '-o', str(library)]
    with (args.output_dir / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    native = ctypes.CDLL(str(library))
    native.launch_moe_sum.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int]
    native.launch_shared_gate.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    native.launch_topk.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    native.launch_router_softmax.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int]
    checks = []

    for batch in (1, 8, 16):
        for seed in (0, 1, 2):
            torch.manual_seed(seed + batch * 100)
            output = torch.randn(batch, 8, 2048, device='cuda').bfloat16()
            shared = torch.randn(batch, 2048, device='cuda').bfloat16()
            scores = torch.randn(batch, 8, device='cuda').bfloat16().float()
            routing = torch.zeros(256, batch, dtype=torch.int32, device='cuda')
            for row in range(batch):
                for slot, expert in enumerate(torch.randperm(256)[:8].tolist()):
                    routing[expert, row] = slot + 1
            actual = torch.empty_like(shared)
            error = native.launch_moe_sum(output.data_ptr(), scores.data_ptr(),
                                          routing.data_ptr(), shared.data_ptr(),
                                          actual.data_ptr(), batch)
            if error:
                raise RuntimeError(f'CUDA MoE sum launch failed: {error}')
            torch.cuda.synchronize()
            expected = sum_expert_slots(output, scores, routing, shared)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            gate = torch.randn(batch, 64, device='cuda').bfloat16()
            actual_gate = torch.empty_like(shared)
            error = native.launch_shared_gate(gate.data_ptr(), shared.data_ptr(),
                                              actual_gate.data_ptr(), batch)
            if error:
                raise RuntimeError(f'CUDA shared gate launch failed: {error}')
            torch.cuda.synchronize()
            expected_gate = torch.sigmoid(gate[:, :1]) * shared
            torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)

            logits = torch.full((batch, 256), -5., dtype=torch.bfloat16, device='cuda')
            for row in range(batch):
                for slot, expert in enumerate(torch.randperm(256)[:8].tolist()):
                    logits[row, expert] = 8 - (slot + 1) // 2
            probabilities = torch.softmax(logits.float(), dim=-1)
            values, indices = probabilities.topk(8, dim=-1)
            expected_scores = (values / values.sum(dim=-1, keepdim=True)).bfloat16().float()
            actual_scores = torch.empty((batch, 8), dtype=torch.float32, device='cuda')
            actual_routing = torch.empty((256, batch), dtype=torch.int32, device='cuda')
            active = torch.empty(257, dtype=torch.int32, device='cuda')
            logits_input = logits.clone()  # The kernel zeros this buffer.
            error = native.launch_topk(logits_input.data_ptr(), actual_scores.data_ptr(),
                                       actual_routing.data_ptr(), active.data_ptr(), batch)
            if error:
                raise RuntimeError(f'CUDA MoE top-k launch failed: {error}')
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
            expected_routing = torch.zeros_like(actual_routing)
            for row in range(batch):
                for slot, expert in enumerate(indices[row].tolist()):
                    expected_routing[expert, row] = slot + 1
            torch.testing.assert_close(actual_routing, expected_routing, rtol=0, atol=0)
            expected_active = set(indices.flatten().tolist())
            assert active[256].item() == len(expected_active)
            assert set(active[:len(expected_active)].tolist()) == expected_active
            checks.append({'batch': batch, 'seed': seed,
                           'moe_sum_exact': True, 'shared_gate_exact': True,
                           'topk_exact': True})
            print(f'PASS B{batch} seed={seed}', flush=True)
    source = repo / 'tests/qwen3_5/qwen36_moe_isolated.cu'
    topk_header = repo / 'include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh'
    report = {
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'topk_header_sha256': hashlib.sha256(topk_header.read_bytes()).hexdigest(),
        'checks': checks, 'passed': True,
    }
    if args.hf_oracle is not None:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            oracle = torch.load(args.hf_oracle, map_location='cpu', weights_only=True)
        case = oracle['cases'][0]
        probe = case['prefill_probe']
        batch = case['batch']
        logits = probe['hf_router_logits'].cuda().contiguous()
        expected_scores = probe['hf_router_scores'].cuda()
        expected_ids = probe['hf_router_indices'].cpu()
        scores = torch.empty((batch, 8), dtype=torch.float32, device='cuda')
        routing = torch.empty((256, batch), dtype=torch.int32, device='cuda')
        active = torch.empty(257, dtype=torch.int32, device='cuda')
        status = native.launch_topk(logits.data_ptr(), scores.data_ptr(),
                                    routing.data_ptr(), active.data_ptr(), batch)
        if status:
            raise RuntimeError(f'Frozen HF top-k launch failed: {status}')
        torch.cuda.synchronize()
        expected_routing = torch.zeros_like(routing)
        for row in range(batch):
            for slot, expert in enumerate(expected_ids[row].tolist()):
                expected_routing[expert, row] = slot + 1
        report['frozen_router'] = {
            'scores_exact': bool(torch.equal(scores, expected_scores.float())),
            'route_exact': bool(torch.equal(routing, expected_routing)),
            'score_differences': int((scores != expected_scores).sum()),
            'route_differences': int((routing != expected_routing).sum())}
        report['passed'] = report['passed'] and all(
            report['frozen_router'][name] for name in ('scores_exact', 'route_exact'))
    if args.moe_snapshot is not None:
        snapshot = torch.load(args.moe_snapshot, map_location='cpu', weights_only=True)
        mpk = snapshot['mpk']
        output = mpk['expert_out'].cuda().contiguous()
        scores = mpk['router_scores'].cuda().contiguous()
        routing = mpk['routing_indices'].cuda().contiguous()
        shared = torch.zeros((8, 2048), dtype=torch.bfloat16, device='cuda')
        actual = torch.empty_like(shared)
        error = native.launch_moe_sum(output.data_ptr(), scores.data_ptr(),
                                      routing.data_ptr(), shared.data_ptr(),
                                      actual.data_ptr(), 8)
        if error:
            raise RuntimeError(f'Real MoE sum launch failed: {error}')
        torch.cuda.synchronize()
        expected = snapshot['hf']['experts'].cuda()
        mismatch = actual != expected
        report['real_moe_sum'] = {'mismatch_count': int(mismatch.sum()),
                                  'first_differences': mismatch.nonzero()[:8].tolist()}
        report['passed'] = report['passed'] and not bool(mismatch.any())
        print(report['real_moe_sum'], flush=True)
    if args.oracle_layers is not None:
        oracle = torch.load(args.oracle_layers, map_location='cpu', weights_only=True)
        logits = oracle['router_logits']
        expected_scores = oracle['router_scores']
        expected_route = oracle['route']
        if logits.ndim != 3 or logits.shape[1:] != (8, 256):
            raise ValueError('Expected B8 router logits [layers, 8, 256]')
        real = []
        focus = None
        softmax_differences = 0
        softmax_focus = None
        for layer in range(logits.shape[0]):
            logits_input = logits[layer].cuda().contiguous()
            probabilities = torch.empty((8, 256), dtype=torch.float32, device='cuda')
            error = native.launch_router_softmax(logits_input.data_ptr(),
                                                 probabilities.data_ptr(), 8)
            if error:
                raise RuntimeError(f'CUDA softmax failed at layer {layer}: {error}')
            torch.cuda.synchronize()
            native_probabilities = torch.softmax(logits[layer].cuda().float(), dim=-1)
            softmax_differences += int((probabilities != native_probabilities).sum())
            if layer == 3:
                ids = [183, 163, 224, 77, 69, 196, 61, 245]
                softmax_focus = {'hf': native_probabilities[5, ids].cpu().tolist(),
                                 'standalone': probabilities[5, ids].cpu().tolist()}
            scores = torch.empty((8, 8), dtype=torch.float32, device='cuda')
            routing = torch.empty((256, 8), dtype=torch.int32, device='cuda')
            active = torch.empty(257, dtype=torch.int32, device='cuda')
            error = native.launch_topk(logits_input.data_ptr(), scores.data_ptr(),
                                       routing.data_ptr(), active.data_ptr(), 8)
            if error:
                raise RuntimeError(f'CUDA top-k failed at layer {layer}: {error}')
            torch.cuda.synchronize()
            if layer == 3:
                reference_probs = torch.softmax(logits[layer, 5].cuda().float(), dim=-1)
                reference_topk, reference_ids = reference_probs.topk(8)
                focus = {'reference_selected_probs': reference_topk.cpu().tolist(),
                         'reference_indices': reference_ids.cpu().tolist(),
                         'standalone_selected': scores[5].cpu().tolist()}
            actual_route = torch.full((8, 8), -1, dtype=torch.int64)
            routing_cpu = routing.cpu()
            for expert in range(256):
                for row in range(8):
                    slot = int(routing_cpu[expert, row]) - 1
                    if slot >= 0:
                        actual_route[row, slot] = expert
            score_changed = (scores.bfloat16().cpu() != expected_scores[layer]).any(-1)
            route_changed = (actual_route != expected_route[layer]).any(-1)
            for row in range(8):
                if score_changed[row] or route_changed[row]:
                    real.append({'layer': layer, 'row': row,
                                 'scores_differ': bool(score_changed[row]),
                                 'route_differ': bool(route_changed[row]),
                                 'hf_scores': expected_scores[layer, row].tolist(),
                                 'standalone_scores': scores[row].bfloat16().cpu().tolist()})
        report['real_oracle'] = {'score_mismatches': sum(x['scores_differ'] for x in real),
                                 'route_mismatches': sum(x['route_differ'] for x in real),
                                 'softmax_differences': softmax_differences,
                                 'softmax_focus': softmax_focus,
                                 'first': real[:20], 'focus': focus}
        report['passed'] = report['passed'] and not (
            report['real_oracle']['score_mismatches'] or
            report['real_oracle']['route_mismatches'] or softmax_differences)
        print({'score_mismatches': report['real_oracle']['score_mismatches'],
               'route_mismatches': report['real_oracle']['route_mismatches'],
               'softmax_differences': softmax_differences,
               'softmax_focus': softmax_focus,
               'focus': focus})
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise AssertionError('Standalone top-k differs from the saved HF oracle')


if __name__ == '__main__':
    main()
