"""Real-checkpoint batched decoders vs separately executed single-row graphs.

Covers GDN, full attention, residuals/MLP, multiple decode steps, row
permutations and reset/replay. Single-row outputs are the regression oracle;
this does not replace the independent HF full-model numerical acceptance.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from mirage.mpk.models.qwen3_5.config import Qwen35TextConfig
from mirage.mpk.models.qwen3_5.decoder import Qwen35DecoderProbe


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 0)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert uuid.lower().removeprefix('gpu-') == os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(381127)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    c = Qwen35TextConfig.from_json(args.model/'config.json')
    mapping = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    report = dict(completed=False, passed=False, batch_size=args.batch_size, gpu_uuid=uuid, checks=[])
    probe = None
    try:
        for layer in (0, 3):
            prefix = f'model.language_model.layers.{layer}.'
            weights = {}
            for shard in sorted({s for n, s in mapping.items() if n.startswith(prefix)}):
                with safe_open(args.model/shard, framework='pt') as reader:
                    for name in reader.keys():
                        if name.startswith(prefix):
                            weights[name.removeprefix('model.language_model.')] = reader.get_tensor(name).to(torch.bfloat16)
            inputs = torch.randn(3, args.batch_size, c.hidden_size, device='cuda', dtype=torch.bfloat16)*.1
            names = ('normalized', 'output', 'decoder_output', 'state', 'conv_state') if layer == 0 else (
                'normalized', 'query', 'key', 'output', 'decoder_output', 'key_cache', 'value_cache')
            expected = []
            probe = Qwen35DecoderProbe(c, weights, layer=layer)
            probe.compile(args.output_dir/f'layer{layer}-single')
            for row in range(args.batch_size):
                probe.reset()
                samples = []
                for x in inputs[:, row]:
                    probe.step(x[None])
                    samples.append({name: probe.tensors[name].cpu().clone() for name in names})
                expected.append(samples)
            probe.close()
            del probe
            probe = None
            gc.collect(); torch.cuda.empty_cache()
            probe = Qwen35DecoderProbe(c, weights, layer=layer, batch_size=args.batch_size)
            probe.compile(args.output_dir/f'layer{layer}-batch')
            saved = None
            for trial, order in [('original', list(range(args.batch_size))),
                                 ('permuted', list(reversed(range(args.batch_size)))),
                                 ('replay', list(range(args.batch_size)))]:
                probe.reset()
                for step, x in enumerate(inputs):
                    probe.step(x[order])
                    for row, original in enumerate(order):
                        for name in names:
                            actual = probe.tensors[name][row].cpu().float().reshape(-1)
                            ref = expected[original][step][name].float().reshape(-1)
                            diff = actual-ref
                            rms = float(diff.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12))
                            peak = float(diff.abs().max())
                            accepted = bool(torch.isfinite(actual).all()) and rms <= .01 and peak <= .03*float(ref.abs().max())+1e-5
                            report['checks'].append(dict(layer=layer, trial=trial, step=step, row=row,
                                tensor=name, exact=bool(torch.equal(actual,ref)), relative_rms=rms, max_abs=peak, accepted=accepted))
                            assert accepted, report['checks'][-1]
                current = {n:probe.tensors[n].clone() for n in names}
                if trial == 'original':
                    saved = current
                elif trial == 'replay':
                    for n in names: assert torch.equal(current[n], saved[n]), n
                else:
                    for n in names: assert torch.equal(current[n], saved[n][order]), n
            probe.close(); del probe
            probe = None
            gc.collect(); torch.cuda.empty_cache()
            print('PASS batch',args.batch_size,'layer',layer,'checks',len(report['checks']),flush=True)
        report.update(completed=True, passed=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if probe is not None: probe.close()
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
