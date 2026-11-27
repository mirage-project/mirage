"""Replay saved prompts against independent native HF and MPK generation.

The original HF tokens, checkpoint/source hashes, device, and numerical gates
are fixed by --reference (the live-generation report). Both implementations
generate independently; the separate logits replay uses HF's forced history.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module
from mirage.mpk.models.qwen3_5.model import Qwen35Model


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def metrics(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    assert a.shape == b.shape
    delta = a-b
    rms = float(delta.norm()/b.norm().clamp_min(1e-20))
    maximum, peak = float(delta.abs().max()), float(b.abs().max())
    return dict(exact=torch.equal(a, b), max_abs=maximum, relative_rms=rms, reference_peak=peak,
                accepted=rms <= .02 and maximum <= .05*peak+1e-5,
                mpk_argmax=int(a.argmax()), hf_argmax=int(b.argmax()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--attn-implementation', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--quick', action='store_true', help='Three diagnostic cases only')
    parser.add_argument('--prefill-mode', choices=('block', 'mpk'), default='block')
    parser.add_argument('--compiled-run', type=Path, help='Reuse a source-verified MPK chunk graph')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    backend = 'default' if args.attn_implementation == 'sdpa' else 'eager'
    original = {c['name']: c for c in reference['backends'][backend]['cases']}
    assert reference['completed']
    assert str(torch.__version__) == reference['torch'] == '2.7.0+cu128'
    assert transformers.__version__ == reference['transformers'] == '5.14.1'
    assert torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert uuid == reference['gpu_uuid']
    assert uuid.lower().removeprefix('gpu-') == os.environ['QWEN35_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-')
    assert sha(inspect.getfile(hf_module)) == reference['hf_source_sha256']
    for name, expected in reference['checkpoint_sha256'].items():
        assert sha(args.model/name) == expected, name
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo/'python/mirage/mpk/models/qwen3_5').glob('*.py'))
    sources += list((repo/'include/mirage/persistent_kernel/tasks/common').glob('qwen35*.cuh'))
    sources += [repo/'src/kernel/graph.cc', repo/'src/kernel/task_register.cc', Path(__file__).resolve()]
    report = dict(completed=False, passed=False, torch=str(torch.__version__),
        transformers=transformers.__version__, gpu_uuid=uuid, attn_implementation=args.attn_implementation,
        prefill_backend=('mpk_chunk64' if args.prefill_mode == 'mpk' else
                         'torch_'+args.attn_implementation+'_cuda'), decode_backend='mpk',
        reference_sha256=sha(args.reference), source_sha256={str(p.relative_to(repo)): sha(p) for p in sources},
        thresholds=dict(relative_rms=.02, reference_peak=.05, absolute_floor=1e-5,
                        ci_prefix=50, require_full_generation=True), cases=[])
    def save():
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    mpk = None
    try:
        mpk = Qwen35Model.from_pretrained(args.model, attn_implementation=args.attn_implementation)
        if args.compiled_run:
            from prefill_test_utils import load_compiled_pair
            load_compiled_pair(mpk, args.compiled_run)
        else:
            mpk.compile(args.output_dir/'kernel', prefill_chunk_size=64 if args.prefill_mode == 'mpk' else None)
        hf = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation, local_files_only=True, device_map='cuda').eval()
        assert hf.config.text_config._attn_implementation == args.attn_implementation
        assert hf.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule.__name__ == reference['backends'][backend]['gdn_chunk']
        for case in reference['cases']:
            if args.quick and case['name'] not in ('chinese', 'explanation.64', 'fresh_math'):
                continue
            ids = torch.tensor([case['prompt_ids']], device='cuda')
            result = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=case['length'], return_dict_in_generate=True, output_logits=True)
            expected = result.sequences[0, len(case['prompt_ids']):].tolist()
            assert expected == original[case['name']]['hf_generated'], 'Native HF reference drift: '+case['name']
            mpk.reset()
            actual = mpk.generate(case['prompt_ids'], case['length'], prefill_mode=args.prefill_mode)
            n = min(50, len(actual), len(expected))
            first = next((i for i in range(max(len(actual), len(expected)))
                          if i >= len(actual) or i >= len(expected) or actual[i] != expected[i]), None)
            row = dict(name=case['name'], group=case['group'], hf_generated=expected, mpk_generated=actual,
                       native_reference_unchanged=True, ci_tokens_compared=n,
                       ci_prefix_exact=n > 0 and actual[:n] == expected[:n], tokens_exact=actual == expected,
                       first_token_difference=first, logits=[])
            report['cases'].append(row)
            mpk.reset()
            logits = mpk.prefill(case['prompt_ids'], mode=args.prefill_mode)
            for index, expected_logits in enumerate(result.logits):
                if index:
                    logits = mpk.step(expected[index-1])
                row['logits'].append(dict(index=index, **metrics(logits, expected_logits)))
            save()
            print(case['name'], 'tokens', row['tokens_exact'], 'prefix', row['ci_prefix_exact'],
                  'logits', sum(r['exact'] for r in row['logits']), '/', len(row['logits']), flush=True)
            del result
        rows = report['cases']
        checks = [r for row in rows for r in row['logits']]
        report['summary'] = dict(cases=len(rows), ci_prefix_exact=sum(r['ci_prefix_exact'] for r in rows),
            full_50_token_cases=sum(r['ci_tokens_compared'] == 50 for r in rows),
            exact_continuations=sum(r['tokens_exact'] for r in rows), logits_total=len(checks),
            logits_exact=sum(r['exact'] for r in checks), logits_accepted=sum(r['accepted'] for r in checks),
            forced_argmax_exact=sum(r['mpk_argmax'] == r['hf_argmax'] for r in checks),
            max_abs=max(r['max_abs'] for r in checks), max_relative_rms=max(r['relative_rms'] for r in checks))
        report['completed'] = True
        report['passed'] = (all(r['tokens_exact'] and r['ci_prefix_exact'] for r in rows)
                            and all(r['accepted'] for r in checks))
        print(json.dumps(report['summary']), flush=True)
        if not report['passed']:
            raise AssertionError('Full-model acceptance failed; see report.json')
    finally:
        if mpk is not None:
            mpk.close()
        save()


if __name__ == '__main__':
    main()
