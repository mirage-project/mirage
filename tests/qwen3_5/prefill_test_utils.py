"""Reuse a compiled test graph only with its recorded source and backend."""
import hashlib
import json
from pathlib import Path
from mirage.mpk.models.qwen3_5.mpk_prefill import Qwen35ChunkPrefill


def load_compiled_decode(model, run, *, allow_other_b200=False):
    """Reuse decode-only kernels on the same GPU with unchanged production code."""
    import torch
    run = Path(run)
    report = json.loads((run/'report.json').read_text())
    assert report['completed'] and report['passed'], 'Compiled smoke run did not pass'
    assert report.get('batch_size', 1) == model.batch_size, 'Compiled batch size differs'
    assert report['attn_implementation'] == model.attn_implementation
    assert report['prefill_backend'] == 'pytorch_block'
    if report['gpu_uuid'] != str(torch.cuda.get_device_properties(0).uuid):
        assert allow_other_b200 and torch.cuda.get_device_capability() == (10, 0)
        assert 'B200' in torch.cuda.get_device_name(0)
        print('Diagnostic reuse on another B200; native HF must independently reproduce the frozen oracle', flush=True)
    assert report['compute_capability'] == list(torch.cuda.get_device_capability())
    repo = Path(__file__).resolve().parents[2]
    verified = 0
    for path, digest in report['source_sha256'].items():
        if path.startswith(('python/', 'include/', 'src/')):
            assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest, path
            verified += 1
    assert verified >= 20
    model.pk.load_mpk_kernel(str(run/'kernel'))
    model._compiled = True


def load_compiled_pair(model, run):
    run = Path(run)
    report = json.loads((run/'report.json').read_text())
    assert report['attn_implementation'] == model.attn_implementation
    assert report['prefill_backend'] == 'mpk_chunk64'
    repo = Path(__file__).resolve().parents[2]
    verified = 0
    for path, digest in report['source_sha256'].items():
        if path.startswith(('python/', 'include/', 'src/')):
            assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest, path
            verified += 1
    assert verified >= 20
    model.pk.load_mpk_kernel(str(run/'kernel'))
    model._prefill_graph = Qwen35ChunkPrefill(model)
    model._prefill_graph.pk.load_mpk_kernel(str(run/'kernel/prefill'))
    model._prefill_graph.compiled = True
    model._compiled = True
