"""Qwen3 built from the graph, not from a call order.

The MLP test covers graph -> partition -> search -> register on one block.
This is the whole model: embedding, decoder layers, final norm, lm_head and
argmax, lowered through builder_low_level_ir onto a real PersistentKernel, then
compiled and run.

Two layers rather than 28 -- the path is identical and the build is a
megakernel compile either way.
"""
import subprocess
import sys
import textwrap

import pytest
import torch


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


_MODEL_SRC = textwrap.dedent(r"""
import sys, torch, mirage as mi
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.qwen3.builder_low_level_ir import Qwen3Shapes
from mirage.mpk.models.qwen3 import builder_low_level_ir as B

T, NL = 8, 2
S = Qwen3Shapes(tokens=T, hidden=1024, intermediate=3072, num_layers=NL,
                num_q_heads=16, num_kv_heads=8, head_dim=128, vocab=151936,
                max_seq=128)
PAGE, MAXPAGES = 128, T
torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16

nw, ns = mi.get_configurations_from_gpu(0)
p = PersistentKernel.get_default_init_parameters()
step = torch.zeros(1, dtype=torch.int32, device=dev)
tokens = torch.zeros(T, 128, dtype=torch.long, device=dev)
input_tokens = torch.zeros(T, 1, dtype=torch.long, device=dev)
output_tokens = torch.zeros(T, 1, dtype=torch.long, device=dev)
num_new_tokens = torch.ones(T, dtype=torch.int32, device=dev)
prompt_lengths = torch.ones(T, dtype=torch.int32, device=dev)
qo = torch.zeros(T + 1, dtype=torch.int32, device=dev)
kvp = torch.zeros(T + 1, dtype=torch.int32, device=dev)
kvi = torch.zeros(MAXPAGES, dtype=torch.int32, device=dev)
kvl = torch.zeros(T, dtype=torch.int32, device=dev)
tokens[:, 0] = 13
input_tokens[:, 0] = 13
p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0,
         world_size=1, max_num_batched_tokens=T, max_num_batched_requests=T,
         max_seq_length=128, max_num_pages=MAXPAGES, page_size=PAGE,
         eos_token_id=-1,
         meta_tensors={"step": step, "tokens": tokens,
                       "input_tokens": input_tokens,
                       "output_tokens": output_tokens,
                       "num_new_tokens": num_new_tokens,
                       "prompt_lengths": prompt_lengths,
                       "qo_indptr_buffer": qo,
                       "paged_kv_indptr_buffer": kvp,
                       "paged_kv_indices_buffer": kvi,
                       "paged_kv_last_page_len_buffer": kvl})
pk = PersistentKernel(**p)

def rnd(*d): return (torch.randn(*d, dtype=bf, device=dev) * 0.05)

bind, meta = {}, {}
bind["input_tokens"] = pk.attach_input(input_tokens, name="input_token")
bind["embed"] = pk.attach_input(rnd(S.vocab, S.hidden), name="embed")
cos = pk.attach_input(rnd(S.max_seq, S.head_dim), name="cos")
sin = pk.attach_input(rnd(S.max_seq, S.head_dim), name="sin")
bind["cos"], bind["sin"] = cos, sin
for i in range(NL):
    bind[f"l{i}.in_norm"]   = pk.attach_input(rnd(1, S.hidden), name=f"n{i}a")
    bind[f"l{i}.qkv"]       = pk.attach_input(rnd(S.hidden, S.qkv_dim), name=f"qkv{i}")
    bind[f"l{i}.q_norm"]    = pk.attach_input(rnd(S.head_dim), name=f"qn{i}")
    bind[f"l{i}.k_norm"]    = pk.attach_input(rnd(S.head_dim), name=f"kn{i}")
    bind[f"l{i}.k_cache"]   = pk.attach_input(
        torch.zeros(MAXPAGES, PAGE, S.num_kv_heads, S.head_dim, dtype=bf, device=dev), name=f"kc{i}")
    bind[f"l{i}.v_cache"]   = pk.attach_input(
        torch.zeros(MAXPAGES, PAGE, S.num_kv_heads, S.head_dim, dtype=bf, device=dev), name=f"vc{i}")
    bind[f"l{i}.o"]         = pk.attach_input(rnd(S.attn_dim, S.hidden), name=f"o{i}")
    bind[f"l{i}.post_norm"] = pk.attach_input(rnd(1, S.hidden), name=f"n{i}b")
    bind[f"l{i}.gate"]      = pk.attach_input(rnd(S.hidden, S.intermediate), name=f"g{i}")
    bind[f"l{i}.up"]        = pk.attach_input(rnd(S.hidden, S.intermediate), name=f"u{i}")
    bind[f"l{i}.down"]      = pk.attach_input(rnd(S.intermediate, S.hidden), name=f"d{i}")
bind["final_norm"] = pk.attach_input(rnd(1, S.hidden), name="fn")
bind["lm_head"]    = pk.attach_input(rnd(S.hidden, S.vocab), name="lm")
meta["argmax_value"] = pk.new_tensor(dims=(T, pk.num_workers), dtype=mi.bfloat16,
                                     name="amv", io_category="cuda_tensor")
meta["argmax_index"] = pk.new_tensor(dims=(T, pk.num_workers), dtype=mi.int64,
                                     name="ami", io_category="cuda_tensor")
meta["output_token"] = pk.attach_input(output_tokens, name="output_token")

graph, groups, env = B.build(pk, S, bind, meta, num_layers=NL, verbose=True)
print(f"LOWERED {len(graph)} nodes -> {len(groups)} tasks", flush=True)
pk.compile(output_dir=None)
print("COMPILED", flush=True)
pk(); torch.cuda.synchronize()
gen = tokens[:, :step[0].item() + 1]
print(f"RAN step={step[0].item()} first_tokens={gen[0, :6].tolist()}", flush=True)
ok = bool(((gen >= 0) & (gen < S.vocab)).all().item())
print("VALID" if ok else "INVALID", flush=True)
sys.exit(0 if ok else 1)

""")


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_model_lowers_from_the_graph_and_runs():
    """Every opaque node reaches a hand-written task, every muGraph group
    reaches search (or its fallback), and the megakernel produces
    in-vocabulary tokens.

    Weights are random, so this is a wiring and liveness gate, not a numeric
    one -- the numeric gate is test_model_graph.py::test_lowered_mlp_matches_torch.
    """
    proc = subprocess.run([sys.executable, "-c", _MODEL_SRC],
                          capture_output=True, text=True, timeout=3600)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"graph-built model failed:\n{tail}"
    assert "LOWERED" in proc.stdout and "COMPILED" in proc.stdout, tail
    assert "VALID" in proc.stdout, tail

    # The opaque four must all have been dispatched to hand-written tasks, and
    # at least one group must have been scheduled by search rather than fallen
    # back -- otherwise this proves nothing about the searched path.
    lowered = [l for l in proc.stdout.splitlines() if l.startswith("[lower]")]
    for name in ("embedding", "rmsnorm", "attention", "argmax"):
        assert any(f"{name}: hand-written" in l for l in lowered), name
    assert any(": search " in l for l in lowered), lowered[:5]
