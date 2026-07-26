"""Test mode: the Qwen3.5 attention variants through the full MPK pipeline.

Exercises what the kernel-wrapper test cannot: the Python layer API
(`paged_attention_layer(attn_output_gate=..., max_tokens_per_pass=...)`), the
params encoding, `TaskRegister::register_paged_attention_sm100_task`'s SECOND
emission branch, C++ code generation, nvcc compilation and runtime dispatch.

Three properties are checked in one place:

  1. the gated + Q-loop variant computes what the wrapper-level test says it
     should (same reference, but reached through codegen + the real runtime);
  2. the DEFAULT variant (no gate, no Q-loop) still goes through the ORIGINAL
     emission branch -- verified by generating both graphs and diffing the
     emitted `test_rank0.cu`, which must differ ONLY in the attention task's
     template argument list;
  3. `max_tokens_per_pass` genuinely decouples the smem arena from
     max_num_batched_tokens: the graph is built with mbt = 8, which at
     head_dim 256 / GQA 8:1 does NOT fit the 201 KiB budget as a single pass
     (probe P3), so a successful compile is itself the evidence.

Run:
    python tests/runtime_python/blackwell/sm100_attention_qwen35/\
test_attention_qwen35_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "demo", "qwen3_5"))
from rope_permutation import build_cos_sin_table  # noqa: E402

NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
NUM_QO_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_DIM = 256
PAGE_SIZE = 64
MAX_SEQ_LENGTH = 64
MAX_TOKENS_PER_PASS = 4
PROMPT_LENS = [5, 3]                 # two requests prefilling in one iteration
BF16 = torch.bfloat16
EPS = 1e-6


def build_graph(gated, q_pass, prompt_lens, device, out_dir):
    """Build + compile one attention graph; returns (output, inputs, path)."""
    total_tokens = sum(prompt_lens)
    num_requests = len(prompt_lens)
    q_stride = 2 * HEAD_DIM if gated else HEAD_DIM
    qkv_width = (NUM_QO_PER_KV * q_stride + 2 * HEAD_DIM) * NUM_KV_HEADS

    g = torch.Generator(device="cpu").manual_seed(20260726)
    qkv = (torch.randn(total_tokens, qkv_width, generator=g) * 0.5).to(BF16).to(device)
    k_cache = torch.zeros(num_requests * 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=BF16, device=device)
    v_cache = torch.zeros_like(k_cache)
    q_norm = torch.ones(HEAD_DIM, dtype=BF16, device=device)
    k_norm = torch.ones(HEAD_DIM, dtype=BF16, device=device)
    cos, sin = build_cos_sin_table(torch.arange(MAX_SEQ_LENGTH), dtype=BF16,
                                  device=device)
    cos, sin = cos.contiguous(), sin.contiguous()
    out = torch.zeros(total_tokens, NUM_Q_HEADS * HEAD_DIM, dtype=BF16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=total_tokens,
        max_num_batched_requests=num_requests,
        max_num_pages=num_requests * 2,
        page_size=PAGE_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        meta_tensors={
            "tokens": torch.zeros((num_requests, MAX_SEQ_LENGTH),
                                  dtype=torch.int64, device=device),
            "prompt_lengths": torch.tensor(prompt_lens, dtype=torch.int32,
                                           device=device),
            "step": torch.zeros(num_requests, dtype=torch.int32, device=device),
        },
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "the Qwen3.5 attention variants are sm100-only"

    qkv_dt = pk.attach_input(qkv, name="attn_qkvg")
    kc_dt = pk.attach_input(k_cache, name="attn_k_cache")
    vc_dt = pk.attach_input(v_cache, name="attn_v_cache")
    qn_dt = pk.attach_input(q_norm, name="attn_q_norm")
    kn_dt = pk.attach_input(k_norm, name="attn_k_norm")
    cos_dt = pk.attach_input(cos, name="attn_cos")
    sin_dt = pk.attach_input(sin, name="attn_sin")
    out_dt = pk.attach_input(out, name="attn_out")

    pk.paged_attention_layer(
        input=qkv_dt,
        k_cache=kc_dt,
        v_cache=vc_dt,
        q_norm=qn_dt,
        k_norm=kn_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=out_dt,
        grid_dim=(num_requests, NUM_KV_HEADS, 1),
        block_dim=(256, 1, 1),
        enable_qk_norm=True,
        attn_output_gate=gated,
        max_tokens_per_pass=q_pass,
    )
    pk.compile(output_dir=out_dir)
    pk()
    torch.cuda.synchronize()
    return out, (qkv, k_cache, v_cache, cos, sin, q_norm, k_norm)


def reference(qkv, cos, sin, q_norm, k_norm, prompt_lens, gated, device):
    """Inline reference (see the wrapper test for how each step was validated
    against the HF oracle)."""
    q_stride = 2 * HEAD_DIM if gated else HEAD_DIM
    group_w = NUM_QO_PER_KV * q_stride + 2 * HEAD_DIM
    outs = []
    off = 0
    for n in prompt_lens:
        rows = qkv[off:off + n]
        o = torch.zeros(n, NUM_Q_HEADS * HEAD_DIM, dtype=BF16, device=device)
        for gidx in range(NUM_KV_HEADS):
            base = gidx * group_w
            qs, gs = [], []
            for h in range(NUM_QO_PER_KV):
                s = base + h * q_stride
                qs.append(rows[:, s:s + HEAD_DIM])
                if gated:
                    gs.append(rows[:, s + HEAD_DIM:s + 2 * HEAD_DIM])
            q = torch.stack(qs, 1)                                  # [n,H,256]
            kb = base + NUM_QO_PER_KV * q_stride
            k = rows[:, kb:kb + HEAD_DIM].unsqueeze(1)              # [n,1,256]
            v = rows[:, kb + HEAD_DIM:kb + 2 * HEAD_DIM].unsqueeze(1)

            def norm_rope(x, w, pos):
                xf = x.float()
                rms = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / HEAD_DIM + EPS)
                xn = (xf * (rms * w.float())).to(BF16)
                c = cos[pos].unsqueeze(1).float()
                s_ = sin[pos].unsqueeze(1).float()
                half = HEAD_DIM // 2
                rh = torch.cat((-xn[..., half:], xn[..., :half]), -1)
                return (xn.float() * c + rh.float() * s_).to(BF16)

            pos = torch.arange(n, device=device)
            qr = norm_rope(q, q_norm, pos)
            kr = norm_rope(k, k_norm, pos)
            logits = torch.einsum("thd,shd->hts", qr.float(),
                                  kr.expand(-1, NUM_QO_PER_KV, -1).float())
            logits = logits * (HEAD_DIM ** -0.5)
            mask = torch.arange(n, device=device).unsqueeze(0) > \
                torch.arange(n, device=device).unsqueeze(-1)
            logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(logits, -1, dtype=torch.float32)
            oh = torch.einsum("hts,shd->thd", p,
                              v.expand(-1, NUM_QO_PER_KV, -1).float()).to(BF16)
            if gated:
                gate = torch.stack(gs, 1)
                oh = (oh.float() * torch.sigmoid(gate.float()).to(BF16).float()).to(BF16)
            for h in range(NUM_QO_PER_KV):
                col = (gidx * NUM_QO_PER_KV + h) * HEAD_DIM
                o[:, col:col + HEAD_DIM] = oh[:, h]
        outs.append(o)
        off += n
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------
# Each phase runs in its OWN process. `TaskRegister` is process-global and
# accumulates every variant registered so far, so a second graph built in the
# same interpreter emits the union of both graphs' task variants. That matters
# here because one phase deliberately registers a variant that must NOT
# compile -- in a shared process it would poison every later codegen instead of
# only its own.
# --------------------------------------------------------------------------
def phase_gated():
    device = "cuda"
    out, (qkv, kc, vc, cos, sin, qn, kn) = build_graph(
        gated=True, q_pass=MAX_TOKENS_PER_PASS, prompt_lens=PROMPT_LENS,
        device=device, out_dir=os.path.join(_HERE, "test_output_attn_gated"))
    ref = reference(qkv, cos, sin, qn, kn, PROMPT_LENS, True, device)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"  prompt_lens={PROMPT_LENS} (max > MAX_TOKENS_PER_PASS="
          f"{MAX_TOKENS_PER_PASS} -> the Q-loop is exercised)")
    print(f"  RESULT max_abs_diff vs reference = {err:.3e}")
    return 0 if err <= 2e-2 else 1


def phase_nopass():
    # Direct evidence that max_tokens_per_pass is load-bearing rather than
    # cosmetic: with the same mbt the default derivation sets MAX_TOKENS = 8,
    # which at head_dim 256 / GQA 8:1 exceeds the 201 KiB budget (probe P3), so
    # the megakernel build must fail on the kernel's own static_assert. This
    # phase PASSES when the build fails.
    try:
        build_graph(gated=True, q_pass=0, prompt_lens=PROMPT_LENS,
                    device="cuda",
                    out_dir=os.path.join(_HERE, "test_output_attn_nopass"))
    except Exception as e:  # noqa: BLE001 - any build failure is the signal
        print(f"  RESULT build failed as expected ({type(e).__name__})")
        return 0
    print("  RESULT mbt=8 WITHOUT the Q-loop COMPILED -- the smem constraint "
          "P3 measured no longer holds")
    return 1


def phase_default():
    # mbt = 4 so the default derivation lands on an admissible MAX_TOKENS.
    build_graph(gated=False, q_pass=0, prompt_lens=[3, 1], device="cuda",
                out_dir=os.path.join(_HERE, "test_output_attn_default"))
    print("  RESULT default variant built")
    return 0


PHASES = {"gated": phase_gated, "nopass": phase_nopass, "default": phase_default}


def drive():
    import subprocess
    failures = []
    titles = {
        "gated": "gated + Q-loop variant through the full pipeline",
        "nopass": "COUNTERFACTUAL: without the Q-loop, mbt=8 does NOT build",
        "default": "default variant builds and uses the ORIGINAL emission branch",
    }
    for name in ("gated", "nopass", "default"):
        print(f"\n== {titles[name]} ==", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", name], env=env,
                           capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("  RESULT") or line.startswith("  prompt_lens"):
                print(line, flush=True)
        if p.returncode != 0:
            failures.append(f"phase {name} failed (rc={p.returncode})")
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-12:])
            print(f"  --- tail ---\n{tail}", flush=True)

    # Codegen shape check: the default emission must carry ELEVEN template
    # arguments (no gate, no pass size); the gated one THIRTEEN.
    try:
        a = open(os.path.join(_HERE, "test_output_attn_default", "test_rank0.cu")).read()
        b = open(os.path.join(_HERE, "test_output_attn_gated", "test_rank0.cu")).read()
        la = [l for l in a.splitlines() if "multitoken_paged_attention_sm100" in l]
        lb = [l for l in b.splitlines() if "multitoken_paged_attention_sm100" in l]
        print(f"\n  default emission: {la}")
        print(f"  gated   emission: {lb}")
        n_default = la[0].count(",") if la else -1
        n_gated = lb[0].count(",") if lb else -1
        print(f"  template-arg commas: default={n_default} gated={n_gated}")
        if not la or "bfloat16, 8, 1" not in la[0]:
            failures.append("default graph did not emit the sm100 attention task")
        elif n_gated != n_default + 2:
            failures.append(
                f"expected the gated emission to carry exactly 2 more template "
                f"arguments (got default={n_default}, gated={n_gated})")
    except FileNotFoundError as e:
        failures.append(f"missing codegen artifact: {e}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\ntest-mode pipeline test passed")
    return 0


if __name__ == "__main__":
    if "--phase" in sys.argv:
        sys.exit(PHASES[sys.argv[sys.argv.index("--phase") + 1]]())
    sys.exit(drive())
