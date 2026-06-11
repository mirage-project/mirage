"""Test: linear_fp8_swapAB_layer at all DSV3 decode-path production shapes.

Covers all FP8 linear projections used in DeepSeek-V3 decode path (batch ≤ 16),
which all route through linear_fp8_swapAB_layer in the builder:

  QKV_A   : (bs, 2176, 7168)              — fused QKV-a projection (TP-independent)
  Q_B     : (bs, 128//TP × 192, 1536)     — Q up-projection (nope+rope heads)
  O_Proj  : (bs, 7168, 128//TP × 128)     — attention output projection
  MLP_L1  : (bs, 36864//TP, 7168)         — fused gate+up projection
  MLP_L2  : (bs, 7168, 18432//TP)         — down projection

Each shape is tested over the union-of-axes (TP, bs) matrix used in the builder:
  {TP=1} × {bs=1,2,4,8,16}  ∪  {bs=16} × {TP=2,4,8}  ∪  {TP=8, bs=1}

BMM shapes (Q-absorption kv_b_k, V-unabsorption kv_b_v) are separately covered in
  tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_linear_fp8_bmm_testmode.py

Prefill KV UpProj (bs > 16) uses fp8_gemm_dense_mediumm_layer, not swapAB.

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_swapAB/test_dsv3_fp8_swapAB_shapes_testmode.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)

FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_dsv3_swapAB")
os.makedirs(FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# DeepSeek-V3 model constants (builder.py)
# ---------------------------------------------------------------------------
NUM_Q_HEADS = 128
Q_LORA_RANK = 1536       # c_q latent dim  (Q_B K)
KV_LORA_RANK = 512       # c_kv latent dim (unused in this file — see BMM test)
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128         # pre-absorption
HIDDEN = 7168
INTERMEDIATE = 18432     # per-TP, gate+up = 2*INTERMEDIATE//TP
# QKV_A_FUSED_N = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM + 64-pad = 2176
QKV_A_FUSED_N = 2176


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_grid_x(full_n: int, num_workers: int) -> int:
    """Mirror builder._fp8_linear_grid_x: largest divisor of N/128 ≤ num_workers."""
    assert full_n % 128 == 0
    max_tiles = full_n // 128
    if max_tiles <= num_workers:
        return max_tiles
    best = 1
    i = 1
    while i * i <= max_tiles:
        if max_tiles % i == 0:
            if i <= num_workers:
                best = max(best, i)
            other = max_tiles // i
            if other <= num_workers:
                best = max(best, other)
        i += 1
    if best * 4 < num_workers:
        return max_tiles
    return best


def _make_pk(batch_size: int) -> tuple:
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    return PersistentKernel(**params), num_workers


def _quantize(x_bf16: torch.Tensor):
    q, s = quantize_to_fp8_packed_ue8m0(x_bf16)
    return q.contiguous(), s.contiguous()


def _run_case(label: str, bs: int, full_n: int, k: int, seed: int = 42) -> bool:
    """Compile + run linear_fp8_swapAB_layer for one (bs, N, K) config."""
    assert bs <= 16, f"swapAB decode kernel caps batch <= 16, got {bs}"
    assert full_n % 128 == 0, f"N={full_n} not divisible by 128"

    device = "cuda"
    torch.manual_seed(seed)
    x_bf16 = (torch.randn(bs, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    w_bf16 = (torch.randn(full_n, k, dtype=torch.bfloat16, device=device)
              / k ** 0.5).contiguous()
    x_fp8, x_sc = _quantize(x_bf16)
    w_fp8, w_sc = _quantize(w_bf16)
    output = torch.zeros(bs, full_n, dtype=torch.bfloat16, device=device)

    x_dq = dequant_from_packed_ue8m0(x_fp8, x_sc)
    w_dq = dequant_from_packed_ue8m0(w_fp8, w_sc)
    ref = (x_dq.float() @ w_dq.float().T).to(torch.bfloat16)

    pk, num_workers = _make_pk(bs)
    grid_x = _pick_grid_x(full_n, num_workers)
    per_n = full_n // grid_x

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  bs={bs}  N={full_n}  K={k}  grid_x={grid_x}  per-task-N={per_n}")

    i_fp8 = pk.attach_input(x_fp8, name="inp_fp8")
    i_sc = pk.attach_input(x_sc, name="inp_sc")
    w_fp8_ = pk.attach_input(w_fp8, name="wgt_fp8")
    w_sc_ = pk.attach_input(w_sc, name="wgt_sc")
    o = pk.attach_input(output, name="out")

    pk.linear_fp8_swapAB_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8_, weight_scale=w_sc_,
        output=o,
        grid_dim=(grid_x, 1, 1),
        block_dim=(256, 1, 1),
    )

    print("  Compiling...")
    pk.compile(output_dir=FOLDER)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    finite = torch.isfinite(output).all().item()
    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = ref.float().abs().mean().item() + 1e-12
    rel = mean_abs / denom
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(), ref.float().flatten(), dim=0).item()

    print(f"  output[0,:6]:    {output[0, :6].tolist()}")
    print(f"  ref   [0,:6]:    {ref[0, :6].tolist()}")
    print(f"  finite={finite}  max_abs={max_abs:.4f}  cos={cos:.6f}  rel={rel*100:.4f}%")

    pk.finalize()
    ok = finite and (cos > 0.99 or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


# ---------------------------------------------------------------------------
# Union-of-axes matrix: hits every TP in {1,2,4,8} and every bs in {1,2,4,8,16}
# ---------------------------------------------------------------------------
_UNION = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]   # tp=1 full bs sweep
    + [(tp, 16) for tp in (2, 4, 8)]         # max bs at each tp
    + [(8, 1)]                               # min bs at max tp
)


# ---------------------------------------------------------------------------
# Shape families
# ---------------------------------------------------------------------------

def _qkv_a_cases():
    """QKV-A projection: (bs, 2176, 7168). TP-independent (each rank computes full output)."""
    # Only sweep bs at tp=1; K and N don't depend on TP.
    return [(f"qkv_a  bs={bs}  N={QKV_A_FUSED_N}  K={HIDDEN}",
             bs, QKV_A_FUSED_N, HIDDEN)
            for bs in (1, 2, 4, 8, 16)]


def _q_b_cases():
    """Q up-projection: (bs, 128//TP × 192, 1536). N is TP-sharded."""
    cases = []
    for tp, bs in _UNION:
        hl = NUM_Q_HEADS // tp
        n = hl * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)   # hl × 192
        cases.append((f"q_b    tp={tp}  bs={bs}  N={n}  K={Q_LORA_RANK}",
                      bs, n, Q_LORA_RANK))
    return cases


def _o_proj_cases():
    """O projection: (bs, 7168, 128//TP × 128). K is TP-sharded."""
    cases = []
    for tp, bs in _UNION:
        hl = NUM_Q_HEADS // tp
        k = hl * V_HEAD_DIM                             # hl × 128
        cases.append((f"o_proj tp={tp}  bs={bs}  N={HIDDEN}  K={k}",
                      bs, HIDDEN, k))
    return cases


def _mlp_l1_cases():
    """MLP gate+up: (bs, 36864//TP, 7168). N is TP-sharded."""
    cases = []
    for tp, bs in _UNION:
        n = (2 * INTERMEDIATE) // tp
        cases.append((f"mlp_l1 tp={tp}  bs={bs}  N={n}  K={HIDDEN}",
                      bs, n, HIDDEN))
    return cases


def _mlp_l2_cases():
    """MLP down: (bs, 7168, 18432//TP). K is TP-sharded."""
    cases = []
    for tp, bs in _UNION:
        k = INTERMEDIATE // tp
        cases.append((f"mlp_l2 tp={tp}  bs={bs}  N={HIDDEN}  K={k}",
                      bs, HIDDEN, k))
    return cases


# ---------------------------------------------------------------------------
# Per-shape test entry points (also usable as pytest test functions)
# ---------------------------------------------------------------------------

def _run_family(name: str, cases: list) -> dict:
    results = {}
    for label, bs, n, k in cases:
        seed = 42 + bs * 7 + n % 997 + k % 113
        try:
            results[label] = _run_case(label, bs, n, k, seed=seed)
        except Exception:
            import traceback
            traceback.print_exc()
            results[label] = False
    n_pass = sum(1 for v in results.values() if v)
    print(f"\n[{name}] {n_pass}/{len(results)} PASS")
    return results


def test_qkv_a():
    results = _run_family("QKV_A", _qkv_a_cases())
    assert all(results.values()), f"QKV_A failures: {[k for k,v in results.items() if not v]}"


def test_q_b():
    results = _run_family("Q_B", _q_b_cases())
    assert all(results.values()), f"Q_B failures: {[k for k,v in results.items() if not v]}"


def test_o_proj():
    results = _run_family("O_Proj", _o_proj_cases())
    assert all(results.values()), f"O_Proj failures: {[k for k,v in results.items() if not v]}"


def test_mlp_l1():
    results = _run_family("MLP_L1", _mlp_l1_cases())
    assert all(results.values()), f"MLP_L1 failures: {[k for k,v in results.items() if not v]}"


def test_mlp_l2():
    results = _run_family("MLP_L2", _mlp_l2_cases())
    assert all(results.values()), f"MLP_L2 failures: {[k for k,v in results.items() if not v]}"


if __name__ == "__main__":
    import sys

    all_results = {}
    for name, cases in [
        ("QKV_A",  _qkv_a_cases()),
        ("Q_B",    _q_b_cases()),
        ("O_Proj", _o_proj_cases()),
        ("MLP_L1", _mlp_l1_cases()),
        ("MLP_L2", _mlp_l2_cases()),
    ]:
        all_results.update(_run_family(name, cases))

    print(f"\n{'='*72}")
    print("Final summary (DSV3 FP8 swapAB shapes):")
    for lbl, ok in all_results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}")
    n_pass = sum(1 for v in all_results.values() if v)
    n_total = len(all_results)
    print(f"\n{n_pass}/{n_total} PASS")
    sys.exit(0 if n_pass == n_total else 1)
