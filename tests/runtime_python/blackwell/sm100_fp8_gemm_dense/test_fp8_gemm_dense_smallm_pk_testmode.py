"""DSV3 dense FP8 block-scaled GEMM (bf16 out) via PersistentKernel test_mode.

Covers BOTH `fp8_gemm_dense_smallm_layer` and `fp8_gemm_dense_mediumm_layer`.
The kernel is selected by `max_seq_length <= 512` (NOT by M):
    max_seq_length = 256  -> smallm
    max_seq_length = 4096 -> mediumm
so each (tp, bs) config is run at both max_seq_length values to exercise
both kernels.

Scale layout (per fp8_gemm_dense_sm100_common.cuh, plain float32):
    sa: float32 [M, K/128]    row-major  (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major (128x128 block weight scale)

DSV3 dense-FP8 GEMM shapes (N shards by world_size, K = contraction dim):
    qkv_a   : N=2176,         K=7168   (down-proj to lora ranks; N NOT sharded)
    gate_up : N=2*18432/tp,   K=7168   (dense MLP gate+up; N sharded by TP)
    kv_b    : N=4096,         K=512    (kv_b decompression; small-K=512)

Run:
    python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_pk_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from pytorch_reference import (  # noqa: E402
    quantize_a_f32scale,
    quantize_b_f32scale,
    reference_gemm,
    cosine_sim,
    rel_mean,
)

HIDDEN = 7168
INTERMEDIATE = 18432  # dense MLP intermediate (TP=1)


def _gate_up_n(tp: int) -> int:
    # gate||up: N = 2 * (intermediate // tp). Must stay a multiple of 128.
    n = 2 * (INTERMEDIATE // tp)
    assert n % 128 == 0, (tp, n)
    return n


def run_case(tp: int, bs: int, max_seq_length: int, N: int, K: int,
             label: str, seed: int = 42):
    kernel = "smallm" if max_seq_length <= 512 else "mediumm"
    tag = (f"tp={tp} bs={bs} msl={max_seq_length}({kernel}) "
           f"N={N} K={K} [{label}]")
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((bs, K), device=device, dtype=torch.bfloat16,
                         generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16,
                         generator=g)

    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)

    output = torch.zeros((bs, N), device=device, dtype=torch.bfloat16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    # TP is exercised as a per-rank SHAPE (N/K shard) on a single GPU; we keep
    # world_size=1 so no NVSHMEM/MPI is needed (decision-log convention). The
    # `tp` label only selects the shrunk N/K passed to the kernel.
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    params["max_seq_length"] = max_seq_length
    # Default meta tensors: total_num_requests=1, prompt_lengths filled with
    # max_num_batched_tokens=bs, so active_rows (= total tokens) = bs and the
    # kernel's runtime_m = min(M, active_rows) = bs. (Matches the proven
    # sm100_linear test pattern.)
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    gemm_layer = (pk.fp8_gemm_dense_smallm_layer if max_seq_length <= 512
                  else pk.fp8_gemm_dense_mediumm_layer)
    gemm_layer(
        input_fp8=a_dt, weight_fp8=b_dt,
        input_scale=sa_dt, weight_scale=sb_dt,
        output=out_dt, num_workers=num_workers,
    )

    compile_dir = os.path.join(
        THIS_DIR, f".pk_compile_dense_{label}_{tp}_{bs}_{max_seq_length}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    pk()
    torch.cuda.synchronize()

    zero_rows = (output.float().abs().sum(dim=1) == 0).nonzero(
        as_tuple=True)[0]
    cos = cosine_sim(output, ref)
    rel = rel_mean(output, ref)
    max_diff = (output.float() - ref.float()).abs().max().item()
    # fp8 GEMM: cosine > 0.99 OR rel <= 5% (decision-log tolerance).
    passed = (cos > 0.99 or rel <= 0.05) and zero_rows.numel() == 0
    print(f"  cos={cos:.5f} rel={rel*100:.3f}% max_abs_diff={max_diff:.4f} "
          f"zero_rows={zero_rows.numel()} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, cos, rel, tag


def main():
    results = []

    if os.environ.get("MPK_SMOKE") == "1":
        # One smallm + one mediumm config to catch setup errors fast.
        results.append(run_case(1, 8, 256, N=_gate_up_n(1), K=HIDDEN,
                                label="gate_up"))
        results.append(run_case(4, 16, 4096, N=_gate_up_n(4), K=HIDDEN,
                                label="gate_up"))
        return _summary(results)

    # Union-of-axes matrix on the TP-varying gate_up shape (N depends on TP):
    #   {tp=1} x {bs=1,2,4,8,16}  U  {bs=16} x {tp=2,4,8}  U  {tp=8, bs=1}
    # x {max_seq_length 256 (smallm), 4096 (mediumm)}.
    union = (
        [(1, b) for b in (1, 2, 4, 8, 16)]
        + [(t, 16) for t in (2, 4, 8)]
        + [(8, 1)]
    )
    for msl in (256, 4096):
        for tp, bs in union:
            N = _gate_up_n(tp)
            results.append(run_case(tp, bs, msl, N=N, K=HIDDEN,
                                    label="gate_up"))

    # qkv_a (small-N, N unsharded) and kv_b (small-K=512) at a couple corners
    # per kernel so both shapes are exercised at both smallm and mediumm.
    for msl in (256, 4096):
        results.append(run_case(1, 8, msl, N=2176, K=HIDDEN, label="qkv_a"))
        results.append(run_case(1, 16, msl, N=4096, K=512, label="kv_b"))

    # TP8-EP2 shape-campaign corners (2026-06-13, tp8_task_shape_list S4/S5):
    # the shared expert at TP8 is the smallest dense GEMM in the model
    # (gate_up N=2*256=512; down K=256 — tile-boundary territory), and the
    # dense-MLP down_proj at TP8 has K=18432/8=2304 (K-loop tail). bs={1,16}
    # on both kernel variants.
    for msl in (256, 4096):
        for bs in (1, 16):
            results.append(run_case(8, bs, msl, N=512, K=HIDDEN,
                                    label="shared_gateup_tp8"))
            results.append(run_case(8, bs, msl, N=HIDDEN, K=256,
                                    label="shared_down_tp8"))
            results.append(run_case(8, bs, msl, N=HIDDEN, K=2304,
                                    label="dense_down_tp8"))
    return _summary(results)


def _summary(results):
    print(f"\n{'='*78}\nSummary\n{'='*78}", flush=True)
    all_passed = True
    for passed, cos, rel, tag in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  cos={cos:.4f} rel={rel*100:.3f}%  {tag}",
              flush=True)
        all_passed = all_passed and passed

    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({sum(r[0] for r in results)}/{len(results)})", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
