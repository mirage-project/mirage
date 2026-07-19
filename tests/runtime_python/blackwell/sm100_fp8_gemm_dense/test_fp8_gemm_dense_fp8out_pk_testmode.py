"""DSV3 dense FP8 GEMM with fused fp8out epilogue via PersistentKernel test_mode.

Covers `fp8_gemm_dense_smallm_fp8out_layer` and
`fp8_gemm_dense_mediumm_fp8out_layer` (selected by max_seq_length <= 512).

Output is FP8 (bs, N) + a flat uint32 UE8M0 scale (bs, N/128). Each scale
entry's low 8 bits hold encode_ue8m0(per-128-N-group-max / 448); the kernel
re-quantizes the f32 GEMM accumulator in registers (one BN=128 group per
consumer thread). See fp8_gemm_dense_qout_sm100_common.cuh:350-404.

DSV3 use: q_b_nope fused GEMM+quantize on the BMM Q-up path:
    N = num_local_q_heads * 128 = (128/tp) * 128 ,  K = q_lora = 1536
    TP=1 -> N=16384 ; TP=8 -> N=2048.

Reference: dense GEMM (f32 accumulator, no bf16 round-trip) -> per-128-N-group
re-quantize to UE8M0. Compare dequantized values (rel ~5%) and the scale
(UE8M0 exponent must match exactly since both sides encode the same max).

Run:
    python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_fp8out_pk_testmode.py
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
    reference_gemm_f32,
    requantize_fp8out_ref,
    dequant_fp8out,
    cosine_sim,
    rel_mean,
)

Q_LORA = 1536  # q_b_nope contraction dim


def _q_b_nope_n(tp: int) -> int:
    # N = num_local_q_heads * 128 = (128 // tp) * 128.
    h_local = 128 // tp
    n = h_local * 128
    assert n % 128 == 0, (tp, n)
    return n


def run_case(tp: int, bs: int, max_seq_length: int, N: int, K: int,
             label: str, seed: int = 42):
    kernel = "smallm" if max_seq_length <= 512 else "mediumm"
    tag = (f"tp={tp} bs={bs} msl={max_seq_length}({kernel}_fp8out) "
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

    # Reference: f32 accumulator -> per-128-N-group UE8M0 re-quantize.
    c_f32 = reference_gemm_f32(a_fp8, sa, b_fp8, sb)
    ref_fp8, ref_scale = requantize_fp8out_ref(c_f32)
    ref_dq = dequant_fp8out(ref_fp8, ref_scale)

    out_fp8 = torch.zeros((bs, N), device=device, dtype=torch.float8_e4m3fn)
    out_scale = torch.zeros((bs, N // 128), device=device, dtype=torch.uint32)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    # TP exercised as per-rank SHAPE on a single GPU; world_size=1 (no NVSHMEM).
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    params["max_seq_length"] = max_seq_length
    # Default meta tensors: total_num_requests=1, prompt=bs -> active_rows=bs,
    # runtime_m = min(M, active_rows) = bs (runtime_m_mode=0, no phase gate).
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    out_scale_dt = pk.attach_input(out_scale, name="out_scale")

    gemm_layer = (pk.fp8_gemm_dense_smallm_fp8out_layer if max_seq_length <= 512
                  else pk.fp8_gemm_dense_mediumm_fp8out_layer)
    gemm_layer(
        input_fp8=a_dt, weight_fp8=b_dt,
        input_scale=sa_dt, weight_scale=sb_dt,
        output_fp8=out_fp8_dt, output_scale=out_scale_dt,
        num_workers=num_workers,
    )

    compile_dir = os.path.join(
        THIS_DIR, f".pk_compile_fp8out_{label}_{tp}_{bs}_{max_seq_length}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    pk()
    torch.cuda.synchronize()

    # Scale: UE8M0 exponent byte must match exactly (both sides encode the
    # same per-group max). The kernel writes scale_byte in the low 8 bits with
    # upper 24 bits zero, and the reference stores scale_byte (0-255) directly,
    # so the full uint32 words match. (torch has no bitwise_and for CUDA uint32,
    # so compare on the CPU int64 view of the low byte.)
    out_lo = (out_scale.cpu().to(torch.int64) & 0xFF)
    ref_lo = (ref_scale.cpu().to(torch.int64) & 0xFF)
    scale_match = torch.equal(out_lo, ref_lo)
    n_scale_diff = int((out_lo != ref_lo).sum().item())

    # Values: dequant both sides and compare on cosine / rel-mean.
    out_dq = dequant_fp8out(out_fp8, out_scale)
    cos = cosine_sim(out_dq, ref_dq)
    rel = rel_mean(out_dq, ref_dq)
    # Also compare against the unquantized f32 GEMM (end-to-end fp8out error).
    rel_vs_f32 = rel_mean(out_dq, c_f32)

    passed = (cos > 0.99 or rel <= 0.05) and scale_match
    print(f"  scale_match={scale_match} (n_diff={n_scale_diff}) "
          f"cos={cos:.5f} rel_vs_ref={rel*100:.3f}% "
          f"rel_vs_f32={rel_vs_f32*100:.3f}% -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, cos, rel, scale_match, tag


def main():
    results = []
    if os.environ.get("MPK_SMOKE") == "1":
        results.append(run_case(1, 8, 256, N=_q_b_nope_n(1), K=Q_LORA,
                                label="q_b_nope"))
        results.append(run_case(8, 16, 4096, N=_q_b_nope_n(8), K=Q_LORA,
                                label="q_b_nope"))
        return _summary(results)
    # Union-of-axes over the TP-varying q_b_nope shape (N = (128/tp)*128):
    #   {tp=1} x {bs=1,2,4,8,16}  U  {bs=16} x {tp=2,4,8}  U  {tp=8, bs=1}
    # x {max_seq_length 256 (smallm_fp8out), 4096 (mediumm_fp8out)}.
    union = (
        [(1, b) for b in (1, 2, 4, 8, 16)]
        + [(t, 16) for t in (2, 4, 8)]
        + [(8, 1)]
    )
    for msl in (256, 4096):
        for tp, bs in union:
            N = _q_b_nope_n(tp)
            results.append(run_case(tp, bs, msl, N=N, K=Q_LORA,
                                    label="q_b_nope"))
    return _summary(results)


def _summary(results):
    print(f"\n{'='*78}\nSummary\n{'='*78}", flush=True)
    all_passed = True
    for passed, cos, rel, scale_match, tag in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  cos={cos:.4f} rel={rel*100:.3f}% "
              f"scale_match={scale_match}  {tag}", flush=True)
        all_passed = all_passed and passed

    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({sum(r[0] for r in results)}/{len(results)})", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
