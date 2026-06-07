"""In-MPK correctness test for fp8_gemm_dense_splitk_tma_reduce_sm100 via
PersistentKernel test_mode.

This validates the ferret-produced FP8 split-K dense GEMM with a TMA reduce-add
epilogue (TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100). Unlike the crashed
red.global.add decode_splitk, each K-slice CTA stages its dequantized bf16
partial tile to SMEM and issues one cp.reduce.async.bulk.tensor.2d (TMA-engine
reduce-add) into a PRE-ZEROED bf16 output. The wrapper layer prepends a
tensor_init that zeros the output before the GEMM (the reduce-add accumulates
into the zero base), so the output is correct without any host pre-zero.

It goes through the FULL MPK compile pipeline (graph.cc dispatch, task_register
codegen, the 3 TMA descriptors — A/B FP8 loads + the C bf16 reduce-add output —
built in tma.cuh, megakernel nvcc, scheduler dispatch) on a SINGLE GPU, then
checks the reduced BF16 output vs an FP32 dequant-matmul reference (cos > 0.99).

DECODE-GATE DRIVING (load-bearing — see the qkva_splitk_v2 test docstring for
the full RCA). The kernel bakes the SAME decode-phase gate in its task_register
codegen:

    int q_len_      = qo_indptr_buffer[1] - qo_indptr_buffer[0];
    if (q_len_ > 8) return;                       // prefill -> skip
    int active_rows_ = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
    int runtime_m_   = min(active_rows_, M);
    if (runtime_m_ <= 0) return;                  // nothing to do -> skip

`qo_indptr_buffer` is rebuilt by prepare_next_batch from the request scheduler
state at run time, so we MUST drive the request state (tokens / step /
prompt_lengths / num_new_tokens) so prepare_next_batch produces M single-token
DECODE requests in one batch => qo_indptr = [0,1,..,M], q_len=1<=8 (gate passes)
and active_rows = M (all rows run). With the decode gate passing AND the
prepended tensor_init zeroing the output, a wrong/no-write output stays all-zero
=> cos~=0; cos>0.99 therefore proves the gate passed and the GEMM accumulated
correctly into every row.

Shapes (the two ferret-validated configs):
    qkv_a   : M=128, K=7168, N=2176, SK=8   (ferret cos PASS, maxerr 0.0069)
    gate_up : M=128, K=7168, N=1024, SK=4   (shared-expert gate_up, SK=4)
plus a tiny shape (M=128, N=256, K=512, SK=4) for a fast compile smoke first.

Run:
  CUDA_VISIBLE_DEVICES=<free_gpu> python \
    tests/runtime_python/blackwell/sm100_fp8_gemm_dense_splitk_tmareduce/\
test_fp8_gemm_dense_splitk_tmareduce_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402

FP8_MAX = 448.0


def quantize_a_f32scale(a_bf16):
    M, K = a_bf16.shape
    assert K % 128 == 0
    nk = K // 128
    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)
    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            am = block.abs().max().item()
            scale = am / FP8_MAX if am > 0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
    return a_fp8, sa


def quantize_b_f32scale(b_bf16):
    N, K = b_bf16.shape
    assert K % 128 == 0 and N % 128 == 0
    nb, nk = N // 128, K // 128
    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((nb, nk), dtype=torch.float32, device=b_bf16.device)
    b_f32 = b_bf16.float()
    for bi in range(nb):
        for ki in range(nk):
            block = b_f32[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128]
            am = block.abs().max().item()
            scale = am / FP8_MAX if am > 0 else 1.0
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    nk = K // 128
    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_fp8[m, ki * 128:(ki + 1) * 128].float() * sa[m, ki])
    nb = N // 128
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                b_fp8[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128].float() * sb[bi, ki])
    return torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)


def cosine_sim(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def run(M, N, K, split_k, num_workers, seed=42, use_gflag=False):
    label = (f"M={M}, N={N}, K={K}, split_k={split_k}, nw={num_workers}"
             f"{' [GFLAG]' if use_gflag else ''}")
    print(f"\n{'='*70}\nPK test_mode splitk_tma_reduce: {label}\n{'='*70}")
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)
    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)

    # Sentinel-fill output. The layer's prepended tensor_init zeros it before the
    # GEMM, so after a successful run the output is the pure GEMM result. If the
    # decode gate were mis-driven (q_len>8 / active_rows==0) the GEMM never runs,
    # leaving only the tensor_init-zeroed buffer -> all-zero -> cos~=0.
    SENTINEL = -1024.0  # bf16-exact power of two
    output = torch.full((M, N), SENTINEL, device=device, dtype=torch.bfloat16)

    num_workers_gpu, num_sched = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = max(num_workers, num_workers_gpu)
    params["num_local_schedulers"] = num_sched
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = M
    PAGE_SIZE = 128
    params["max_seq_length"] = max(PAGE_SIZE * 2, M)
    params["max_num_pages"] = max(M, 4)
    params["page_size"] = PAGE_SIZE

    compile_dir = os.path.join(
        THIS_DIR, f"pk_tmar_{M}_{N}_{K}_sk{split_k}")
    os.makedirs(compile_dir, exist_ok=True)

    # Drive request state -> prepare_next_batch emits M single-token DECODE
    # requests in one batch => qo_indptr=[0,1,..,M] (q_len=1<=8, active_rows=M).
    qo = torch.zeros(M + 1, dtype=torch.int32, device=device)
    tokens = torch.zeros(M, params["max_seq_length"], dtype=torch.int64,
                         device=device)
    step = torch.ones(M, dtype=torch.int32, device=device)
    prompt_lengths = torch.ones(M, dtype=torch.int32, device=device)
    num_new_tokens = torch.ones(M, dtype=torch.int32, device=device)
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo,
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "num_new_tokens": num_new_tokens,
    }
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    if use_gflag:
        # gflag path: zero-init int32 scratch of ceil(N/128); the kernel's
        # k0-store/k>0-spin fork self-initializes the output (NO tensor_init).
        # HANG-SAFE only at nn*split_k<=num_workers (asserted in the register
        # fn) -> pass num_workers large enough for this decode shape.
        nn = (N + 127) // 128
        gflag = torch.zeros(nn, dtype=torch.int32, device=device)
        gflag_dt = pk.attach_input(gflag, name="gflag")
        pk.fp8_gemm_dense_splitk_tma_reduce_gflag_layer(
            input_fp8=a_dt, weight_fp8=b_dt, input_scale=sa_dt,
            weight_scale=sb_dt, output=out_dt, gflag=gflag_dt,
            num_workers=num_workers, split_k=split_k)
    else:
        pk.fp8_gemm_dense_splitk_tma_reduce_layer(
            input_fp8=a_dt, weight_fp8=b_dt, input_scale=sa_dt,
            weight_scale=sb_dt, output=out_dt,
            num_workers=num_workers, split_k=split_k)

    print("  Compiling...")
    pk.compile(output_dir=compile_dir)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    print(f"  ref[0,:4]: {ref[0,:4].tolist()}")
    print(f"  out[0,:4]: {output[0,:4].tolist()}")
    # The prepended tensor_init zeros output; a no-write (gate mis-driven) run
    # therefore leaves rows all-zero, not SENTINEL. all-zero rows => no GEMM
    # accumulation on that row.
    zero_rows = (output.float().abs().sum(dim=1) == 0).sum().item()
    sentinel_rows = (output.float() == SENTINEL).all(dim=1).sum().item()
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cosine_sim(output, ref)
    passed = cos > 0.99 and zero_rows == 0 and sentinel_rows == 0
    print(f"  max_abs_diff={max_diff:.5f}  cos={cos:.6f}  "
          f"zero_rows={zero_rows}  sentinel_rows={sentinel_rows}  "
          f"-> {'PASS' if passed else 'FAIL'}")
    if sentinel_rows:
        print(f"  WARNING: {sentinel_rows} rows still SENTINEL -> the prepended "
              f"tensor_init did not zero them (unexpected).")
    if zero_rows and not sentinel_rows:
        print(f"  WARNING: {zero_rows} rows all-zero -> decode gate likely "
              f"mis-driven (GEMM did not accumulate) or split-K reduce missed "
              f"those rows.")
    pk.finalize()
    return passed, cos, max_diff


def main():
    results = {}
    # tiny shape first (fast compile, K%512==0) — correctness smoke.
    p, c, d = run(M=128, N=256, K=512, split_k=4, num_workers=64, seed=1)
    results["tiny M128 N256 K512 sk4"] = (p, c, d)

    # gate_up shape: M=128, N=1024, K=7168, SK=4 (shared-expert gate_up at TP=4).
    p, c, d = run(M=128, N=1024, K=7168, split_k=4, num_workers=128, seed=2)
    results["gate_up M128 N1024 K7168 sk4"] = (p, c, d)

    # qkv_a shape: M=128, N=2176, K=7168, SK=8 (ferret's headline config).
    p, c, d = run(M=128, N=2176, K=7168, split_k=8, num_workers=128, seed=3)
    results["qkv_a M128 N2176 K7168 sk8"] = (p, c, d)

    # GFLAG variant (Stage B: pre-zero elimination via k0-store/k>0-spin fork).
    # num_workers MUST be >= nn*split_k (hang-safety): qkv_a nn=17*sk8=136,
    # gate_up nn=8*sk4=32. Correct cos here proves the epoch handshake + reduce
    # are race-free AND the spin completed (no hang) at the decode regime.
    p, c, d = run(M=128, N=1024, K=7168, split_k=4, num_workers=136, seed=4,
                  use_gflag=True)
    results["gate_up GFLAG M128 N1024 K7168 sk4"] = (p, c, d)
    p, c, d = run(M=128, N=2176, K=7168, split_k=8, num_workers=136, seed=5,
                  use_gflag=True)
    results["qkv_a GFLAG M128 N2176 K7168 sk8"] = (p, c, d)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    ok = True
    for k, (p, c, d) in results.items():
        print(f"  {k}: {'PASS' if p else 'FAIL'} cos={c:.4f} maxdiff={d:.4f}")
        ok = ok and p
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
