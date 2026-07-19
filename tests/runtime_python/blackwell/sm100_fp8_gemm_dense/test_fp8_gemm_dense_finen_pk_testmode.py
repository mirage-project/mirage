"""DSV3 fine-N dense FP8 block-scaled GEMM (bf16 out) via PersistentKernel
test_mode -- with FAITHFUL per-task in-MPK latency.

Validates the new `pk.fp8_gemm_dense_finen_layer` ->
TASK_FP8_GEMM_DENSE_FINEN_SM100 (308). The finen kernel is the mediumm device
body re-tiled to BN=16 (NS default 6, NE=4 baked in the finen fn). Its host-side
delta vs the dense GEMM is solely the B (weight) TMA descriptor box height (=16
instead of 128); A keeps box=128. It handles all M (correct at M=1 decode AND
M>1 prefill), so this exercises both.

================================ FIDELITY ================================
This test reports, for each kernel candidate, BOTH:
  (a) cos  -- real-path correctness vs the PyTorch FP8 reference (as before).
  (b) the in-MPK per-task latency -- the slowest-CTA body (slowCTA_us) and the
      call's true wall (wall_us), extracted from the persistent profiler buffer
      of the REAL single-task megakernel run at the PRODUCTION grid/worker
      config. (Not the misleading whole-megakernel e2e, and not a
      standalone-on-empty-GPU bench.) See _build_helper.profiled_per_task_latency.

  This is the thing the standalone bench MISSES: the per-task in-MPK SETUP +
  real grid/worker dispatch overhead at the production worker count. It is
  SINGLE-task (no co-resident contention from the ~20 other decode tasks) --
  that is the v2 refinement; do not read it as the full in-decode number.

PRODUCTION-FAITHFUL DEFAULT CONFIG (the #1 fidelity fix, 2026-06-16):
  * num_workers = get_configurations_from_gpu(0) = 136 on a B200 (148 SMs).
    This is the EXACT decode-build persistent worker count. (CLAUDE.md: "decode
    TP8 = 136 persistent workers".)
  * The dense FP8 GEMM's grid is grid.x = num_workers BY CONSTRUCTION
    (_fp8_gemm_dense_layer_impl: TBGraph(CyTBGraph((num_workers,1,1),...)));
    idle CTAs early-return. In the DSv3 *decode* build (_use_prefill=False),
    _fp8_dense_num_workers() returns the FULL pool = self.num_workers = 136, so
    the qkv_a dense GEMM runs at grid.x = 136 in production -- which is exactly
    what this test passes by default (num_workers=136).
  * Profiling is ON by default here (profiler_tensor set before compile()).

OPT-IN OVERRIDES (default unset -> production-faithful, byte-identical grid):
  * MPK_TEST_NUM_WORKERS=N  -- emulate "only N workers free" (sweeps; scales
    schedulers to hold the ratio). DO NOT set for a verdict-grade number.
  * MPK_TEST_TIMING_ITERS=K  -- number of profiled launches to median over
    (default 30 in this test; >0 enables the faithful timing path).
  * MPK_DENSE_FINEN_VALIDATE=1 -- run the mediumm-vs-finen head-to-head on the
    qkv_a decode shape (M=1) and print the fidelity table + verdict.

Scale layout (per fp8_gemm_dense_sm100_common.cuh, plain float32):
    sa: float32 [M, K/128]    row-major  (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major (128x128 block weight scale)

Shapes respect the builder.py finen gate (weight.dim(0)=N <= 2304, N%16==0,
K%512==0):
    qkv_a : N=2176, K=7168   (down-proj to lora ranks; the ferret v003 target)
    kv_b  : N=2048, K=512    (small-K)
    small : N=2304, K=1024   (N at the gate's upper bound; BN=16 tile sweep)

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> \
      python tests/runtime_python/blackwell/sm100_fp8_gemm_dense/\
test_fp8_gemm_dense_finen_pk_testmode.py
The validation head-to-head (mediumm vs finen, qkv_a M=1):
    CUDA_VISIBLE_DEVICES=<free gpu> MPK_DENSE_FINEN_VALIDATE=1 \
      python .../test_fp8_gemm_dense_finen_pk_testmode.py
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
from _build_helper import (  # noqa: E402
    resolve_num_workers,
    timing_iters,
    make_profiler_tensor,
    profiled_per_task_latency,
    resolve_shape,
)

# Task-type names emitted by the profiler for each kernel (profiler_persistent
# .event_name_list): finen -> 308, the dense (mediumm/smallm) GEMM -> 306.
_TASK_NAME = {
    "finen": "TASK_FP8_GEMM_DENSE_FINEN_SM100",
    "mediumm": "TASK_FP8_GEMM_DENSE_SM100",
    "smallm": "TASK_FP8_GEMM_DENSE_SM100",
}

# Default number of profiled launches to median the faithful per-task latency
# over (overridable via MPK_TEST_TIMING_ITERS; 0 -> correctness-only).
_DEFAULT_TIMING_ITERS = 30

# Output-columns-per-CTA (BN) per kernel: finen re-tiles to BN=16; the dense
# mediumm/smallm body is BN=128. Used only to print the ANALYTIC tile-owner CTA
# count (ceil(N/BN)) alongside the profiled-CTA count, so the reader sees how
# many of the launched 136 workers actually owned an output tile.
_KERNEL_BN = {"finen": 16, "mediumm": 128, "smallm": 128}


def _resolved_grid() -> int:
    """The grid the run_case below will launch at: the production worker count
    (get_configurations_from_gpu(0)[0], 136 on B200) unless MPK_TEST_NUM_WORKERS
    overrides it. Used only for the table headers so they cite the SAME grid the
    faithful number is measured at."""
    gpu_workers, gpu_schedulers = mirage.get_configurations_from_gpu(0)
    nw, _ = resolve_num_workers(gpu_workers, gpu_schedulers)
    return nw


def _emit_gemm(pk, kernel, a_dt, b_dt, sa_dt, sb_dt, out_dt, num_workers):
    """Register the chosen dense-GEMM kernel onto the task graph."""
    if kernel == "finen":
        pk.fp8_gemm_dense_finen_layer(
            input_fp8=a_dt, weight_fp8=b_dt,
            input_scale=sa_dt, weight_scale=sb_dt,
            output=out_dt, num_workers=num_workers)
    elif kernel == "mediumm":
        pk.fp8_gemm_dense_mediumm_layer(
            input_fp8=a_dt, weight_fp8=b_dt,
            input_scale=sa_dt, weight_scale=sb_dt,
            output=out_dt, num_workers=num_workers)
    elif kernel == "smallm":
        pk.fp8_gemm_dense_smallm_layer(
            input_fp8=a_dt, weight_fp8=b_dt,
            input_scale=sa_dt, weight_scale=sb_dt,
            output=out_dt, num_workers=num_workers)
    else:
        raise ValueError(kernel)


def run_case(bs: int, N: int, K: int, label: str, kernel: str = "finen",
             seed: int = 42, profile: bool = True):
    tag = f"bs={bs} N={N} K={K} [{label}] kernel={kernel}"
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

    # PRODUCTION-FAITHFUL: num_workers = the GPU's real persistent worker count
    # (136 on a B200). MPK_TEST_NUM_WORKERS overrides only for opt-in sweeps.
    gpu_workers, gpu_schedulers = mirage.get_configurations_from_gpu(0)
    num_workers, num_schedulers = resolve_num_workers(gpu_workers,
                                                      gpu_schedulers)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    params["max_seq_length"] = 4096
    iters = timing_iters() if timing_iters() else (
        _DEFAULT_TIMING_ITERS if profile else 0)
    # Enable the persistent profiler so we can read THIS task's per-task span.
    # MUST be set before compile() (it keys -DMIRAGE_ENABLE_PROFILER off it).
    if iters > 0:
        params["profiler_tensor"] = make_profiler_tensor()
        params["trace_name"] = f"finen_test_{label}_{kernel}_{bs}"
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    _emit_gemm(pk, kernel, a_dt, b_dt, sa_dt, sb_dt, out_dt, num_workers)

    compile_dir = os.path.join(
        THIS_DIR, f".pk_compile_finen_{label}_{kernel}_{bs}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    # Correctness run (always). The output tensor holds the last launch's result.
    pk()
    torch.cuda.synchronize()

    # FAITHFUL per-task in-MPK latency (median over `iters` profiled launches).
    lat = None
    if iters > 0 and pk.profiler_tensor is not None:
        bn = _KERNEL_BN.get(kernel, 128)
        tile_ctas = (N + bn - 1) // bn  # analytic output-tile owners
        lat = profiled_per_task_latency(
            pk, _TASK_NAME[kernel], iters,
            label=f"{label} kernel={kernel} bs={bs} grid={num_workers}",
            grid=num_workers, tile_ctas=tile_ctas)

    # Tool diagnostic (added 2026-06-16): classify the output BEFORE cos so a
    # kernel that emits nan/inf or all-zeros is reported as a KERNEL fault with
    # the likely cause, NOT a bare "cos=nan" that looks like a harness bug.
    # (A KDA GEMV burned several rounds on an FP8-half2 overflow -> nan that the
    # bare cos=nan hid; this names the fault in one line. Carry this same
    # classify-before-cos into the group-GEMM harness, with PER-GROUP locality.)
    of = output.float()
    n_nan = int(torch.isnan(of).sum())
    n_inf = int(torch.isinf(of).sum())
    zero_rows = (of.abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if n_nan or n_inf:
        print(f"  OUTPUT NON-FINITE: nan={n_nan} inf={n_inf} -> FAIL "
              f"(KERNEL arithmetic emitted nan/inf, NOT a harness bug -- e.g. "
              f"FP8 e4m3 max=448, 448^2=200704 > fp16 max 65504, so accumulate "
              f"the dot product in fp32, not half2)", flush=True)
        pk.finalize()
        return False, float("nan"), float("nan"), tag, lat
    if zero_rows.numel() == of.shape[0]:
        print(f"  ALL-ZERO OUTPUT ({zero_rows.numel()}/{of.shape[0]} rows) -> "
              f"FAIL (KERNEL wrote nothing -- check the store path / write-gate, "
              f"NOT a harness bug)", flush=True)
        pk.finalize()
        return False, 0.0, float("inf"), tag, lat
    cos = cosine_sim(output, ref)
    rel = rel_mean(output, ref)
    max_diff = (of - ref.float()).abs().max().item()
    # Stricter than the decision-log 0.99 floor: this is a bit-for-bit re-tile
    # of the mediumm body, so require cos >= 0.999 (per the integration spec).
    # (mediumm itself: same body; smallm: 0.99 floor.)
    floor = 0.99 if kernel == "smallm" else 0.999
    passed = cos >= floor and zero_rows.numel() == 0
    print(f"  cos={cos:.5f} rel={rel*100:.3f}% max_abs_diff={max_diff:.4f} "
          f"zero_rows={zero_rows.numel()} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, cos, rel, tag, lat


def validate_finen_vs_mediumm():
    """The decisive fidelity test: measure the in-tree mediumm baseline (BN=128)
    AND the fine-N candidate (BN=16) for a dense decode shape (default qkv_a
    K=7168 N=2176; any registry shape via --shape/MPK_TEST_SHAPE), M=1, using the
    FAITHFUL per-task in-MPK measurement at the PINNED production grid (136).

    The known e2e fact: fine-N is ~1.7x faster STANDALONE yet NULL in the MPK
    e2e. If the faithful per-task measure reproduces that (fine-N ~= mediumm,
    the standalone 1.7x gone), the test-mode is faithful. If fine-N still shows
    a big per-task win here, the e2e NULL is elsewhere (and we report that --
    it's diagnostic).
    """
    sh = resolve_shape()
    N, K = sh.N, sh.K
    nw = _resolved_grid()
    print(f"\n{'#'*78}\n# FIDELITY VALIDATION: mediumm (BN=128) vs fine-N "
          f"(BN=16), shape={sh.name} (N={N} K={K} M=1), grid={nw}\n{'#'*78}",
          flush=True)
    res = {}
    for kernel in ("mediumm", "finen"):
        passed, cos, rel, tag, lat = run_case(
            bs=1, N=N, K=K, label=f"{sh.name}_M1_validate", kernel=kernel,
            profile=True)
        res[kernel] = {"passed": passed, "cos": cos, "lat": lat}

    print(f"\n{'='*78}\nFIDELITY TABLE -- faithful per-task IN-MPK latency "
          f"(shape={sh.name} N={N} K={K} M=1, grid={nw})\n{'='*78}", flush=True)
    hdr = (f"  {'kernel':<10}{'slowCTA_us':>12}{'wall_us':>10}"
           f"{'e2e_us':>10}{'grid':>6}{'prof_ctas':>10}{'tile_ctas':>10}"
           f"{'cos':>9}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for kernel in ("mediumm", "finen"):
        lat = res[kernel]["lat"]
        cos = res[kernel]["cos"]
        if lat is None:
            print(f"  {kernel:<10}{'(no profile data)':>48}", flush=True)
        else:
            print(f"  {kernel:<10}{lat['slowCTA_us']:>12.2f}"
                  f"{lat['wall_us']:>10.2f}{lat['e2e_us']:>10.2f}"
                  f"{(lat.get('grid') or 0):>6}"
                  f"{lat['profiled_ctas']:>10}"
                  f"{(lat.get('tile_ctas') or 0):>10}{cos:>9.4f}", flush=True)

    # Verdict.
    lm = res["mediumm"]["lat"]
    lf = res["finen"]["lat"]
    if lm is None or lf is None:
        print("\n  VERDICT: could not measure both kernels -- profiler data "
              "missing; cannot judge fidelity.", flush=True)
        return
    # Compare on slowCTA (the per-instance compute body -- the apples-to-apples
    # quantity vs a standalone single-tile bench) and on wall.
    def _ratio(a, b):
        return (a / b) if b > 0 else float("inf")
    slow_ratio = _ratio(lm["slowCTA_us"], lf["slowCTA_us"])
    wall_ratio = _ratio(lm["wall_us"], lf["wall_us"])
    print(f"\n  mediumm/finen slowCTA ratio = {slow_ratio:.2f}x   "
          f"wall ratio = {wall_ratio:.2f}x", flush=True)
    print("  (ratio ~1.0x => the standalone 1.7x fine-N win DISAPPEARS in the "
          "in-MPK per-task measure => reproduces the e2e NULL.)", flush=True)
    # Scoped verdict (reviewer + Codex, 2026-06-16): reproducing ONE e2e-NULL
    # case is a passing POSITIVE CONTROL -- necessary for fidelity, NOT a proof
    # of general faithfulness (a negative control + more shapes + the v2
    # co-residency harness are needed for that). State the narrow claim.
    if slow_ratio < 1.25 and wall_ratio < 1.25:
        print("\n  ==> POSITIVE CONTROL PASSED: at qkv_a M=1, fine-N ~= mediumm "
              "in the REAL in-MPK per-task measure (no ~1.7x win) by BOTH "
              "slowCTA and wall. The standalone advantage does NOT transfer "
              "in-MPK, reproducing the known e2e NULL.", flush=True)
        print("      (Equal *wall* not just slowCTA rules out 'dispatch stagger "
              "hides a faster body'. Mechanism: at M=1 the slowest-CTA body is "
              "the K=7168 reduction (BN-independent loop count); fine-N's win "
              "was OCCUPANCY (136 active CTAs vs ~17 tile-owners), which does "
              "not shorten the critical path.)", flush=True)
        print("      SCOPE: this is a POSITIVE control for ONE shape/M, NOT a "
              "proof the harness is faithful in general; it is SINGLE-task (no "
              "co-resident contention -- the v2 refinement). Reproducing the "
              "NULL WITHOUT co-residency shows co-residency is not REQUIRED to "
              "explain this NULL.", flush=True)
    else:
        print("\n  ==> fine-N STILL shows a per-task win in the in-MPK measure "
              f"(slowCTA {slow_ratio:.2f}x). If the e2e is NULL, the NULL is "
              "NOT in this isolated single-task GEMM (look downstream / "
              "co-residency / a consumer dependency).", flush=True)


def main():
    if os.environ.get("MPK_DENSE_FINEN_VALIDATE") == "1":
        validate_finen_vs_mediumm()
        return 0

    results = []
    # If MPK_TEST_SHAPE is set, run ONLY that registry shape (the parametrized
    # path the ferret/KDA bridge drives). Otherwise keep the historical 3-shape
    # sweep so existing usage is byte-unchanged.
    if os.environ.get("MPK_TEST_SHAPE"):
        sh = resolve_shape()
        shapes = [(sh.N, sh.K, sh.name)]
    else:
        # M=1 (decode, the lever's target) AND M>1 (prefill/ingest) for each
        # shape, since finen handles all M with no dual-dispatch.
        shapes = [
            (2176, 7168, "qkv_a"),
            (2048, 512, "kv_b"),
            (2304, 1024, "small_Nmax"),
        ]
    smoke = os.environ.get("MPK_SMOKE") == "1"
    bs_list = [1] if smoke else [1, 4, 8]
    if smoke:
        shapes = shapes[:1]
    for N, K, label in shapes:
        for bs in bs_list:
            results.append(run_case(bs, N=N, K=K, label=label))
    return _summary(results)


def _summary(results):
    print(f"\n{'='*78}\nSummary\n{'='*78}", flush=True)
    all_passed = True
    for passed, cos, rel, tag, lat in results:
        status = "PASS" if passed else "FAIL"
        lt = (f"  slowCTA={lat['slowCTA_us']:.2f}us wall={lat['wall_us']:.2f}us"
              if lat else "")
        print(f"  {status}  cos={cos:.4f} rel={rel*100:.3f}%  {tag}{lt}",
              flush=True)
        all_passed = all_passed and passed

    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({sum(r[0] for r in results)}/{len(results)})", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
