"""Perf reproducer for `rms_norm_hopper_impl` (the DSv3 production RMSNorm).

Goal
----
This test reproduces the exact RMSNorm shape and grid-shape that the DeepSeek-V3
megakernel uses at TP=4 EP=2 mbt=128, and exposes the scheduler-level
utilization story so kernel-team can see how their kernel behaves *inside*
the persistent-kernel scheduling envelope (not just in isolation).

In the production megakernel each layer issues 3-4 RMSNorm tasks:

    1. input_layernorm           (mbt=128, hidden=7168)   [post-AR2 residual]
    2. q_a_layernorm             (mbt=128,  q_lora=1536)  [in-place slice]
    3. kv_a_layernorm            (mbt=128,  kv_lora=512)  [in-place slice]
    4. post_attention_layernorm  (mbt=128, hidden=7168)   [post-AR1, gated]

The single biggest call is the hidden=7168 one. In the current trace each
launch takes 12-28 μs because the kernel uses 1 CTA per output row and only
~16 CTAs run concurrently (`rows_per_cta=8` builder default — see
builder.py::_rmsnorm_grid). At 7168 bf16 elements per row, this is bandwidth-
bound on each CTA and saturates global memory long before SM occupancy.

This test gives the kernel author a clean reproducer for each call shape AND
each launch-shape sweep so they can measure both:

  * single-shape wallclock (correctness-checked vs PyTorch ref), and
  * how the kernel behaves at the *real* grid count the megakernel uses
    (sweep `rows_per_cta` to see throughput vs CTA fan-out).

The test goes through the full PersistentKernel test_mode pipeline (task graph
build, megakernel codegen, nvcc, scheduler dispatch), so the wallclock is
representative of what the megakernel sees.

How to run
----------
This test is self-contained — invoke as a script with the GPU available to
the current process:

    python tests/runtime_python/blackwell/sm100_rmsnorm/test_rmsnorm_perf.py

Optional environment knobs:
    RMSNORM_PERF_SHAPES=hidden,q_lora    # only these tags (default = all)
    RMSNORM_PERF_ROWS_PER_CTA=1,4,8,16   # comma list to sweep (default = "1,4,8,16")
    RMSNORM_PERF_ITERS=20                # timed iters per (shape, rows_per_cta)
                                          # (default = 10)
    RMSNORM_PERF_BATCH=128               # batch / mbt to test (default = 128)
    RMSNORM_PERF_NO_REF=1                # skip the bf16 max-abs-diff check;
                                          # only collect wallclock.

The output table shows, for each (shape, rows_per_cta) pair, the per-`pk()`
wallclock distribution. With rows_per_cta=1 the kernel runs as 128 CTAs each
doing 1 row (~legacy unbatched layout); higher rows_per_cta values fan in
to fewer CTAs each looping over multiple rows. This sweep is the cleanest
way to see whether the kernel's per-row time is bandwidth-bound (in which
case fan-in doesn't help) or CTA-occupancy-bound (in which case fan-out wins).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Shape catalogue — pulled from the DSv3 19L TP=4 megakernel.
# (tag, hidden_dim)
# ---------------------------------------------------------------------------
SHAPES = [
    ("hidden",   7168),   # main input_layernorm / post_attn_layernorm — the slow one
    ("q_lora",   1536),   # q_a_layernorm
    ("kv_lora",   512),   # kv_a_layernorm
]


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps) * w.to(torch.float32)).to(x.dtype)


def run_shape(tag: str, batch: int, hidden: int,
              rows_per_cta: int, timed_iters: int, do_ref: bool, seed: int = 42):
    """Build PK test_mode that runs ONE rmsnorm_layer call with the given
    launch shape, then measure wallclock over `timed_iters` pk() invocations.

    `rows_per_cta` controls grid_dim_x:
        grid_x = ceil(batch / rows_per_cta)
    With batch=128:
        rows_per_cta=1   → grid_x=128 (legacy, 1 CTA per row)
        rows_per_cta=4   → grid_x=32
        rows_per_cta=8   → grid_x=16 (current builder default after B34)
        rows_per_cta=16  → grid_x=8
    """
    print(f"\n{'='*78}")
    print(f"[{tag}]  batch={batch}  hidden={hidden}  rows_per_cta={rows_per_cta}  "
          f"grid_x={(batch + rows_per_cta - 1) // rows_per_cta}")
    print(f"{'='*78}")

    device = "cuda"
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn((batch, hidden), device=device, dtype=torch.bfloat16, generator=g)
    w = torch.randn((hidden,), device=device, dtype=torch.bfloat16, generator=g)
    out = torch.zeros_like(x)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = nw
    params["num_local_schedulers"] = ns
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch
    params["max_num_batched_requests"] = batch
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1)  # rms_norm_hopper kernel runs 256 threads/CTA
    grid_x = max(1, (batch + rows_per_cta - 1) // rows_per_cta)
    pk.rmsnorm_layer(
        input=x_dt, weight=w_dt, output=out_dt,
        grid_dim=(grid_x, 1, 1), block_dim=block_dim,
    )

    compile_dir = os.path.join(
        THIS_DIR, f"pk_rmsnorm_perf_{tag}_b{batch}_h{hidden}_rpc{rows_per_cta}"
    )
    os.makedirs(compile_dir, exist_ok=True)

    print(f"  compiling megakernel into {compile_dir} ...")
    pk.compile(output_dir=compile_dir)

    # Warm-up.
    for _ in range(3):
        pk()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    for i in range(timed_iters):
        starts[i].record()
        pk()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(times_us)
    p50 = times_us[n // 2]
    p_min = times_us[0]
    p_max = times_us[-1]
    p_mean = sum(times_us) / n
    print(f"  wallclock per pk() call (μs): "
          f"min={p_min:.2f}  median={p50:.2f}  mean={p_mean:.2f}  max={p_max:.2f}  n={n}")

    # Bandwidth sanity:  per call we read   batch*hidden bf16  (x)
    #                                +  hidden bf16            (weight, small)
    #                  and write   batch*hidden bf16  (out).
    bytes_per_call = (batch * hidden * 2) * 2 + hidden * 2
    gbps = bytes_per_call / (p50 / 1e6) / 1e9
    print(f"  achieved bandwidth: {gbps:.1f} GB/s @ p50  "
          f"(memory volume = {bytes_per_call / 1024:.1f} KiB / call)")

    if do_ref:
        ref = torch_rmsnorm(x, w)
        max_diff = (out.float() - ref.float()).abs().max().item()
        passed = max_diff < 0.05
        print(f"  correctness: max_abs_diff={max_diff:.4f}  {'PASS' if passed else 'FAIL'}")
    else:
        passed = True
        max_diff = float("nan")

    pk.finalize()
    return {
        "tag": tag, "hidden": hidden, "rows_per_cta": rows_per_cta,
        "grid_x": grid_x, "min_us": p_min, "median_us": p50,
        "mean_us": p_mean, "max_us": p_max, "gbps_p50": gbps,
        "max_abs_diff": max_diff, "passed": passed,
    }


def main():
    only = set(s for s in os.environ.get("RMSNORM_PERF_SHAPES", "").split(",") if s)
    sweep_rpc = [
        int(x) for x in os.environ.get(
            "RMSNORM_PERF_ROWS_PER_CTA", "1,4,8,16").split(",") if x
    ]
    iters = int(os.environ.get("RMSNORM_PERF_ITERS", "10"))
    batch = int(os.environ.get("RMSNORM_PERF_BATCH", "128"))
    do_ref = os.environ.get("RMSNORM_PERF_NO_REF") != "1"

    todo = [s for s in SHAPES if not only or s[0] in only]
    if not todo:
        print(f"No shape matched RMSNORM_PERF_SHAPES={os.environ.get('RMSNORM_PERF_SHAPES')!r}",
              file=sys.stderr)
        return 2

    results = []
    for tag, hidden in todo:
        for rpc in sweep_rpc:
            results.append(run_shape(tag, batch, hidden, rpc, iters, do_ref))

    print(f"\n{'='*78}")
    print(f"SUMMARY  (batch={batch}, timed iters={iters})")
    print(f"{'='*78}")
    print(f"{'tag':10s} {'hidden':>6} {'rpc':>4} {'grid_x':>6}   "
          f"min_μs  med_μs  mean_μs  max_μs   GB/s  max_abs_diff")
    for r in results:
        print(f"{r['tag']:10s} {r['hidden']:>6d} {r['rows_per_cta']:>4d} "
              f"{r['grid_x']:>6d}   "
              f"{r['min_us']:6.2f}  {r['median_us']:6.2f}  "
              f"{r['mean_us']:6.2f}  {r['max_us']:6.2f}  "
              f"{r['gbps_p50']:5.1f}  {r['max_abs_diff']:.4f}")

    all_ok = all(r["passed"] for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
