"""In-MPK correctness test for fp8_gemm_dense_qkva_splitk_sm100 (race-free
internal split-K, ferret workspace3) via PersistentKernel test_mode.

This is the cheap in-framework validation for the NEW kernel that replaces
the CRASHED red.global.add decode_splitk: it goes through the FULL MPK
compile pipeline (graph.cc dispatch, task_register codegen, TMA descriptor
creation in tma.cuh, megakernel nvcc, scheduler dispatch) on a SINGLE GPU,
then checks the reduced BF16 output vs an FP32 reference.

DECODE-GATE DRIVING (the load-bearing part of this test, see RCA below).
The integrated kernel bakes a decode-phase gate in its task_register codegen:

    int q_len_      = qo_indptr_buffer[1] - qo_indptr_buffer[0];
    if (q_len_ > 8) return;                       // prefill -> skip
    int active_rows_ = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
    int runtime_m_   = min(active_rows_, M);
    if (runtime_m_ <= 0) return;                  // nothing to do -> skip

`qo_indptr_buffer` is NOT a static input you can pre-seed via `meta_tensors`:
`init_kernel` ZEROS it at init time, and then `prepare_next_batch`
(MODE_OFFLINE, fired at the first EVENT_END_OF_TASK_GRAPH *before* the first
real task-graph iteration) REBUILDS it entirely from the request scheduler
state (`tokens.shape[0]` => total_num_requests, `step`, `prompt_lengths`,
`num_new_tokens`). So setting `qo_indptr_buffer=arange(M+1)` before compile is
silently discarded -> the kernel sees q_len=M (prefill) or active_rows=0 and
EARLY-EXITS, leaving the sentinel-filled output untouched (cos~=0). That is a
HARNESS bug, not a kernel defect.

CORRECT DRIVING: seed the request state so prepare_next_batch produces M
single-token DECODE requests in one batch:
    tokens          : shape [M, max_seq]      -> total_num_requests = M
    prompt_lengths  : [1] * M                 -> prompt_len - step <= 0
    step            : [1] * M                 -> decode branch
    num_new_tokens  : [1] * M  (new_token_nums) -> 1 token per request
plus max_num_batched_requests = max_num_batched_tokens = M so all M requests
land in a single batch. prepare_next_batch then writes
qo_indptr_buffer = [0,1,2,...,M], i.e. q_len(req0)=1<=8 (decode gate passes)
and active_rows = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS] = M (process
all M rows).

WITNESSING THAT THE GATE PASSED. A post-pk() readback of qo_indptr is NOT a
reliable witness: prepare_next_batch fires AGAIN at the terminating
EVENT_END_OF_TASK_GRAPH and resets the buffer to zeros. The authoritative
execution-time witness is `sentinel_rows`: the kernel writes only rows
[0, active_rows); sentinel_rows == 0 (no untouched poison rows) therefore
proves active_rows reached M and the decode gate passed. Re-run with
MPK_DEBUG_BATCH=1 to print the literal exec-time batch (active_reqs /
active_tokens / per-slot q_len) from prepare_next_batch.

Shape under test: qkv_a (N=2176, K=7168). Also a tiny shape for speed.

PERFORMANCE (test-mode's MAIN purpose, not just correctness).
Test mode runs the FULL megakernel once on ONE GPU; with profiling enabled it
emits a Perfetto trace + CSV from which we read the kernel-under-test's true
in-MPK latency (standalone benchmarks don't capture the shared-worker
megakernel context). We opt in by passing `params["profiler_tensor"]` +
`params["trace_name"]` BEFORE compile; after `pk()` + sync, `<trace_name>.csv`
exists and `scripts/parse_profile.py <csv> <TASK_NAME> --stat all` returns
min/max/avg/median AND the WALL-SPAN duration for that task type. To get a perf
RATIO we run the SAME shape through the BASELINE mediumm GEMM (the path split-K
replaces) under the identical test-mode harness and compare.

KERNEL-LATENCY METRIC = WALL-SPAN, NOT median. The per-task `duration_ns` is a
per-CTA span, and at decode this GEMM is BIMODAL: grid_dim=128 CTAs are
launched but only `ceil(active_rows*N / tile)` do real work; the rest idle-exit
in <1us. So the MEDIAN duration_ns is an *idle CTA* (mediumm: ~0.66us) and
grossly understates kernel latency — using it gives a nonsense ratio (~0.06x).
The faithful single-kernel latency is the WALL-SPAN = max(end_ts)-min(begin_ts)
over the task's events (first CTA start -> last CTA finish): split-K 22.27us vs
mediumm 29.31us => 1.32x (split-K faster). The WIN/SLOWER verdict is driven by
WALL-SPAN; median/max are kept only as secondary fields that characterize the
per-CTA work split.

Run (split-K only):
    CUDA_VISIBLE_DEVICES=<free_gpu> python \
      tests/runtime_python/blackwell/sm100_fp8_gemm_dense/\
test_fp8_gemm_dense_qkva_splitk_v2_testmode.py

Run a single config (kernel = splitk | mediumm):
    CUDA_VISIBLE_DEVICES=<free_gpu> python \
      tests/runtime_python/blackwell/sm100_fp8_gemm_dense/\
test_fp8_gemm_dense_qkva_splitk_v2_testmode.py mediumm
"""

import json
import os
import subprocess
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# repo root = .../mirage (THIS_DIR is tests/runtime_python/blackwell/sm100_*)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
PARSE_PROFILE = os.path.join(REPO_ROOT, "scripts", "parse_profile.py")

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

FP8_MAX = 448.0

# TaskType name (as emitted in the profiler CSV) for each kernel variant under
# test. parse_profile.py keys on these to pull the per-task duration_ns.
TASK_NAME = {
    "splitk": "TASK_FP8_GEMM_DENSE_QKVA_SPLITK_SM100",
    "mediumm": "TASK_FP8_GEMM_DENSE_MEDIUMM_SM100",
    # External-reduce split-K: the GEMM (partial-write) reuses the QKVA_SPLITK
    # TaskType name, and a SEPARATE reduce task does the FP32->BF16 sum. The
    # in-MPK cost of this design is the SUM of the two task wall-spans.
    "extreduce": "TASK_FP8_GEMM_DENSE_QKVA_SPLITK_SM100",
}
# Companion reduce TaskType for the external-reduce path (added to the GEMM
# wall-span to get the true (GEMM+reduce) cost of the race-free design).
EXTREDUCE_REDUCE_TASK = "TASK_FP8_GEMM_DENSE_SPLITK_REDUCE_SM100"
# Task-type IDs (runtime_header.h) for direct-CSV span computation. The reduce
# (323) is event-gated strictly AFTER the GEMM (321), so the FAITHFUL combined
# in-MPK latency is GEMM.first_begin -> REDUCE.last_end (NOT the sum of the two
# independent wall-spans, which omits the inter-task dispatch gap).
GEMM_TASK_ID = "321"
REDUCE_TASK_ID = "323"


def combined_extreduce_span_us(csv_path):
    """Faithful (GEMM+reduce) latency = min(GEMM.begin) -> max(REDUCE.end), in
    us, read straight from the trace CSV. Also returns the per-task busy spans
    (max single-CTA duration) and the inter-task gap so the breakdown is clear.
    Returns None if the CSV or either task is absent."""
    import csv as _csv
    if not os.path.isfile(csv_path):
        return None
    rows = list(_csv.DictReader(open(csv_path)))
    g = [(int(r["begin_ts"]), int(r["end_ts"]))
         for r in rows if r["task_type_id"] == GEMM_TASK_ID]
    r_ = [(int(r["begin_ts"]), int(r["end_ts"]))
          for r in rows if r["task_type_id"] == REDUCE_TASK_ID]
    if not g or not r_:
        return None
    g_first = min(b for b, _ in g)
    g_last = max(e for _, e in g)
    r_first = min(b for b, _ in r_)
    r_last = max(e for _, e in r_)
    return {
        "combined_us": (r_last - g_first) / 1000.0,
        "gemm_busy_max_us": max(e - b for b, e in g) / 1000.0,
        "reduce_busy_max_us": max(e - b for b, e in r_) / 1000.0,
        "gap_us": (r_first - g_last) / 1000.0,
        "gemm_n_active": sum(1 for b, e in g if e - b > 1000),
        "reduce_n_active": sum(1 for b, e in r_ if e - b > 1000),
    }


def parse_kernel_duration_ns(csv_path, task_name):
    """Run scripts/parse_profile.py --stat all and return that task's dict.

    Returns {"min_ns","max_ns","avg_ns","median_ns","wall_ns","wall_us",
    "count"} on success, or None if the CSV/task is missing (profiling off, or
    kernel never executed). `wall_ns`/`wall_us` (the WALL-SPAN = first-CTA-start
    -> last-CTA-finish) is the kernel-latency metric used for the ratio; median
    is bimodal-skewed (an idle CTA) and reported only as a secondary field.
    Never raises: perf is reported alongside correctness, it does not gate here.
    """
    if not os.path.isfile(csv_path):
        print(f"  PERF: csv not found ({csv_path}) -> no perf number")
        return None
    try:
        out = subprocess.run(
            [sys.executable, PARSE_PROFILE, csv_path, task_name, "--stat", "all"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        print(f"  PERF: parse_profile.py failed to launch: {e}")
        return None
    if out.returncode != 0:
        print(f"  PERF: parse_profile.py rc={out.returncode} "
              f"task={task_name}: {out.stdout.strip()} {out.stderr.strip()}")
        return None
    try:
        d = json.loads(out.stdout)
    except Exception:  # noqa: BLE001
        print(f"  PERF: could not parse JSON: {out.stdout!r}")
        return None
    if "error" in d:
        print(f"  PERF: {task_name} not in trace: {d['error']}")
        return None
    return d


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


def run(M, N, K, split_k, num_workers, seed=42, kernel="splitk", profile=True):
    """Run ONE config of the kernel-under-test through full MPK test mode.

    kernel="splitk"  -> the qkv_a internal split-K FP8 GEMM (kernel-under-test)
    kernel="mediumm" -> the baseline dense-mediumm FP8 GEMM (the path split-K
                        replaces); same A/B/scales/reference, so the two
                        duration_ns are directly comparable for a perf ratio.

    Returns (passed, cos, max_diff, perf) where perf is the parse_profile dict
    {"min_ns","max_ns","avg_ns","median_ns","wall_ns","wall_us","count"} for
    this kernel's TaskType, or None if profiling was off / the kernel never
    executed. The perf ratio is driven by wall_us (WALL-SPAN), not median.
    """
    assert kernel in TASK_NAME, kernel
    label = (f"kernel={kernel} M={M}, N={N}, K={K}, "
             f"split_k={split_k}, nw={num_workers}")
    print(f"\n{'='*70}\nPK test_mode qkva_splitk_v2: {label}\n{'='*70}")
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)
    a_fp8, sa = quantize_a_f32scale(a_bf16)
    b_fp8, sb = quantize_b_f32scale(b_bf16)
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)

    # Sentinel-fill output so an early-exit (no write) is visible as non-zero
    # garbage rather than masquerading as a (wrong) all-zero pass.
    # -1024.0 is BF16-exact (a power of two); -987.0 rounds to -988.0 in BF16,
    # so an exact-equality sentinel scan against -987.0 would miss every
    # untouched row and falsely report "kernel wrote" even on a no-write exit.
    SENTINEL = -1024.0
    output = torch.full((M, N), SENTINEL, device=device, dtype=torch.bfloat16)

    # Reduction scratch (compile-time M == runtime M here, mm=ceil(M/128)).
    mm = (M + 127) // 128
    nn = (N + 127) // 128
    c_partial = torch.zeros(split_k * mm * nn * 128 * 128,
                            device=device, dtype=torch.float32)
    arrive_cnt = torch.zeros(nn, device=device, dtype=torch.int64)

    num_workers_gpu, num_sched = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = max(num_workers, num_workers_gpu)
    params["num_local_schedulers"] = num_sched
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M
    params["max_num_batched_requests"] = M
    # PERFORMANCE opt-in: a uint64 profiler buffer + absolute trace_name make
    # pk() emit <trace_name>.perfetto-trace + <trace_name>.csv after the run
    # (the CSV is what parse_profile.py reads for this kernel's duration_ns).
    # 3000*128 entries is the demo-conventional size (2 entries per task event).
    # The trace stem is absolutized into the per-config compile_dir so the CSV
    # is found regardless of the process cwd.
    compile_dir = os.path.join(
        THIS_DIR, f"pk_qkva_v2_{kernel}_{M}_{N}_{K}_sk{split_k}")
    os.makedirs(compile_dir, exist_ok=True)
    trace_stem = os.path.join(compile_dir, f"trace_{kernel}")
    if profile:
        params["profiler_tensor"] = torch.zeros(
            3000 * 128, dtype=torch.uint64, device=device)
        params["trace_name"] = trace_stem
    # Each request schedules 1 token => 1 page at page_size=128. Give the page
    # queue >= M pages so the M-request batch never wraps the page ring.
    PAGE_SIZE = 128
    params["max_seq_length"] = max(PAGE_SIZE * 2, M)
    params["max_num_pages"] = max(M, 4)
    params["page_size"] = PAGE_SIZE

    # Drive request state so prepare_next_batch (MODE_OFFLINE) emits M
    # single-token requests in one batch (see module docstring) =>
    # qo_indptr_buffer = [0,1,..,M] at execution time (q_len=1<=8, active_rows=M).
    # The qo_indptr_buffer passed here is what init_kernel zeros and
    # prepare_next_batch rebuilds; we still pass it as the runtime's buffer
    # (read back post-run only as a diagnostic — it shows the terminal reset).
    qo = torch.zeros(M + 1, dtype=torch.int32, device=device)
    tokens = torch.zeros(M, params["max_seq_length"], dtype=torch.int64,
                         device=device)
    step = torch.ones(M, dtype=torch.int32, device=device)
    prompt_lengths = torch.ones(M, dtype=torch.int32, device=device)
    num_new_tokens = torch.ones(M, dtype=torch.int32, device=device)
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo,
        "tokens": tokens,          # tokens.shape[0] == M => total_num_requests
        "step": step,              # step==prompt_len => decode branch
        "prompt_lengths": prompt_lengths,
        "num_new_tokens": num_new_tokens,  # new_token_nums: 1 token/request
    }
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8, name="a_fp8")
    b_dt = pk.attach_input(b_fp8, name="b_fp8")
    sa_dt = pk.attach_input(sa, name="sa")
    sb_dt = pk.attach_input(sb, name="sb")
    out_dt = pk.attach_input(output, name="output")

    if kernel == "splitk":
        cp_dt = pk.attach_input(c_partial, name="c_partial")
        ac_dt = pk.attach_input(arrive_cnt, name="arrive_cnt")
        pk.fp8_gemm_dense_qkva_splitk_layer(
            input_fp8=a_dt, weight_fp8=b_dt, input_scale=sa_dt,
            weight_scale=sb_dt, output=out_dt, c_partial=cp_dt,
            arrive_cnt=ac_dt, num_workers=num_workers, split_k=split_k)
    elif kernel == "extreduce":
        # RACE-FREE external-reduce: GEMM writes exclusive FP32 partials into
        # c_partial (no arrive_cnt), then a SEPARATE event-gated reduce task
        # sums split_k -> bf16 output. c_partial is the GEMM's output and the
        # reduce's input; the dep-tracker gates the reduce after the GEMM.
        cp_dt = pk.attach_input(c_partial, name="c_partial")
        pk.fp8_gemm_dense_qkva_splitk_extred_layer(
            input_fp8=a_dt, weight_fp8=b_dt, input_scale=sa_dt,
            weight_scale=sb_dt, c_partial=cp_dt,
            num_workers=num_workers, split_k=split_k)
        # ncol=4 fans the reduce columns across ~4x more CTAs (fills idle SMs
        # at decode M=1). Tunable here to characterize the reduce's occupancy.
        er_ncol = int(os.environ.get("EXTREDUCE_NCOL", "4"))
        pk.fp8_gemm_dense_splitk_reduce_layer(
            c_partial=cp_dt, output=out_dt, M=M, N=N,
            split_k=split_k, num_workers=num_workers, ncol=er_ncol)
    else:  # mediumm baseline — same A/B/scales/output, no split-K scratch.
        # runtime_m_mode=3 is the decode-phase gate (Q_LEN<=8 + active_rows
        # cap), matching the split-K kernel's baked runtime_m_mode=3 so the
        # SAME request-state drive (q_len=1, active_rows=M) makes BOTH kernels
        # run on all M rows — an apples-to-apples decode comparison.
        pk.fp8_gemm_dense_mediumm_layer(
            input_fp8=a_dt, weight_fp8=b_dt, input_scale=sa_dt,
            weight_scale=sb_dt, output=out_dt, num_workers=num_workers,
            runtime_m_mode=3)

    print("  Compiling...")
    pk.compile(output_dir=compile_dir)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    # NOTE on observing the execution-time gate. The runtime uses the SAME `qo`
    # pointer (init wires meta_tensors[6] directly), BUT prepare_next_batch fires
    # a SECOND time at the terminating EVENT_END_OF_TASK_GRAPH and RESETS the
    # whole buffer to zeros (no requests left -> num_tokens=0). So a post-pk()
    # readback shows [0,..,0] regardless of what iteration-1 used; it is NOT a
    # reliable execution-time witness. It is printed only as a diagnostic.
    # The AUTHORITATIVE execution-time gate witness is `sentinel_rows`: the
    # kernel only writes rows [0, active_rows); if active_rows had been < M the
    # tail rows would remain SENTINEL. sentinel_rows == 0 therefore proves
    # active_rows reached M (the decode gate passed and all M rows ran). To see
    # the literal qo_indptr / active_tokens the kernel gated on, re-run with
    # MPK_DEBUG_BATCH=1 (prints prepare_next_batch's [BATCH ...] lines).
    qo_rt = qo.detach().cpu().tolist()
    print(f"  qo_indptr@post-run(terminal reset, diagnostic only) = "
          f"[{qo_rt[0]}, {qo_rt[1]}, ..., {qo_rt[-1]}]  "
          f"(set MPK_DEBUG_BATCH=1 to see exec-time batch)")

    print(f"  ref[0,:4]: {ref[0,:4].tolist()}")
    print(f"  out[0,:4]: {output[0,:4].tolist()}")
    sentinel_rows = (output.float() == SENTINEL).all(dim=1).sum().item()
    if sentinel_rows:
        print(f"  HARNESS/KERNEL: {sentinel_rows} rows still SENTINEL -> kernel "
              f"did not write them. If sentinel_rows==M the decode gate was "
              f"mis-driven (q_len>8 or active_rows==0) => TEST bug; if "
              f"0<sentinel_rows<M only the gated head ran => investigate.")
    # active_rows reached M iff every row was written (no SENTINEL left).
    all_rows_written = sentinel_rows == 0
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cosine_sim(output, ref)
    passed = cos > 0.99 and all_rows_written
    print(f"  max_abs_diff={max_diff:.5f}  cos={cos:.6f}  "
          f"sentinel_rows={sentinel_rows}  all_rows_written={all_rows_written}  "
          f"-> {'PASS' if passed else 'FAIL'}")

    # PERFORMANCE: pull this kernel's in-MPK duration_ns from the trace CSV.
    # Reported alongside correctness (does NOT gate PASS/FAIL here — a vacuous
    # early-exit is already caught by sentinel_rows). The CSV was written to
    # <trace_stem>.csv by pk() above.
    perf = None
    if profile:
        csv_path = trace_stem + ".csv"
        perf = parse_kernel_duration_ns(csv_path, TASK_NAME[kernel])
        if perf is not None:
            # WALL-SPAN is the kernel-latency metric; median/max are secondary
            # (bimodal: most CTAs idle-exit, so the median is an idle CTA).
            print(f"  PERF: kernel={TASK_NAME[kernel]} count={perf['count']} "
                  f"WALL_us={perf['wall_us']:.2f} "
                  f"(median_us={perf['median_ns']/1000.0:.2f} "
                  f"max_us={perf['max_ns']/1000.0:.2f} "
                  f"avg_us={perf['avg_ns']/1000.0:.2f})")
        # External-reduce path: the true in-MPK cost is GEMM + reduce. Pull the
        # reduce task's wall-span separately and ADD it to the GEMM wall-span
        # (these two tasks are event-gated sequential — reduce starts only after
        # all GEMM CTAs finish — so summing wall-spans is the faithful cost).
        if kernel == "extreduce" and perf is not None:
            # FAITHFUL combined latency from the CSV: GEMM.first_begin ->
            # REDUCE.last_end (the reduce is event-gated strictly after the
            # GEMM, so this is the true in-MPK cost — it includes the
            # inter-task dispatch gap, unlike summing two wall-spans).
            comb = combined_extreduce_span_us(csv_path)
            if comb is not None:
                perf = dict(perf)
                perf["gemm_wall_us"] = perf["wall_us"]   # GEMM-only wall span
                perf["reduce_busy_us"] = comb["reduce_busy_max_us"]
                perf["gap_us"] = comb["gap_us"]
                perf["wall_us"] = comb["combined_us"]    # GEMM.first->REDUCE.last
                perf["wall_ns"] = comb["combined_us"] * 1000.0
                print(f"  PERF: extreduce TOTAL (GEMM.first->REDUCE.last) "
                      f"WALL_us={perf['wall_us']:.2f} "
                      f"(GEMM_wall={perf['gemm_wall_us']:.2f} active="
                      f"{comb['gemm_n_active']}, gap={comb['gap_us']:.2f}, "
                      f"REDUCE_busy={comb['reduce_busy_max_us']:.2f} active="
                      f"{comb['reduce_n_active']})")
            else:
                print("  PERF: reduce task not in trace (gemm-only wall shown)")
    pk.finalize()
    return passed, cos, max_diff, perf


def _wall_us(perf):
    """Kernel-latency metric: WALL-SPAN in us (drives the WIN/SLOWER ratio)."""
    return None if perf is None else perf["wall_us"]


def _median_us(perf):
    """Secondary characterization only — do NOT drive the verdict (bimodal)."""
    return None if perf is None else perf["median_ns"] / 1000.0


def _max_us(perf):
    return None if perf is None else perf["max_ns"] / 1000.0


def main():
    # Optional CLI selector: "extreduce" | "mediumm" | "ervm" | "splitk".
    #
    # DESIGN-A IS NOW THE ONLY SPLIT-K PATH (2026-06-03). The in-kernel
    # last-arriver reduction (kernel="splitk") was retired: its non-extred entry
    # point now writes ONLY exclusive FP32 partials (race-free) and does NOT
    # reduce in-kernel, so running it WITHOUT a companion reduce leaves the
    # output unfilled (cos~=0). The validated path is the external-reduce
    # ("extreduce") design: split-K GEMM partial-write + a SEPARATE event-gated
    # reduce task (enum-323) that sums in FP32 and casts to BF16, compared vs the
    # torch FP32 reference. The DEFAULT selector is therefore "ervm" (extreduce +
    # mediumm baseline) so the gate validates Design-A correctness AND reports
    # its perf ratio vs the mediumm path it replaces.
    #   ervm      -> extreduce (Design-A) + mediumm baseline (default; the gate)
    #   extreduce -> Design-A correctness only
    #   mediumm   -> baseline only
    #   splitk    -> RETIRED in-kernel last-arriver (kept only as a tripwire:
    #                it now write-partials-only, so this is EXPECTED to FAIL
    #                cos/sentinel — do not use it as a correctness gate).
    which = sys.argv[1] if len(sys.argv) > 1 else "ervm"
    assert which in ("splitk", "mediumm", "both", "extreduce", "ervm"), which

    results = {}
    perf = {}
    # the real qkv_a shape: N=2176, K=7168, split_k=4 (K%512==0). M=128=mbt.
    QSHAPE = dict(M=128, N=2176, K=7168, split_k=4, num_workers=128)

    if which in ("splitk", "both"):
        # tiny shape first (fast compile + K divisible by 128*4) — correctness
        # smoke only; perf number not meaningful at this size.
        p, c, d, _ = run(M=128, N=256, K=512, split_k=4, num_workers=64,
                         kernel="splitk")
        results["tiny M128 N256 K512 sk4 (splitk)"] = (p, c, d)
        p, c, d, pf = run(kernel="splitk", **QSHAPE)
        results["qkv_a (splitk)"] = (p, c, d)
        perf["splitk"] = pf

    if which in ("extreduce", "ervm"):
        # tiny shape first — correctness smoke for the external-reduce path
        # (GEMM partial-write + event-gated reduce); perf not meaningful here.
        p, c, d, _ = run(M=128, N=256, K=512, split_k=4, num_workers=64,
                         kernel="extreduce")
        results["tiny M128 N256 K512 sk4 (extreduce)"] = (p, c, d)
        p, c, d, pf = run(kernel="extreduce", **QSHAPE)
        results["qkv_a (extreduce GEMM+reduce)"] = (p, c, d)
        perf["extreduce"] = pf

    if which in ("mediumm", "both", "ervm"):
        p, c, d, pf = run(kernel="mediumm", **QSHAPE)
        results["qkv_a (mediumm baseline)"] = (p, c, d)
        perf["mediumm"] = pf

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    ok = True
    for k, (p, c, d) in results.items():
        print(f"  {k}: {'PASS' if p else 'FAIL'} cos={c:.4f} maxdiff={d:.4f}")
        ok = ok and p

    # PERFORMANCE summary + ratio. Machine-greppable PERF_SUMMARY line.
    # The WIN/SLOWER verdict is driven by WALL-SPAN (max(end_ts)-min(begin_ts)),
    # NOT median: this GEMM is bimodal at decode (most of its grid_dim=128 CTAs
    # idle-exit in <1us, only ~17 do real work), so the median duration_ns is an
    # idle CTA and understates kernel latency by ~30x. median/max are reported
    # as secondary fields only — they characterize the per-CTA work split but
    # must not drive the perf decision.
    sk_us = _wall_us(perf.get("splitk"))
    mm_us = _wall_us(perf.get("mediumm"))
    sk_med = _median_us(perf.get("splitk"))
    mm_med = _median_us(perf.get("mediumm"))
    sk_max = _max_us(perf.get("splitk"))
    mm_max = _max_us(perf.get("mediumm"))
    print(f"\n{'-'*70}\nPERF (in-MPK single-kernel WALL-SPAN, qkv_a N=2176 "
          f"K=7168, 1 GPU)\n{'-'*70}")
    print(f"  splitk_wall_us  = {sk_us if sk_us is None else f'{sk_us:.2f}'}"
          f"   (median_us={'-' if sk_med is None else f'{sk_med:.2f}'}, "
          f"max_us={'-' if sk_max is None else f'{sk_max:.2f}'})  [secondary]")
    print(f"  mediumm_wall_us = {mm_us if mm_us is None else f'{mm_us:.2f}'}"
          f"   (median_us={'-' if mm_med is None else f'{mm_med:.2f}'}, "
          f"max_us={'-' if mm_max is None else f'{mm_max:.2f}'})  [secondary]")
    if sk_us is not None and mm_us is not None and sk_us > 0:
        ratio = mm_us / sk_us  # >1 => split-K faster than mediumm baseline
        print(f"  ratio (mediumm_wall/splitk_wall) = {ratio:.3f}x  "
              f"({'splitk FASTER' if ratio > 1 else 'splitk SLOWER'})")
        # Machine-greppable line scraped by ferret/scripts/mpk_validate.sh.
        # NOTE field names: splitk_wall_us / mediumm_wall_us (WALL-SPAN driven).
        med_fields = ""
        if sk_med is not None and mm_med is not None:
            med_fields = (f" splitk_median_us={sk_med:.2f} "
                          f"mediumm_median_us={mm_med:.2f}")
        print(f"PERF_SUMMARY: splitk_wall_us={sk_us:.2f} "
              f"mediumm_wall_us={mm_us:.2f} ratio={ratio:.3f}{med_fields}")
    elif which not in ("extreduce", "ervm"):
        print("PERF_SUMMARY: incomplete (need both splitk + mediumm; "
              "run with no arg or 'both')")

    # EXTERNAL-REDUCE perf summary + ratio. wall_us here is the FOLDED
    # (GEMM + reduce) total (see run()): the faithful in-MPK cost of the
    # race-free design. ratio = mediumm_wall / (GEMM+reduce)_wall; >1 => the
    # external-reduce split-K NETS a win over the mediumm baseline.
    er = perf.get("extreduce")
    er_us = _wall_us(er)
    if er_us is not None:
        gemm_us = er.get("gemm_wall_us")        # GEMM-only wall span
        red_us = er.get("reduce_busy_us")        # reduce busy-max (its real work)
        gap_us = er.get("gap_us")                # inter-task dispatch gap
        print(f"\n{'-'*70}\nPERF (external-reduce split-K, qkv_a N=2176 K=7168, "
              f"1 GPU)\n{'-'*70}")
        extra = ""
        if gemm_us is not None and red_us is not None:
            extra = (f"   (GEMM_wall={gemm_us:.2f} + gap={gap_us:.2f} + "
                     f"reduce_busy={red_us:.2f})")
        print(f"  extreduce_total_wall_us = {er_us:.2f}{extra}")
        if mm_us is not None:
            print(f"  mediumm_wall_us         = {mm_us:.2f}")
            if er_us > 0:
                er_ratio = mm_us / er_us
                print(f"  ratio (mediumm/extreduce_total) = {er_ratio:.3f}x  "
                      f"({'extreduce NET WIN' if er_ratio > 1 else 'extreduce SLOWER'})")
                gr_fields = ""
                if gemm_us is not None and red_us is not None:
                    gr_fields = (f" extreduce_gemm_us={gemm_us:.2f} "
                                 f"extreduce_reduce_busy_us={red_us:.2f}")
                print(f"PERF_SUMMARY_EXTREDUCE: extreduce_total_wall_us={er_us:.2f} "
                      f"mediumm_wall_us={mm_us:.2f} ratio={er_ratio:.3f}{gr_fields}")
        else:
            print("PERF_SUMMARY_EXTREDUCE: (run 'ervm' for the mediumm ratio)")

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
