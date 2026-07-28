---
name: mpk-faithful-gate
description: >-
  Build or run a FAITHFUL in-MPK per-task latency gate (slowCTA at the production
  grid + cos) for a DeepSeek-V3 MPK decode kernel or shape. Use this WHENEVER you
  need to measure, gate, or head-to-head-optimize an MPK kernel's per-task latency
  (dense FP8 GEMM, routed group-GEMM W13/W2, MLA decode, router/topk, AllReduce,
  etc.), stand up a faithful gate for a NEW kernel/shape, or dispatch a KDA/Ferret
  kernel agent against a faithful measure — and ESPECIALLY before trusting any
  per-task µs number in a DSv3 decode perf campaign. The faithful in-MPK
  slowCTA is the trusted measure; a standalone green-ctx bench MIS-RANKS and a
  whole-megakernel e2e number hides per-task cost — do not use either as the gate.
  Covers the slowCTA definition, the _faithful_helper reuse, the decode (M=1)
  input geometry, the GPU-broker + exclusive remote box, the gate watchdog, and
  the candidate-overlay bridge to faithful_eval.py.
---

# MPK Faithful Per-Task Gate

## Why this exists (read first — it's the whole point)

Two tempting measures LIE for the DSv3 decode campaign:

- A **standalone green-ctx kernel bench** (ferret/cpp_examples) runs the kernel alone
  on a fresh CUDA context. It MIS-RANKS — a kernel that wins standalone can be NULL or
  a REGRESS once it's compiled into the shared megakernel at the production grid with
  co-resident tasks (3 confirmed "standalone-doesn't-transfer" cases: kv-up CUDA-core,
  o_proj split-K, dense fine-N TP8).
- The **whole-megakernel e2e** per-token latency hides which task moved.

The trusted measure is the **FAITHFUL in-MPK per-task** number: compile the kernel INTO
the real megakernel, launch the PRODUCTION persistent grid (`grid.x = num_workers`, 136
on a B200), turn the persistent profiler ON, and extract THIS task's per-instance span:

- **slowCTA** = max-over-CTAs of (end − begin) = the slowest single-CTA BODY (per-instance
  compute). THIS is the verdict metric. Not P50, not per-kernel-aggregate, not wall.
- **wall** = max(end) − min(begin) = the isolated makespan (includes dispatch stagger).
- **sumCTA** = total busy CTA-time = the ANTI-SLOUGHING guard. A candidate that lowers
  slowCTA by spreading the SAME work onto more CTAs (no real speedup) shows a RISING
  sumCTA — reject it. Judge on slowCTA AND wall AND sumCTA together.
- **cos** ≥ 0.99 vs a PyTorch reference (correctness; never trade it for speed).

## The existing gates — your templates (copy structure, don't reinvent)

| Family | Gate file | Notes |
|---|---|---|
| Dense FP8 GEMM (qkv_a/q_b/q_b_pe/kv_b/o_proj/shared_*/router) | `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_*_pk_testmode.py` + `_build_helper.py` (`SHAPE_REGISTRY`) + `_faithful_helper.py` | M=1 decode; shapes are named in `SHAPE_REGISTRY` (add a row for a new dense shape) |
| Routed group-GEMM W13/W2 | `tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_fp8_group_gemm_faithful_pertask.py` | decode geometry via `MPK_GG_DECODE_M1`; active experts via `meta` row-1 mask |
| (template core, kernel-agnostic) | `sm100_fp8_gemm_dense/_faithful_helper.py` → `profiled_per_task_latency_by_id`, `resolve_num_workers`, `make_profiler_tensor` | import these; do not rewrite |
| Candidate-overlay bridge | `~/ferret/scripts/faithful_eval.py` (KIND map + `_run_test` + `_run_group_gemm`) | drives baseline vs shadow-overlay candidate in two subprocesses |
| GPU broker | `~/gpu_broker/` (`gpu_gate.sh`, `gpu_pool.conf`, `acquire/release`; machine-local) | shares local + remote box; flock-serialized; exclusivity-checked |

When the task fits one of these families, EXTEND it (add a shape row / a mode), don't
write a new harness. Only author a fresh `test_*_faithful_pertask.py` for a genuinely new
family (e.g. attention) — and even then mirror the group-GEMM file's structure exactly.

## Recipe — stand up + run a faithful gate for a new (kernel, shape)

1. **Find the task_type_id.** Get the numeric `TASK_*_SM100` enum from `runtime_header.h`
   / the kernel's `register_*` in `src/kernel/task_register.cc`. Match the profiler row
   by this NUMERIC id, NOT the name — many task names are missing from
   `profiler_persistent.py`'s `event_name_list` so the CSV writes `UNKNOWN_<id>`.
   `profiled_per_task_latency_by_id(pk, TASK_ID, iters, label=...)` handles this.

2. **Build the input at PRODUCTION decode geometry** (see "Decode geometry" below) — this
   is the #1 correctness trap. Register the kernel through `PersistentKernel(test_mode=True,
   num_workers=resolve_num_workers(...))`, attach inputs, `pk.compile()`, `pk()` once for
   correctness, then the profiled timing run.

3. **Report** slowCTA + wall + sumCTA + cos, and emit a single
   `FAITHFUL_RESULT {...}` JSON line (mirror the group-GEMM gate's `_lat_table` /
   FAITHFUL_RESULT format) so `faithful_eval.py` can parse baseline vs candidate.

4. **Per-worker-count sweep.** The optimal kernel FLIPS with the grid (GEMV wins at high
   occupancy nw≥128; tcgen05 MMA wins at low nw≤68). Sweep **nw ∈ {8, 64, 68, 128, 136}**
   (128 first), via `MPK_TEST_NUM_WORKERS` / the `--num-workers` plumb. The 64/68 points
   feed the group-GEMM ‖ shared-expert worker-partition design.

5. **Run it via the broker** (never pick a fixed GPU; faithful needs an EXCLUSIVE card):
   ```
   ~/gpu_broker/gpu_gate.sh --holder <tag> -- \
     --kind <finen|gemv_m1|largem_compact> --shape <S> --num-workers <W> \
     --configs <S> [--candidate-role smallm] [--kernel <cand.cuh>] [--baseline] [--decode-m1]
   ```
   It auto-acquires a free local-or-remote slot, runs the gate, streams
   `KERNEL_RESULT`/`FAITHFUL_RESULT` back, and releases (trap-on-EXIT). `--baseline` = the
   in-tree kernel (ratio 1.0); `--kernel <cand.cuh>` overlays a candidate via a shadow
   MIRAGE_ROOT (production tree never written).

## Decode geometry (the critical correctness point)

Production decode is **bs=1 → M=1 per active unit**. A gate that feeds an all-128-rows-real
activation measures a PREFILL-like geometry and CANNOT reward an M=1-aware kernel.

- **Dense projections**: M=1 (the dense gate is already M=1).
- **Routed group-GEMM**: exactly 1 real token per ACTIVE expert (top-k routing). Build
  `a_bf16` with only the live row (`expert*128 + 0`) of each active expert non-zero (the
  other 127 rows are pad/zero), and validate cos on the LIVE rows ONLY. This is the
  `MPK_GG_DECODE_M1=1` mode. The active experts come from `meta` row-1's `active_expert_mask`
  (~4–8 active at bs=1 TP8 EP2).
- **MLA / attention**: per-rank decode (1 query token, the real KV-seq length).

## Invariants & gotchas (these have each cost real hours — bake them in)

1. **Exclusive GPU only.** A faithful slowCTA on a contended card is INVALID. The gate
   torch-probes + exclusivity-checks (refuses on a foreign compute proc). The broker
   free-checks before acquiring.
2. **Gate watchdog (`MPK_GATE_TIMEOUT_S`, default 420s).** A buggy candidate can DEADLOCK
   the megakernel and wedge the (exclusive) box indefinitely. `faithful_eval.py`'s
   `_run_test` wraps the test in `subprocess.run(timeout=...)` → on timeout it SIGKILLs the
   child = the megakernel-HOST process → frees the card → returns rc=124 `VERDICT=TIMEOUT_HANG`.
   The timeout MUST live in the .py (which runs the test as its subprocess); a wrapper-level
   `timeout` orphans the test grandchild (it keeps holding the GPU). Keep this.
3. **Candidate kernels: NO block barrier inside the persistent-sweep tile loop.** In MPK
   each worker strides tiles by `num_workers`; workers with fewer tiles EARLY-EXIT the loop,
   so a `__syncthreads()`/`bar.sync` inside it deadlocks (the exited workers never reach it).
   Each lane/warp does its full K-sweep independently; only a final intra-warp `__shfl`
   reduction is allowed. (This deadlock wedged the exclusive box for 24 min.)
4. **Remote box runs its OWN repo copy.** The exclusive remote box (`<BOX_USER>@<BOX_IP>` —
   site-specific, resolve per session per `v2-model-support/references/box-orchestration.md`
   §1-2; its mirage repo + ferret-scripts paths are box-local) does NOT see your local edits.
   After editing a gate test / `_build_helper.py` (new shape) / `faithful_eval.py` (watchdog,
   new flag), `scp` them there (keep a `.bak`). Verify with a remote
   `python3 -c "import ast; ..."`.
5. **The remote box's `faithful_eval.sh` has an arg-allowlist** (and is OFF-LIMITS to edit —
   along with `cc-run.sh`). It REJECTS unknown flags (`--decode-m1` → "unknown arg"). Forward
   a new knob as an ENV var instead, via the machine-local box shim
   (`~/nebius_gate/faithful_eval_<box>.sh`, which translates e.g. `--decode-m1` →
   `MPK_GG_DECODE_M1=1` prefix on the remote command).
   The test reads the env var directly and `_run_test` does NOT pop it. Local-only: the
   ferret `faithful_eval.sh` similarly hardcodes a stale "GPU 5 = KDA job" refusal — route
   around it via the broker `gpu_pool.conf` (disable the blocked local slot), don't edit the .sh.
6. **Add `--num-workers` / new flags to `faithful_eval.py`** (argparse → `run()` →
   `_run_test`/`_run_group_gemm` → the test env), NOT just an env var, so they cross the
   remote shim as args. (The shim then env-translates the ones the remote .sh rejects.)
7. **Per-(shape) one number.** The gate measures ONE shape per invocation; mapping it onto
   multiple differently-shaped `--configs` mis-gates. Pass `--configs` matching `--shape`.

## Dispatching a kernel agent against the gate

When handing a kernel to KDA/Ferret (the `kda-kernel-agent` / `ferret-kernel-agent`), the
faithful gate IS the promotion authority. Give the agent: the exact `gpu_gate.sh` command
(with `--kind/--shape/--num-workers/--decode-m1`), the measured in-tree baseline slowCTA
(measure it FIRST), the vLLM ref + the ≥20% target (`≤ vLLM_ref ÷ 1.2`), the per-worker-count
sweep set, and the barrier-free-tile-loop + alignment crash-safety constraints (gotcha #3).
The agent's candidate is overlaid via `--kernel`; the shadow MIRAGE_ROOT keeps the prod tree clean.

## Output: the FAITHFUL_RESULT contract

Each gate run prints one machine-parseable line the bridge/agents consume, e.g.:
```
FAITHFUL_RESULT {"kind":"...","shape":"...","num_active":8,"candidate":{"slowCTA_us":..,"wall_us":..,"sumCTA_us":..,"cos":1.0,"status":"PASS"},"baseline_in_tree":{...},"slowCTA_ratio_base_over_cand":..,"verdict":"PASS_GATE|BELOW_BAR|TIMEOUT_HANG|INVALID"}
```
Keep this shape stable — the ferret/KDA loop greps it.
