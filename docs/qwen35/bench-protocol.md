# vLLM benchmark protocol — the pinned measurement contract (M1-I6)

This is the measurement contract for the vLLM side of AC-4 (decode throughput) and AC-5 (e2e
latency). It is pinned once here; the M4 accept harness and every vLLM baseline re-capture
reuse it verbatim. The implementation is `workspace/demo/qwen3_5/accept/bench_vllm.py`;
captured artifacts live under `workspace/demo/qwen3_5/accept/baselines/vllm-<version>-<date>/`.
This doc does not carry numbers — it carries the *method*, so it does not need editing every
time the sweep is re-run. Read `vllm-graph.md` §3.5–3.7 and `v1-architecture.md` §8–9 first;
this doc cites both rather than re-deriving their findings.

---

## 1. Model + checkpoint identity

- Model: `Qwen/Qwen3.5-35B-A3B-FP8`.
- Revision (pinned sha, not `main`): `9d1823d2dee688a6b25e77009dc727688c44936e` — same revision
  as the AC-3 HF reference and the M1-I5 vLLM smoke (`reference/README.md`).
- Same checkpoint on both sides of AC-4/AC-5 by construction — there is no second "mpk
  checkpoint," MPK loads and re-quantizes this same FP8 checkpoint (constraint.md 2a).

## 2. Pinned workload

**input_len = 256, output_len = 1024, identical across every batch size.** Both are within
goal.md's bounds (input ≥ 64, output ≥ 256).

Why this pair and not the 1024/1024 first considered:

- The MPK v1 architecture decision (`v1-architecture.md` §8.2) shows MPK's prefill runs as
  chunked in-kernel iterations, each costing about as much as one B=16 decode step
  (`prefill_iters = ⌈B·I/mbt⌉`, `mbt=16`). At B=16 that overhead is `4·I/O` of the decode
  budget. For AC-5 (e2e ≤ 1.25× vLLM) to be closeable from AC-4 (decode win) alone — the
  argument `v1-architecture.md` §8.2 spells out — the workload needs `I ≲ O/4` at B=16. The
  doc's own literal recommendation sitting exactly on that line is I=64/O=256; **256/1024 keeps
  the identical 1:4 ratio** (same margin, "meets it with equality" per §8.2) while giving a
  longer, more realistic prompt and a longer decode window.
- A longer, four-times-scaled decode window (1024 vs 256 steps) is also independently better
  *for this issue's own measurement*: with `output_len` steps of the same fixed per-step cost,
  fixed one-time effects (any residual first-touch/cache-fill left after engine warmup) amortize
  into a smaller fraction of the window, and dispersion sampled over more steps is a more
  representative estimate of steady state.
- At batch size 16, `input_len × 16 = 4096` tokens of prefill — comfortably inside vLLM's
  default chunked-prefill budget (`max_num_batched_tokens`, observed 16384 by default, recorded
  per-run in `engine_assertions.max_num_batched_tokens`) — so every batch size in the sweep
  prefills in a single scheduler chunk, keeping the prefill/decode *regime* identical across the
  whole sweep (no batch size crosses into multi-chunk prefill while others don't).
- **This choice is coordinator-directed** (mid-task correction, superseding an earlier
  1024/1024 recommendation, citing `v1-architecture.md` §8.2) and is recorded here as the final
  pinned value per constraint.md 2b ("secondary details... are the coordinator's call within the
  pinned protocol bounds"). No measurement had been taken under 1024/1024 before the correction
  arrived, so there is no stale/superseded artifact to reconcile.

Prompt **content** is synthetic: `batch_size` distinct sequences of exactly `input_len` real
(non-special) token ids, sampled uniformly at random with a fixed seed
(`bench_vllm.py:build_synthetic_prompts`). This is deliberate and standard for a throughput
benchmark (vLLM's own `benchmark_throughput.py --dataset-name random` does the same): every
GEMM here (dense fp8 linears, the FlashInfer TRT-LLM MoE kernel) does fixed-shape work
independent of token identity, so content carries no fairness-relevant signal, only *length*
does. `tokenizer.vocab_size` (248044 for this checkpoint) already excludes every special-token
id — all registered special ids are ≥ 248044 — so sampling uniformly from `[0, vocab_size)`
cannot emit a special/control token; no extra filtering needed.

Content is **fresh per `(batch_size, rep)`**, not reused across reps: `seed = seed_base +
batch_size·1000 + rep_index`. Reusing identical content across reps was the first design
considered (it would remove content as a variance source) but was rejected: it would make the
protocol depend on `enable_prefix_caching` staying at its currently-observed resolved value
(`False` for this model/config — note the bare class default is actually `True`,
`vllm/config/cache.py:93`; something in this model's config resolution turns it off, recorded
per-run in the artifact rather than assumed) to avoid a rep-2-onward cache-hit bias. Fresh-per-rep content sidesteps
that dependency entirely — with no request in the whole sweep sharing a prefix with any other,
prefix caching cannot affect the measurement regardless of whether it is on or off, so nothing
needs to be overridden away from "best standard config" defaults. Every request additionally
carries the checkpoint-content-independence argument above, so this swap costs nothing in
variance and removes a fragile assumption.

## 3. Batch sizes, decoding, speculative decoding

- Batch sizes: `{1, 2, 4, 8, 16}` — pinned by AC-4, not a choice made here.
- Greedy: `SamplingParams(temperature=0.0, top_p=1.0)`. vLLM treats `temperature=0` as exact
  argmax (matches AC-2/AC-3's `do_sample=False` convention on the HF side).
- Every request is forced to generate **exactly** `output_len` tokens:
  `ignore_eos=True, min_tokens=output_len, max_tokens=output_len` (all three set — belt and
  suspenders). This is required for a clean batch-level decode window (§5): with heterogeneous
  stop points, "batch size N decode throughput" stops being well-defined once fewer than N
  requests remain in flight. `bench_vllm.py` asserts `finish_reason == "length"` and
  `len(token_ids) == output_len` for every request in every rep and raises if violated.
- No speculative decoding / MTP on the vLLM side: `speculative_config` is never set (default
  `None`), matching AC-2.

## 4. vLLM's "best standard config"

Per constraint.md 2a, vLLM gets its best standard config: CUDA graphs allowed, no explicit
backend overrides. Concretely, `bench_vllm.py`'s `LLM(...)` call passes only: `model`,
`revision`, `dtype="auto"`, `gpu_memory_utilization=0.85`, `max_model_len=input_len+output_len`,
`disable_log_stats=False` (needed for timing, §5 — does not change any compute path),
`language_model_only` (this section), and `seed=0` (defensive; inert under pure greedy
decoding). Nothing else is overridden: `enforce_eager` stays `False` (CUDA graphs on),
`--moe-backend`/`--linear-backend`/`--kv-cache-dtype`/`--mamba-ssm-cache-dtype` all stay
`auto`, no LoRA, TP=PP=DP=1. Every one of these resolved values is recorded per run in
`engine_assertions` (§8) rather than assumed.

### `--language-model-only` ruling

`vllm-graph.md` §2.2.3 documents that `language_model_only=True` switches the full-attention
layers' QK-norm+RoPE+gate computation from three separate ops (chunk → RMSNorm → RoPE) to one
fused Triton launch (`fused_qk_rmsnorm_rope_gate`), and states both "compute the same function."
It is off by default (`vllm/config/multimodal.py:77-78`) because this checkpoint's architecture
carries a multimodal config even though AC-2 scopes vision out. The goal says give vLLM its
*best* standard config, so this needs a ruling, not an assumption.

**Verification method**: `bench_vllm.py --mode ruling`, invoked twice (once
`--language-model-only on`, once `off`), each booting its own engine and running the same
fixed-seed synthetic prompts at batch sizes {1, 4} for 64 greedy output tokens, writing
`ruling_lmo_on.json` / `ruling_lmo_off.json` with full output token ids. Ruling: **include
`language_model_only=True` in the pinned config iff neither run crashes and every output token
id is byte-identical between the two files at both batch sizes; otherwise document the failure
and pin `False`.**

**Result** (see `baselines/<run>/ruling_lmo_on.json` / `ruling_lmo_off.json` for the raw
artifacts): recorded in the run's `summary.json.shared_meta.cli_args.language_model_only` and in
this issue's final report. The sweep in §6 always records which value was pinned
(`engine_assertions.language_model_only`) so a reader never has to trust this section alone.

## 5. Timing boundaries

vLLM's own OTLP tracing code (`vllm/v1/engine/output_processor.py:712-733`, `do_tracing`, and
independently re-derived at `vllm/v1/metrics/stats.py:437-448`) defines, per request, from
`RequestStateStats` (`vllm/v1/metrics/stats.py:202-217`):

```
prefill_time  = first_token_ts - scheduled_ts
decode_time   = last_token_ts  - first_token_ts
inference_time= last_token_ts  - scheduled_ts        (= prefill_time + decode_time)
```

`scheduled_ts`, `first_token_ts`, `last_token_ts` are all stamped with `time.monotonic()` inside
the EngineCore process (`vllm/v1/core/sched/scheduler.py:430` for `scheduled_ts`;
`vllm/v1/engine/__init__.py:248` for the `EngineCoreOutputs.timestamp` that becomes
`first_token_ts`/`last_token_ts`). `CLOCK_MONOTONIC` is machine-wide on Linux (not
per-process), so these remain directly comparable even though EngineCore runs in a separate
spawned process from the main script (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) — no cross-process
clock-skew concern on one host. This benchmark reuses vLLM's own boundary definition rather
than inventing a new one: **decode excludes prefill by definition** (`decode_time` starts only
once the first output token — necessarily produced by the prefill forward pass — has already
been emitted).

`RequestOutput.metrics` (the `RequestStateStats` above) is `None` unless the engine is
constructed with `disable_log_stats=False` — the offline `LLM` class flips this to `True` by
default (`vllm/entrypoints/llm.py:235-236`), unlike the server default; `bench_vllm.py` passes
`disable_log_stats=False` explicitly. This only enables statistics bookkeeping, it does not
change any compute path.

### 5.1 Steady-state decode throughput (AC-4)

Per rep, over the `batch_size` requests that were all submitted together and are all forced to
produce exactly `output_len` tokens (§3):

```
decode_window_start = max over requests of first_token_ts   # latest request to enter decode
decode_window_end   = min over requests of last_token_ts    # earliest request to finish
decode_wall_seconds = decode_window_end − decode_window_start
decode_tokens        = batch_size × (output_len − 1)         # token 1 came from prefill, excluded
decode_tokens_per_second = decode_tokens / decode_wall_seconds
```

`max(first_token_ts)`/`min(last_token_ts)` (not `min`/`max` the other way, and not an average)
is deliberate: `[decode_window_start, decode_window_end]` is the *largest interval guaranteed to
have all `batch_size` requests simultaneously in their decode phase* — before
`decode_window_start` at least one request is still finishing prefill, after
`decode_window_end` at least one request has already completed and the true concurrency has
dropped below `batch_size`. Throughput measured over any other window would not be "batch size
`batch_size`, steady-state" by construction. Because every request shares `input_len`/
`output_len` and is submitted in one `llm.generate()` call, in practice `first_token_ts` and
`last_token_ts` cluster tightly across requests (one shared decode step produces every
request's token at ~the same instant), so this definition and its naive average differ
negligibly — but the definition above is the one that is *correct by construction*, not just
empirically close.

### 5.2 End-to-end request latency (AC-5)

`e2e_wall_seconds` = a plain `time.perf_counter()` bracket placed by `bench_vllm.py` around the
entire `llm.generate(...)` call for the batch (both timestamps taken in the same process, same
clock — no cross-process comparison, so no dependency on the monotonic-clock argument above).
This is the time for prefill **and** decode of every request in the batch to fully complete,
i.e. exactly "prefill + decode" per AC-5's own wording, with no exclusions. Because all
`batch_size` requests are homogeneous and submitted together, every request's own
`inference_time` (§5, `last_token_ts − scheduled_ts`) is expected to sit close to
`e2e_wall_seconds` minus a small constant (Python-side submission/collection overhead); each rep
also records the median/min/max of `inference_time` across requests as a diagnostic — a
meaningful divergence between that median and `e2e_wall_seconds` is itself worth root-causing
before trusting the number, not something the protocol papers over.

## 6. Warmup, repetitions, dispersion

- **Engine-level warmup** happens once, inside `LLM(...)` construction: CUDA graph capture,
  `torch.compile`, and FlashInfer's autotune cache are all built/loaded before the constructor
  returns (this is also where the fp8-path assertions of §8 are collected — kernel selection is
  static at `create_weights()` time, `vllm-graph.md` §3.5, so it cannot change mid-sweep on one
  engine instance).
- **Per-batch-size warmup**: 1 full rep at that batch size, identical to a measured rep, run and
  discarded before the measured reps. vLLM's default `cudagraph_capture_sizes` already include
  every batch size in {1,2,4,8,16} (captured once at engine construction, not on first use per
  size), so this rep is not covering graph capture; it is insurance for anything else that could
  still be shape-sensitive on first touch (allocator steady state, OS page-cache warmth for the
  freshly-touched KV blocks at that shape) and for uniformity/rigor across the sweep.
- **≥ 3 measured reps per batch size** (default 3, `--reps`). Each rep draws fresh synthetic
  content (§2) — this is *not* a source of measurement noise for the reasons given in §2, so
  observed rep-to-rep variance in a valid run should reflect system noise (scheduler jitter,
  clock/thermal drift), which is exactly what the dispersion check below is meant to surface.
- **Co-tenant re-check, per rep**: `nvidia-smi --query-compute-apps` is snapshotted right after
  engine construction (this engine's own worker process(es) become the trusted baseline PID
  set) and re-checked immediately before and after every rep. Any PID appearing that is not in
  the baseline set marks that rep `valid: false` and it is excluded from the summary statistics
  — matching resources.md's "measurements are only valid with ZERO co-tenants... discard any rep
  where a co-tenant appeared." `bench_vllm.py` automatically runs extra attempts (bounded) to
  backfill discarded reps up to the required count, and raises rather than silently reporting a
  short count.
- **Max-dispersion bound for a valid run: `(max − min) / median ≤ 5%`** on
  `decode_tokens_per_second` across the measured reps at a given batch size. This is a pinned,
  not tuned, bound: CUDA-graph-replayed decode on an exclusive GPU is expected to be very low
  noise (no Python launch-overhead variance once graphs are captured; the residual sources are
  GPU boost-clock/thermal drift, on the order of a few percent at most on a properly cooled
  datacenter part). If a batch size's run exceeds the bound, `bench_vllm.py` flags
  `dispersion_ok: false` loudly in both the log and the JSON but does **not** silently drop or
  re-roll reps to hide it — the correct response is to root-cause it first (cross-check the
  recorded `gpu_clocks_before`/`gpu_clocks_after` for throttling, re-confirm no co-tenant slipped
  past the PID check, e.g. a process that exited between checks) before treating that batch
  size's median as final.
- **Reported spread**: `median ± (max − min)/2` (half-range) for the human-facing table; the
  full per-rep array and `dispersion_pct` are always retained in the JSON.
- **Escalation rule when a run exceeds the bound (measured amendment, 2026-07-25; statistic
  CORRECTED same day after independent review):** run a SECOND independent engine boot of the
  same batch size (≥3 more reps, same protocol; ≥6 reps total). The binding value becomes the
  median over the MERGED rep set, valid iff (a) the merged-set **IQR/median ≤ 5 %** and (b)
  every boot's median deviates ≤ 3 % from the merged median (`bench_vllm.py --mode merge`
  computes and fails closed). The merged FULL range/median is recorded but NOT bounded: range
  is monotone in rep count — one outlier rep would dominate a merged set forever, making
  "collect more evidence" counterproductive; IQR is the standard robust replacement. (The
  first version of this amendment quoted half-range numbers by mistake — caught in review;
  the honest full-range values are below.) Empirical basis on the shared B200 host (co-tenant
  HBM/power cross-effects; GPUs 0/1/4 fully loaded): bs 8/16 single-boot full-range dispersion
  5.2 %/6.4 %; merged (n=8) full-range 6.0 %/6.9 % — driven by isolated FAST outlier reps, so
  the median is conservative in vLLM's favor — while merged IQR/median is 3.1 %/3.4 % and
  boot medians deviate ≤ 2.1 %/1.1 %. bs ≤ 4 passed the single-boot bound outright
  (0.06–3.5 %).

## 7. FP8-path and fairness assertions

Every sweep run performs these checks **once**, immediately after engine construction (§4:
kernel selection is static per engine instance), and embeds the result in every per-batch-size
artifact. Source: `vllm-graph.md` §3.7.5 ("what the benchmark must pin"), narrowed to what's
mechanically checkable from the offline Python API plus the combined stdout+stderr log.

**Hard assertions — the run aborts (raises, non-zero exit) if any fail:**

| # | Check | How |
|---|---|---|
| 1 | Quantization is fp8 | `llm.llm_engine.vllm_config.model_config.quantization == "fp8"` |
| 2 | FlashInfer present and its TRT-LLM fused-MoE symbols are available | `has_flashinfer()` and `has_flashinfer_trtllm_fused_moe()` (`vllm/utils/flashinfer.py:48,244`) both `True` |
| 3 | DeepGEMM auto-disabled for this model on Blackwell | log contains `"Auto-disabled DeepGemm"` and `"Falling back to CUTLASS"` |
| 4 | Dense linear kernel is CUTLASS block-scaled | log contains `"Selected CutlassFp8BlockScaledMMKernel"` |
| 5 | MoE backend is FlashInfer TRT-LLM | log contains `"Using FLASHINFER_TRTLLM Fp8 MoE backend"` |
| 6 | `VLLM_TEST_FORCE_FP8_MARLIN` not truthy | direct env read |
| 7 | `VLLM_BATCH_INVARIANT` not truthy | direct env read |
| 8 | TP = PP = DP = 1 | `vllm_config.parallel_config` |
| 9 | No LoRA | `vllm_config.lora_config is None` |

Checks 3–5 are **log-substring** checks against the combined stdout+stderr of the whole process
tree (the DeepGEMM-autodisable warning is logged in the main process; the kernel-selection lines
are logged inside the spawned EngineCore worker — confirmed by inspecting
`vllm_smoke_run.log` from M1-I5, where both appear in the same captured file). There is no
cross-process Python object to introspect for the two subprocess-side lines (weight loading and
kernel selection happen in the worker's own memory), so `bench_vllm.py` requires its caller's
wrapper script to redirect the *entire* run's stdout+stderr to the exact path passed as
`--log-file`, then re-reads that file itself after engine construction to build this table —
self-contained rather than trusting the caller got the redirect right.

**Recorded, not hard-asserted** (their current defaults are already what "best standard config"
wants; a future default change should be visible, not silently masked by an assertion that would
just be updated to match): `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM`,
`VLLM_DISABLED_KERNELS`, `moe_backend`/`linear_backend` resolved values, `kv_cache_dtype`,
`mamba_cache_dtype`/`mamba_ssm_cache_dtype` — confirmed empirically (first sweep run) to read
back `"auto"` / `"float32"` respectively: `mamba_ssm_cache_dtype` is not left as the literal
string `"auto"`, vLLM's own config resolution already eagerly resolves it against the
checkpoint's `mamba_ssm_dtype=float32` (M1-I1) before this script ever reads it — a stronger,
more direct confirmation than assumed when this section was first drafted — `enable_prefix_caching`,
`enforce_eager`, `max_num_batched_tokens`, flashinfer version.

## 8. Artifact JSON schema

One JSON per batch size, `bsN.json`, plus one `summary.json` per run directory. Every `bsN.json`
is self-contained (repeats `shared_meta`) so a single file is reviewable on its own.

```
shared_meta:
  schema_version, run_tag, generated_at_utc, model_id, revision,
  gpu_id_physical, gpu_preflight {memory_used_mib, utilization_pct},
  engine_load_seconds, engine_assertions {…all of §7, plus resolved config…},
  versions {python, torch, vllm, transformers, flashinfer}, cli_args {…every flag…}

per batch-size file (bsN.json):
  batch_size, input_len, output_len,
  gpu_clocks_before / gpu_clocks_after: {graphics_mhz, sm_mhz, memory_mhz, video_mhz}
                                          (nvidia-smi -q -d CLOCK, first "Clocks" block)
  warmup: [ {..rep record.., role:"warmup"} ]
  reps:   [ {..rep record.., role:"measured", valid:true, co_tenant_pre, co_tenant_post} ]
  discarded_reps: [ {..rep record.., valid:false, why co-tenant info..} ]
  summary:
    decode_tokens_per_second: {median, min, max, spread_half_range, dispersion_pct, n}
    e2e_latency_seconds:      {median, min, max, spread_half_range, dispersion_pct, n}
    max_dispersion_pct_bound, dispersion_ok
  shared_meta: <as above>

rep record:
  e2e_wall_seconds, decode_window_start, decode_window_end, decode_wall_seconds,
  decode_tokens, decode_tokens_per_second,
  per_request_inference_time_{median,min,max}_s,
  batch_size, seed, output_ids_sha256, sample_output_ids_request0
```

`summary.json` additionally holds a `table` keyed by batch size with just the two `summary`
blocks, for quick cross-batch-size reading without opening all five files.

## 9. Execution recipe

```bash
# on catalyst-B200, user muhengl - GPU etiquette first, every time:
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# pick a GPU with ~0% util and <500MiB used - it becomes <gpu_id> below.

# scripts/run_bench_vllm.sh (on the B200 box, not committed to this repo - mechanical
# wrapper under ~/mpk-qwen35/scripts/) takes: <mode> <output_dir> <log_file>
# <language_model_only: on|off> <gpu_id> [extra bench_vllm.py args...]. It sets
# CUDA_VISIBLE_DEVICES/HF_HOME/PATH, activates venv-vllm, does `exec > "$LOG" 2>&1` as its
# FIRST action so the *entire* process tree's output (including the spawned EngineCore
# worker) lands in one file at the exact path it also passes to Python via --log-file, then
# execs bench_vllm.py.

# language-model-only ruling (once; both must be run before the sweep is invoked with a
# final --language-model-only value) - diff the two ruling_lmo_*.json output_ids afterward:
bash scripts/run_bench_vllm.sh ruling <run_dir> <run_dir>/ruling_on.log  on  <gpu_id>
bash scripts/run_bench_vllm.sh ruling <run_dir> <run_dir>/ruling_off.log off <gpu_id>

# pinned sweep (long-running - engine construction alone took ~500s warm-cache during this
# issue's own runs, and the full 5-batch-size x >=4-rep decode workload adds more):
bash scripts/run_bench_vllm.sh sweep <run_dir> <run_dir>/sweep.log <on|off> <gpu_id>
```

Long-running note (resources.md: "run under nohup/tmux, poll via ssh; don't hold an
interactive ssh open as the job owner"): plain `nohup ... & disown` launched *through a
one-shot non-interactive `ssh host "cmd &"` call* was tried first for this issue's own runs
and did **not** reliably survive the SSH session closing in this environment (the child never
appeared in `ps` afterward, despite `nohup`) - root cause not fully isolated (a job-control /
SIGHUP interaction specific to non-pty SSH exec is the leading suspect, not confirmed). What
worked reliably: run the wrapper as the remote command of a *single persistent* `ssh` call and
manage its lifetime as a background job on the **caller** side (e.g. this repo's agent
tooling backgrounds the `ssh ... 'bash scripts/run_bench_vllm.sh ...'` invocation itself and
polls `<log_file>` for the `BENCH_VLLM_EXIT_CODE=` marker written at the end of every wrapper
run); a `tmux`/`screen` session on the B200 box is the equivalent manual pattern. Either way,
poll the log file's tail rather than holding a foreground terminal open on it.

## 10. Where the numbers live

This doc intentionally carries no measured numbers. Captured baselines are commit-ready
artifacts under `workspace/demo/qwen3_5/accept/baselines/vllm-<version>-<date>/` — see that
directory's own contents (`summary.json` for the cross-batch-size table) and the M1-I6 issue
report for the headline table. M3/M4 re-runs of this exact protocol land in sibling
`vllm-<version>-<date>/` directories, keyed by vLLM version and capture date, never overwriting
a prior capture in place.

## Admission-cap policy (M3-I9 landing, 2026-07-27)

Binding for every benchmark and for M4's final harness: **`--per-request-token-cap auto` at
bs16; NO cap at bs 1/2/4/8.** Basis: at bs16 cap=1 == the uncapped chunk structure (mbt=16
already admits ~1 token/request/iteration), tokens are byte-identical to the adjudicated M2
dumps, live in-wave compaction drops to ZERO migrations (the six contaminated duplicate
pairs flip to identical:true), and the wave gains +84.2% at the AC-3 geometry / +14.1% e2e
at the matched 256/1024 workload. At bs<16 the cap genuinely changes prefill chunk
boundaries — perf-neutral-to-negative by design AND numerically shifted at bs4 (p10-logic
margin-0.625 flip; see M3-I9b) — so it stays off. Evidence: opt/m3i9/{predictions.md,
predictions_addendum.md,results/}.

## Determinism protocol (M3-I9b/I11, 2026-07-27)

Rules, in force regardless of where the remaining nondeterminism comes from: (a) any
"policy/config X changed the tokens" claim requires >=2 same-config reps (a single divergent
dump is an anomaly candidate, not a finding); (b) margin/waiver arguments use the ENGINE's
own logits (reference-side margins overstate robustness — p10@49 was 0.625 reference-side
but 0.375 = 3 bf16 ULPs engine-side); (c) harness runs refuse ambiguous dump trees —
`run_ac3.py --engine-dump-dir` now hard-errors (exit 2) when the tree offers more than one
`bs<N>.json` for a batch size, or a single one that is not at the top level, listing every
candidate (`resolve_dump_tree`, tested in `harness/tests/test_dump_tree.py`). The
bs16-conditional cap policy above is justified on PERF grounds (the cap is bit-transparent at
every bs — I9b).

M3-I11 state (evidence: `demo/qwen3_5/accept/opt/m3i11/`). A live run-to-run-varying write in
the decode path is CONFIRMED on hardware: the MoE router's compacted `mpk_active_expert_ids`
list comes out in a different order in 767 of 800 same-input comparisons (set and count
identical in all 800). It is value-neutral at the geometries we run — the grouped GEMM
strides the list and addresses by expert id — and 42 runs / 420 trajectories / 8176
state-level comparisons at 65e42ee8 produced ZERO differences, all matching the census
consensus md5. M3-I5c's `0c8b4cf5` removes that atomicAdd; its hardware validation is
pre-registered and still owed, and I11's E7 numbers are its pre-fix baseline (post-fix must
be 0 order differences and 0 non-ascending lists).

What is NOT closed: the M3-I9b census's own token divergences (6 anomalous dumps of 80) did
not reproduce — P(0 in 42 runs) is about 2e-4 at the census rate — so the >=2-rep rule stays
mandatory for M4's 1024-token gate. Every census anomaly landed on GPU 1/4/7; every clean run
here was on GPU 6, and that is the one untested difference. A second defect found by source
reading is reported but not patched: `linear_sm100_mpk.cuh` ends a task with
`cp.async.bulk.wait_group.read`, which does not cover the TMA store's destination write
before the task's release-increment publishes it to a consumer CTA.
