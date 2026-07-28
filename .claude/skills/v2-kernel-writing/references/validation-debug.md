# Runtime-V2 validation + debug playbook

Order matters: test-mode numeric → §1.1/protocol audit → in-MPK probe → MULTI-STEP → perf.
Never skip a rung because the previous one "looked clean" — every trap below passed at least
one earlier rung before detonating (see FUSED_KERNEL_DEBUG_METHODOLOGY.md for the v1 ancestry).

## 1. Test-mode numeric gate (first rung, CPU→1-GPU)

Harness: `tests/runtime_python/blackwell_v2/` — everything runs the REAL v2 pipeline
(`PersistentKernel(use_v2_runtime=True)` → registration → v2 queue/smem planning → JIT nvcc →
`launch_v2_func`). Files: `v2_harness.py` (case builders), `case_runner.py` (one case per
subprocess — hang/crash isolated), `run_suite.py` (matrix, GPU pinning, timeouts),
`pytorch_reference.py` (fp32 refs, `gen_tensor(name)` seeds = sha1(name) so v1/v2 subprocesses
see bit-identical inputs). DSv3-specific: `dsv3_attn_harness.py`/`dsv3_ffn_harness.py` +
`run_attn_suite.py`/`run_ffn_suite.py`; bit-match harnesses `attn_mega_v2_bitmatch.py`.

- **PASS bar**: `cos ≥ 0.999 ∧ rel_max ≤ 3e-2 ∧ no NaN` vs torch fp32; plus the v1-counterpart
  compare (bit-exact expected for elementwise; NOT for linears — different accumulation order).
- Add a case for the new op in the correctness matrix; keep linear-family cases at M ≤ 16
  (the framework's own M=128 catch: cos=√(1/8), bitexact=16/128 — the v2 linear computes ONE
  16-row tile).
- LIMIT: test-mode is npes=1 — it CANNOT catch a missing cross-rank collective
  (`feedback_single_rank_gate_misses_tp_collective`). "Garbage at TP>1, gate fine" ⇒ suspect a
  missing/wrong AllReduce FIRST.

## 2. Protocol audit (static, before first run)

- §1.1: consumer body starts with `consumer_dep_prefix` (grep the registration).
- Every async-arrived mbar has a start-of-task re-init owned by its arriving role
  (house-style §2 loader/launcher re-inits).
- Every declared page is arrived EXACTLY once per task on EVERY path (bounds-fail included).
- No `__syncthreads` in role bodies; named-barrier ids don't collide (1 linear, 2 rmsnorm,
  3 ffn consumer_sync, 6 attn) — a new 128T consumer-only body picks a free id.
- `extern __shared__` is `__align__(1024)`. If in doubt, MEASURE the base
  (`__cvta_generic_to_shared(smem_ptr)`) — static analysis once "refuted" this footgun and was
  wrong: driver base measured 4304 (sub-128) in the crashing build vs 5120 clean
  (`feedback_extern_smem_align_megakernel_convention`). A PTX grep
  `.extern .shared .align <128` over the generated test.cu is a cheap CI-grade check.

## 3. In-MPK probe (the only real verification for an MPK task)

`--layers 0-3` demo run with the op wired (isolated harnesses + small multi-task builds ALL
passed while the full megakernel crashed — the align footgun is multi-task-instantiation-
dependent). Then the full-op-matrix v2 test-mode graph under local memcheck as the boxless repro.

## 4. MULTI-STEP gate — iter ≥ 1 is MANDATORY

A single-token run proves nothing about persistent state. Run ≥ 3 decode steps.

**Signature: iter-0 fine / iter-1 hang ⇒ persistent-state re-init, NOT a missing event.**
Discriminator: `num_tasks*1 ≥ num_tasks*(it+1)` iff `it ≤ 0` — a missing trigger would hang
iter 0 too. Root-cause class: a monotonic grid-barrier counter (`need=num_tasks*(iter+1)`,
never reset) being re-zeroed each step because `skip_after_step0` was dropped from its
tensor_init (M3, 2026-07-07: attn task 353 + ffn task 348;
`feedback_monotonic_barrier_skip_after_step0_port_trap`). Fix contract in wiring-recipe.md §8.

## 5. Hang vs crash tooling

| Symptom | Tool | Notes |
|---|---|---|
| HANG (0 tokens, SMs pegged, log frozen) | **watchdog**: compile `-DMPK_V2_BREADCRUMB` + env `MPK_V2_HANG_WATCHDOG_S=<sec>` | Host thread dumps the pinned breadcrumb + `_Exit(134)` on no-progress (persistent_kernel_v2.cuh:~705-728). PROVEN: named TASK_ATTN_BLOCK_MEGAKERNEL_V2(353)/consumer/iter-1. Without it `cudaStreamSynchronize` blocks forever — breadcrumb alone is CRASH-only (dump only runs when sync RETURNS an error, :698-706) |
| HANG matching a HISTORICAL signature — profiled-only mlp-chain wedge, unprofiled correctness-matrix iter-0 wedge (tasks 242/243), or end-of-run iter-31 wedge | **RESOLVED 2026-07-16 — all three were v2 runtime RACES, root-caused + fixed (§5.1)**: `689dadc5` / `7d271a01`+`7b6ae2bb` / `025029a1`. Former wedge windows PASS post-fix (mlp L=6/iters=32 profiled, L=4; the profiled correctness matrix) | The 2026-07-15 era's "profiler×tcgen05-wait codegen Heisenbug" framing and its "FIX ATTEMPT REFUTED / profiled arm SUSPENDED / root cause OPEN" status are **SUPERSEDED** — the branchless-wrap fix attempt was refuted because the wrap was never the mechanism; the profiled BIAS was race 3's forced-iteration exposure (profiled builds pin `g_v2_gen_done=0`, making the racy exit read the sole loop exit). The reference-kernel-profiled discriminator (run the candidate-free reference chain at the same L/iters, watchdog armed) remains the right FIRST move on any new profiled-only hang — but on a tree ≥ `7b6ae2bb` such a hang is a NEW bug: read the four race commits' messages first, then apply the §5.1 fingerprint method. Budget ≤3 wedge repros, short timeouts, never re-run unchanged |
| CRASH (cudaErrorIllegalAddress / Misaligned) | **compute-sanitizer memcheck under mpirun = GROUND TRUTH** | Exact instruction+block+line. The breadcrumb IN-FLIGHT count is a BASE-RATE ARTIFACT — dominant count = widest terminal consumer (argmax), not the faulter (linear_v3 loader, 49/49 records, workers 70-131). Never derive culpability from count dominance; SKIP-N probes keyed on breadcrumb false-confirm bystanders (`feedback_breadcrumb_inflight_base_rate_artifact`) |
| Post-fix ablation | sanitizer records → 0 | Crash-elimination alone can be a shifted tile, not a fix |
| Xid 145/45 at first NVSHMEM barrier | box stop→start reboot | Fabric fault, NOT code |

Breadcrumb wiring: readback helper `demo/deepseek_v3/v2_breadcrumb_readback.py`; role ids must
match `MPK_V2_BREADCRUMB_ROLES` (runtime_v2.cuh:756-763); STARTED≠COMPLETED = in-flight
candidate SET, not proof.

### 5.1 The three fixed v2 runtime races (2026-07-16) — mechanisms, method, durable rules

All of §5's historical wedge signatures were these; each is FIXED, kept here for the lesson:

- **Race 1 (`689dadc5`) — launcher ITS early page release.** The task-end blanket page
  release ran with no warp reconvergence after the `if (elect_sync()) { MMA loop }` block;
  under sm_100a Independent Thread Scheduling lanes 1..13 released pages while elected lane 0
  was still inside the loop, so a release beat the SAME task's loader-prefix claim (parity one
  use ahead → all-role wedge). Fix: `__syncwarp()` between the elect block and the release.
- **Race 2 (`7d271a01`, residual closed by `7b6ae2bb`) — consumer suffix vs loader claim.**
  The codegen consumer page-release SUFFIX could overtake the same task's lagging-loader
  prefix CLAIM (different warps, no ordering edge). Fix: Design E consumer-owned claim
  (program-ordered claim→body→release; opt-in rmsnorm/silu). Residual: Design E's SkipUsed
  loader made page observation SPARSE — a mod-2 parity wait cannot distinguish 0 releases
  from 2, so the FFN GEMV chain aliased two-early. Fix: consumer-TOTAL page lifecycle for the
  FFN chain + a plan-time page-window assertion (which caught a real qwen3 plan shape);
  nwarps=7 forms proven structurally excluded.
- **Race 3 (`025029a1`) — iteration-barrier half-exit.** The loop-exit `config.step[0]` read
  sat AFTER the end-of-iteration barrier, racing worker 0's next-iteration prepare; straggler
  workers TERMINATEd one iteration early while everyone else waited on their never-published
  tasks. This was the "profiled bias": profiled builds force iterations, making that read the
  sole loop exit. Fix: snapshot exit reads BEFORE barrier arrival.

**The fingerprint method that cracked all three (reuse on any NEW wedge):** zero-perturbation
pinned-memory state dump on the live wedge → match role positions + page-parity arithmetic +
raw mbar words against the plan's ground-truth tables with ZERO free parameters (+ wait-site
markers); for timing races, a reviewer-required amplifier arm (injected delay) that reproduces
the wedge on demand with a population-scaled signature.

**Durable protocol rules (also in house-style.md §3):** (1) arriver-set == waiter-set —
reconverge (`__syncwarp()`) before any lane-parallel mbar arrive after a divergent block;
(2) dense-observer-or-full-owner for mod-2 parity mbars — sparse observation aliases;
(3) exit-reads snapshot BEFORE barrier arrival; (4) protocol invariants become plan-time
assertions (`build_v2_plan` aborts loudly beat silent wedges).

## 6. Two-build trap (before ANY hypothesis from box artifacts)

Two artifacts from DIFFERENT binaries are not a contradiction. Before theorizing:
`git grep HEAD <symbol>` (is the fix in HEAD?) + `git status --porcelain <file>` (working tree ≠
HEAD?) → attribute EACH artifact to its binary. The linear_v3 W-TMA case nearly shipped a wrong
descriptor-OOB hypothesis + GEMV swap because sanitizer ran pre-fix and the arg-dump ran
post-fix (`feedback_two_build_trap_compare_binaries_before_hypothesis`).

## 7. Math-changing changes on the nondeterministic TP8 path

Token-identity is INCONCLUSIVE at TP8 EP2 (FFN cross-CTA FP atomicAdd; runs diverge ~token 10,
`feedback_dsv3_tp8_fp_nondeterministic`). Gates that work:
- **Canary token-identity at 0-3 layers** (deterministic there).
- **NaN poison-fill** for skip-re-init levers: fill the no-longer-zeroed region with 0x7FC07FC0,
  require clean in-distribution output. **PRESERVE the self-maintained barrier head bytes**
  (e.g. first 8 bytes attn counter) — poisoning them = deterministic hang = false positive.
  Judge DISTRIBUTIONALLY (ascii_ratio, unique-token count, same divergence points as
  baseline-vs-baseline), plus a cast audit that no scratch byte can form an address
  (`feedback_skip_lever_poison_fill_gate`).
- Full-61L coherence-in-envelope, NOT full-61L token-identity.
- **Same-binary probe-pair** when a reviewer flags binary-mixing: AP→BP inside one binary; agree
  on sign+magnitude with the cross-binary A/B ⇒ confound dissolved.

## 8. Perf measurement (TIER hierarchy — the verdict rules)

- **TIER 1 (sole verdict-grade)**: in-MPK %globaltimer probe / per-position slowCTA at the
  production grid (136 workers, cold weights, co-resident tasks).
- **TIER 2**: faithful per-task harness slowCTA @ grid=136 (prototyping; corroborates TIER 1).
- **TIER 3 (diagnostic only, never promote/reject)**: cudaEvent-WALL (+~8µs envelope),
  standalone-warm. The 25.6µs qkv_a cudaEvent number minted a bogus "4× gap".
- %globaltimer only for 5-40µs bodies; min/median over ≥ n=3; cold-L2 target.

## 9. v2 profiler → perfetto (the profiling tool)

- Build with `-DMPK_ENABLE_PROFILING`; **profiler buffer MUST be
  `V2_PROF_BUF_ENTRIES = 120000*128` (15.36M) entries** — demo.py pins
  `V2_PROFILER_BUFFER_ENTRIES = 120000 * 128` (demo.py:29-30) to match runtime_v2.cuh:58; a
  mismatched buffer corrupts the reserved tail (cursors/spin/suffix/trigger-ring).
- 8 tracks/SM: consumer/loader/launcher/storer/controller + consumer/loader/launcher PHASE
  tracks (dep-wait, page-wait, timed waits > 2µs) — runtime_v2.cuh:27-58. Only the last
  `V2_PROF_WINDOW_ITERS=25` steps are recorded.
- Text tables: `mirage.mpk.prof` (`cmd_summary`/`cmd_check` — check reports MISC drop counts;
  must be 0).
- Perfetto: `python scripts/v2_perfetto_export.py <buffer.npy> <out.json> [--last-steps K]
  [--sm N]` (the v1 exporter parses this buffer as garbage; full window OOMs ui.perfetto.dev —
  always cut). Deeper analysis: `scripts/perfetto_analyze.py` / `scripts/perfetto_depgraph.py`.
- Grid > `V2_PROF_SM_SLOTS=256` workers would corrupt the tail arrays — launch aborts loudly
  (runtime_v2.cuh:1328-1335); don't "fix" that by shrinking the buffer.

## 10. GPU safety (non-negotiable)

Never crash-loop the megakernel (D-state zombies). Local cards: torch-probe before use (broken
cards throw cudaErrorDevicesUnavailable). Box runs: lease + watchdog + trap-cleanup + timeout
wrapper (the `scratch/run_dsv3_mpk_remote.sh` pattern — that script is machine-local/
git-ignored, absent from a clone; the pattern = memory-cap via systemd-run, timeout, trap
that kills the job tree, D-state check after); verify-stopped after box use. Test-mode first,
always.
