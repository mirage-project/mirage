# Phase (c) — DEBUG: the gate ladder

Distilled from `FUSED_KERNEL_DEBUG_METHODOLOGY.md` (written after ~9 wasted debug
rounds) + the V2-campaign memory. Work TOP-DOWN; each rung exists because skipping it
burned real days. Read this BEFORE instrumenting anything.

## A. Wrong-output ladder (run in THIS order)

**A1 — Full-layer TOKEN-MATCH first.**
Run chain (baseline) vs suspect (new path/kernel) at the FULL layer count and compare
generated tokens (`--save-tokens`). Correct = token-identical for ~tens of tokens,
then FP non-associativity divergence (expected, not a bug).
- NEVER judge correctness from few-layer output coherence — low-layer output is
  garbage regardless of correctness (a known artifact; "looks degenerate at 4 layers"
  cost 8 rounds chasing a non-bug). Coherence ≠ correctness.
- On a NONDETERMINISTIC path, A1 becomes the C-protocol below.

**A2 — If broken: diff vs the CHAIN, stage-by-stage.**
The chain is ground truth. Instrument BOTH paths to dump the SAME tensor at the SAME
position; the first stage diverging from the chain is the bug.
- Compare at a CLEAN, distinct-token position — chat templates can duplicate tokens at
  pos 0/1 (double-BOS), making pos-0/1 intermediates legitimately identical.
- Dump the FULL output vector. A tap striding `i += gthreads` with gthreads > N summed
  only element 0 and made an unverified final stage look "fine" for rounds.
- Metric: cosine + sum|Δ|, never sum|x| (RMSNorm equalizes magnitudes — equal L1 ≠
  equal vectors). Tiny leading dims are often just a tiny LN weight, not degeneracy.

**A3 — Garbage at TP>1 but gate/TP1 fine ⇒ MISSING CROSS-RANK COLLECTIVE, first
hypothesis.** A single-rank/cooperative gate is one process = TP1 where every
AR/reduce-scatter/EP-dispatch is a no-op — structurally blind. Before fusing/replacing
a chain op, read its SHARD_RULES entry + `world_size>1` branch: if it does a
collective, the replacement must reproduce it, and add residual/bias EXACTLY ONCE
post-AR (zero-residual binding + fold-in-AR). Reading multi-pointer taps under
`mpirun -np N`: N distinct out-pointers at ONE step = the N RANKS, not layers; print
`runtime_config.my_gpu_id`.

**A4 — No blind fixes.** Confirm the mechanism with your OWN in-kernel probe before
shipping. Two past "fixes" were data no-ops (e.g. an output-binding fix — input/output
descs of a root cuda_tensor already alias the same address).

**A5 — Right reference.** Per-task v2 ports are validated bit-exact vs the v1 kernel
on IDENTICAL input bytes (deterministic in isolation); whole-graph checks use the
C-protocol. Never hold a fused mega to the chain's bit pattern (different reduction
order) — hold it to its v1 twin, or to cos/rel_max envelopes vs fp32 torch.

## B. The 6 gate-fidelity classes (a per-kernel gate must match ALL)

1. **COLD-L2** — flush ≥256 MB per timed iter (or let the chain's footprint be the
   flush). Warm gates over-state wins ~2.5×; target the COLD number.
2. **FULL-GRID geometry** — 136 worker-CTAs cooperative + the real thread count
   (256 attn / 512 FFN class), never `<<<1,256>>>`.
3. **PRODUCTION scale/weight FORMAT** — e.g. per-128-block fp32 `weight_scale_inv`
   `[N/128,K/128]` read as plain fp32; no self-invented per-row formats.
4. **RECURRENCE** — stateful kernels (KV cache) need real multi-step decode
   (write@N, read-as-history@N+1), not one pre-filled call.
5. **PER-RANK EP/TP SHARDING + typical active count** — measure at deployed per-rank
   sizes (DSv3 TP8 EP2: 128 local experts, ~4 of top-8 active — not 256/8).
6. **CROSS-RANK COLLECTIVES** — un-testable single-process (A3). Multi-rank micrograph
   (TP2 first) is the ONLY gate for collective ops.

## C. Correctness on a NONDETERMINISTIC path (TP8-class atomicAdd graphs)

DSv3 TP8 EP2 decode is token-level nondeterministic (FFN cross-CTA FP atomicAdd; two
identical runs diverge ~token 10). Token-identity A/B is INCONCLUSIVE there.
- **Control first**: run OFF1 vs OFF2 (identical baselines). If OFF-vs-ON diverges at
  the same point as OFF1-vs-OFF2, the difference is baseline noise.
- **3-part gate for math-changing folds** (all three, in order):
  1. Token-identity CANARY on a deterministic slice (DSv3: `--layers 0-3`, no MoE
     atomicAdd) — OFF/OFF identical AND ON/OFF identical.
  2. **NaN poison-fill negative control**: fill the weight slice the change reads with
     NaN, verify corruption from token 0 (proves the kernel genuinely reads the folded
     input at the asserted offset AND the canary is sensitive). Revert before commit.
  3. Full-model coherence-in-envelope: ON/OFF divergence point inside the OFF/OFF
     spread.
  This proves "genuine lever, no gross corruption" — NOT sub-ULP numeric correctness
  (that needs a per-layer logit-cosine harness; keep the ledger explicitly open).
- **Dead-task claims** ("output unused, safe to remove"): the box token-identity A/B
  in the EXACT deployed env is the ONLY ground truth. Static analysis + reviewer +
  Codex all converged on "dead" once and the box refuted it. ~10 min; always run it.

## D. Hang / crash triage (v2-specific)

> The three historical v2 runtime races are FIXED (2026-07-16: `689dadc5` launcher-ITS early
> page release, `7d271a01`+`7b6ae2bb` consumer-suffix/page-parity alias, `025029a1`
> iteration-barrier half-exit). A hang matching an old signature on a ≥`7b6ae2bb` tree is a
> NEW bug — read those commit messages + `v2-kernel-writing/references/validation-debug.md`
> §5.1 (fingerprint method + durable rules) before instrumenting.

**D1 — iter-0 fine, iter-1 hang ⇒ re-init of PERSISTENT state, not a missing event.**
The signature of re-zeroing monotonic barrier scratch: `count ≥ num_tasks*(iter+1)`
holds at iter 0 only. Check every `tensor_init_layer` on barrier scratch has
`skip_after_step0=True`, and that a runtime port PRESERVED that flag.

**D2 — deadlock right at graph start ⇒ §1.1 bodyless consumer.** A task type with no
v2 consumer body never arrives `SEM_DEP_READY`; the next ring-slot reuse spins forever.
The build-time guard should have caught it — if a hang smells like this, check the
guard actually ran (`use_v2_runtime` truly set) and that no task bypasses registration.

**D3 — fused-mega deadlock at its grid barrier ⇒ two usual suspects:**
(a) the new task type missing from the `task_offset = bid.x` block in `runtime.cc`
(garbage CTA index; v2 megas read union offset 0, v1 megas `merge_task_offset` offset
4); (b) `num_tasks != num_workers` (the Form-2 co-residency contract — host asserts
exist, verify they fired) or a rank early-returning before a team rendezvous.

**D4 — name the hung task with the WATCHDOG, not the breadcrumb.** The breadcrumb is
CRASH-only (can't dump on a live hang); a watchdog dump (per-SM current
task/iteration) named the wedged consumer directly. Do not re-run a hang without it.

**D5 — breadcrumb/in-flight COUNTS are base-rate artifacts.** The dominant in-flight
task at fault time is usually the terminal CONSUMER (argmax-class), not the faulter.
For deterministic illegal-address faults, `compute-sanitizer --tool memcheck` is
ground truth (it named the real faulting line when counts pointed elsewhere).

**D6 — two-build trap.** Before theorizing across artifacts from different runs:
`git status --porcelain <file>` + `git grep HEAD <symbol>` — attribute each artifact
to its exact binary. Pre-fix and post-fix artifacts mixed look like contradictions.

**D7 — box-level noise.** Xid 145/45 at the very first barrier = reboot the box, not
code. `cudaErrorMisalignedAddress` on a NEW task = check `extern __shared__` region
`alignment=1024` in the `_spec.h` (smaller aligns misalign OTHER tasks' TMA/AR in the
shared test.cu; only an in-MPK run catches it — hence the post-port smoke).

**D8 — after ANY hang/crash: clean up, verify no D-state zombies, STOP.** Never
crash-loop the megakernel; a wedged box needs reboot, not retries.

## E. Roster + when to invoke

- `mpk-correctness-gate` — before trusting any baseline; before every math-changing
  commit. Runs test-mode + routed-path NON-NULL check (num_active≈4 class; a lean/
  null-MoE config invalidates every downstream number) + token-identity/numeric +
  cross-model regression smoke.
- `ablation-logic-reviewer` + Codex MCP (default params) — MANDATORY on every
  non-trivial conclusion (root cause, "structurally impossible", lever verdicts)
  BEFORE acting on it. Over-claims have repeatedly been wrong here.
- `mpk-profiler` — perf ground truth only AFTER correctness is green.
- Validation assets to reuse: `tests/runtime_python/blackwell_v2/` (per-op harness; README
  documents metrics + trust gates — its "profiled-sparse-chain wedge" caveat is HISTORICAL:
  those wedges were the v2 races fixed 2026-07-16, see the note atop §D), per-kernel
  `tests/runtime_python/blackwell/sm100_*/` test-mode dirs, TP2 micrograph
  pattern for collectives.

## F. Checklist to close Phase (c) for a milestone

- [ ] Per-op: harness PASS (cos ≥ 0.999, rel_max ≤ 3e-2, no NaN; bit-exact where owed)
- [ ] Collectives: TP2 micrograph bit-exact on all ranks, then TP8
- [ ] Fused megas: bit-match vs v1 twin + in-MPK small-slice smoke (align/oob class)
- [ ] E2E slice: A1 (or C-protocol) PASS; §1.1 guard green; reachability diff clean
- [ ] No open hang: any observed hang root-caused via D1-D7 (never parked)
- [ ] `mpk-correctness-gate` PASS recorded; conclusions reviewer+Codex-checked
