---
name: mpk-lever-cleanup
description: >-
  Use when a batch of env-gated (`#ifdef MPK_DSV3_*` / `os.environ`-controlled,
  default-OFF) MPK optimization levers needs to be consolidated into a single
  clean code path for a PR: hard-wire every winning lever as the default, delete
  the legacy `#else` branches, remove the env vars that select new-vs-old logic,
  revert dead levers that measured KILL/NULL/regress, delete diagnostic probes,
  then commit one clean version. Applies to the wrap-up stage where the
  optimization work has settled and is being merged to mainline. Not for the
  exploration phase (levers should stay env-gated default-OFF there) or for
  runtime/execution-model changes.
tags: ["mpk", "cleanup", "refactor", "pr"]
related_skills: ["mpk-internals", "add-mpk-task"]
version: "1.0.0"
---

# MPK Lever Cleanup — consolidate env-gated optimizations into one clean path

During exploration, every MPK performance optimization is an **env-gated,
default-OFF** lever (`#if MPK_DSV3_XXX` + the `#else` legacy path +
`os.environ.get(...)` → `-D` injection in persistent_kernel.py). Once the work
has settled and is headed for mainline, the levers must be consolidated into a
**single path**: hard-wire the winners as the default, delete the legacy code,
remove the control variables. This skill is the complete procedure — plus the
pitfalls actually hit — from that wrap-up refactor.

> Core mindset: this is a refactor that **intentionally changes the default
> build** — the default path flips from "the safe legacy logic" to "the
> optimized path". The usual "default build must stay byte-identical" commit
> gate is therefore **deliberately waived** here; that is the whole point of
> this commit.

## Procedure (7 steps, in order)

### 1. Enumerate all gates
```bash
grep -rhoE "MPK_(DSV3_)?[A-Z0-9_]+" <the megakernel .cuh files> <builder.py> <persistent_kernel.py> \
  | sort -u | grep -vE "MPK_(MAX|PAGE|PROFILING|NUM)"
```
List both the `#ifdef` gates in the `.cuh` files **and** the `os.environ`/`-D`
injections in persistent_kernel.py.

### 2. Classify every gate (VERIFY each one — never assume default-OFF)
| Class | Action | How to decide |
|---|---|---|
| **WIN** | Hard-wire ON: remove the gate + delete the `#else` legacy + remove the env injection; **keep the geometry guard** (TP8/mbt/workers) | Lever already committed in git log + a WIN row in experiment_history |
| **DEAD** | Fully revert (delete all of its code; **roll back any ABI it changed**) | A KILL/NULL/REGRESS row in experiment_history |
| **DIAGNOSTIC** | Delete (probe/no-op/poison/xor — not a lever) | Name contains PROBE/NOOP/POISON/XOR; used only for measurement |
| **ALREADY-ON** | Confirm it is still present; keep it as the unconditional default (do not add a gate) | grep persistent_kernel.py: is it already injected via `-D` unconditionally? |
| **LEAVE-UNTOUCHED** | Do not touch | A fallback outside this path (e.g. the TP<8 ROUTER_GEMV), inert, or a generic non-DSv3 flag |

**⚠️ Classification must be verified, never recalled from memory.** Pitfalls
actually hit: `TOPK_PARALLEL`, assumed default-OFF and slated for revert, is in
fact **ON in every decode build** (part of the current winning stack — KEEP);
`ROUTER_GEMV`, assumed obsolete, is in fact **the router of the TP<8 fallback**
(must not be deleted). Use `grep -n` on the injection condition in
persistent_kernel.py and check whether the gate is actually exercised at the
production geometry.

### 3. Codex-vet the classification + refactor plan
Hand the gate inventory + classification + goal to Codex (`mcp__codex__codex`)
for a multi-round discussion: validate the classification, agree on a safe
execution order, the correctness risks, how to verify the ABI-revert, and the
structural-vs-leaf distinction. Codex will catch conflicts in the
classification (see the step-2 pitfalls).

### 4. Freeze the reference (the correctness comparison baseline)
Before changing anything, run the **current winning stack** (all levers ON) and
record its output: e2e tpot + logits/prose (run it twice to get the A/A
nondeterminism envelope). The final "clean default build" is compared against
**this winning-stack reference**, not against the old safe default.

### 5. Execute in the safe order (build-check after every step)
1. **First revert the dead levers that changed the ABI** (most dangerous; do it
   in isolation and verify). If a dead lever touched the
   `(num_in,num_out,TASK_ENUM,variant)` tuple in graph.cc/task_register or
   added a tensor, a single ABI mismatch = "Invalid global read" at runtime. If
   those changes were **never committed**, simply
   `git checkout HEAD -- <producer files>` to return to a clean ABI, then:
   ```bash
   git diff HEAD -- graph.cc task_register.cc tasks.py multigpu.py allreduce.cuh | wc -l   # expect ≈ 0
   grep -rn "tile_sumsq|<sidecar tokens>|input_ptrs\[N\]" <files>                          # expect 0
   ```
   **Delete the dead lever's consumer half at the same time** (otherwise it is
   a stale-env out-of-bounds landmine).
2. **Delete the remaining dead levers + all diagnostic probes** → build-check.
3. **Hard-wire the LEAF wins** (leaf optimizations with a clean `#else`):
   remove the gate, delete the `#else`, keep the geometry guard → build-check.
4. **Hard-wire the STRUCTURAL wins last** (path-selectors that change the graph
   shape / large control flow): delete the entire alternate path, keep the
   TP8/mbt/workers guard → build-check + a `--layers 0-3` in-MPK smoke.
   (Structural gates are more dangerous than leaves — removing one deletes a
   whole alternate code path. Do it after the tree has already shrunk.)

### 6. Verify correctness (the default path's math has changed)
- **Token-identity cannot be used** (DSv3 TP8 decode is FP-nondeterministic —
  cross-CTA atomicAdd).
- Use instead: the **A/A envelope** (winning stack compared against itself) +
  clean default vs the winning-stack reference with the **per-step
  logit-cosine inside the envelope** + stable top-k overlap + 512 tokens of
  coherent prose + no NaN/Inf.
- **Perf smoke**: e2e tpot should ≈ the winning-stack reference (confirms no
  win was silently dropped).
- **TP8 JIT smoke**: `--layers 0-3` confirms the hard-wired megakernel really
  instantiates and runs (the `#else`-deleted path only instantiates at
  world_size==8).
- Qwen3 / TP4 regression smokes protect the untouched fallback / non-DSv3
  paths.

### 7. Commit one clean version (for the PR)
- **Stage source files only** (kernels/.cuh, builder.py, persistent_kernel.py,
  task_register.cc); **exclude** `.claude/`, `scratch/`, `experiment_history/`,
  CSVs/outputs/`.pk_compile` and other local artifacts.
- Run `mpk-commit-reviewer` and **tell it explicitly that the default-build
  change is intentional** (otherwise it will BLOCK per the standard gate); it
  still checks staged-path hygiene, the allowed surface, the message, and the
  correctness story.
- The commit message lists everything: which levers were hard-wired (+ each
  one's Δ), which dead levers were reverted, which diagnostics were deleted,
  that the ABI was restored, the verification evidence, and **an explicit
  pre-merge gate** (if TP8 runtime validation was blocked by box capacity,
  write it into the message as a must-run item before merging). Include
  `Co-Authored-By`.

## Key pitfalls (all hit in practice)
- **The default build is intentionally NOT byte-identical** — that is the goal,
  not a bug; waive that one commit gate.
- **The ABI-revert is the most dangerous step** — do it first and in
  isolation, grep it clean, diff against the last clean commit, and delete the
  consumer half together with it.
- **Structural wins go last** — a path-selector is more dangerous than a
  leaf-opt (removing it deletes an entire alternate path).
- **Classification must be verified** — some gates are already unconditionally
  ON, some are fallbacks; never assume "default-OFF".
- **Orphaned legacy functions** — after deleting the `#else` call site, the
  `__device__` function definition may remain (nvcc elides it; harmless but
  unclean); either delete it or flag it as a known nit in the PR.
- **The correctness gate = A/A envelope + coherence**, not token-identity (the
  path is FP-nondeterministic).
- **The TP8 runtime gate may be blocked by box capacity** — a commit meant for
  PR review may land with a documented pre-merge gate (a PR is review, not
  auto-merge); never fabricate numbers.
- **Sub-agents can contradict each other about the same fact** (e.g. "the
  lever was deleted" vs "the lever was hard-wired") — verify yourself with
  `grep -c <the win's body symbol>` that the win's body is still present
  (zero refs to the macro ≠ the win was deleted; possibly only the gate was
  removed and the body became unconditional).
