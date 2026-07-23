---
name: mpk-development-norms
description: The MPK team's "where does a change belong + what a clean PR looks like" norms, extracted from mirage-project/mpk merged-PR history. Read FIRST — before starting any MPK change, opening/shaping a PR, deciding which file a change goes in, reviewing a diff's shape, or cleaning up a branch that grew messy/off-norm ("改得太乱/不符合开发规范"). Complements add-mpk-model / add-mpk-task / v2-model-support (the HOW) with the WHERE + the PR-shape gate.
---

# MPK development norms — right place, minimal surface, clean PR

These are the *change-shape* norms the maintainers actually enforce, reverse-engineered from
merged PRs on `mirage-project/mpk` (see `references/exemplar-prs.md` for the cited commits and
per-category file-touch tables). The sibling skills tell you HOW to add a task/model/kernel;
this one tells you WHERE the change belongs and what a landable PR looks like. When in doubt,
find the closest recent merged PR of the same category and mirror its footprint.

The rule underneath all of them: **put a change in the file that OWNS that concern, at the
smallest generic surface — not in the file that is convenient to reach from where you already
are.** A diff that sprawls into shared runtime/python files to serve one model is the smell
this skill exists to prevent.

## 1. Where does my change belong? (ownership map)

| Change | Lives in | Must NOT touch |
|---|---|---|
| New GPU op / kernel | `include/mirage/persistent_kernel/tasks/<arch>/<op>.cuh` + its `tests/runtime_python/.../sm100_<op>/` unit test (+ `runtime_kernel_wrapper`) | `multigpu.py`; a model's builder |
| Wiring a new task type into the runtime | C++ registration only: `runtime_header.h` (enum) → `src/kernel/{task_register,graph,runtime}.cc` → `tma.cuh` if TMA — coherently, all-or-nothing | — |
| A **generic** Python op-API for that task | one `<operation>_layer` method in `python/mirage/mpk/persistent_kernel.py`, **named by the operation/algorithm, never by the model** (`moe_w13_linear_layer`, `splitk_linear_layer` — not `qwen3_*`/`deepseek_*`) | — |
| Model bring-up | `python/mirage/mpk/models/<model>/builder.py` (topology, TP/EP shard rules, layer composition) + `demo/<model>/` (demo.py, HF reference, shard loader) | `persistent_kernel.py` beyond generic ops; `persistent_kernel.cuh`; `multigpu.py` |
| Runtime / scheduler change | `persistent_kernel.cuh` / `runtime_header.h` / `src/kernel/runtime.cc` — as its **own** PR | a model dir; unrelated kernels |
| Multi-GPU / collectives | `python/mirage/mpk/multigpu.py` — allreduce-runtime-owned (historically one owner PR) | model builders |

Model composition is data, not shared code: the *order and choice* of layers, the shard-rule
regexes, and the weight-name mapping are all model-specific and belong in
`models/<model>/builder.py` + `demo/<model>/`. Only a genuinely reusable **operation** earns a
method in `persistent_kernel.py`, and it is named for the operation.

## 2. The norms (what a reviewer checks)

1. **Right place, not convenient place.** Use the ownership map above. Ownership ≠ exclusivity:
   a *new task type* legitimately spans `runtime_header.h` + `src/kernel` + wrapper + `tma.cuh`
   (that IS its home); a runtime fix may touch a task `.cuh` when the invariant crosses the
   worker/task boundary. What's off-norm is reaching into a shared file to serve one model.
2. **Minimal shared-surface diff.** A model-support PR does **not** touch `persistent_kernel.cuh`
   or `multigpu.py`, and touches `persistent_kernel.py` only to add/fix a **generic**
   operation-level primitive. Shared APIs are named by operation/algorithm, never `<model>_*`.
   (Counter-smell this catches: `deepseek_mla_rope_q_layer`, `mla_kv_gather_unified_layer`,
   `dsv3_router_gate_gemv_layer` added to the shared file — those belong behind a generic API
   called from the model builder.)
3. **No experiment env-vars in landed code.** In `persistent_kernel.py` the only `os.environ`
   uses are the **5 build-path/infra vars** (`MIRAGE_HOME`, `NVSHMEM_INC_PATH`,
   `NVSHMEM_LIB_PATH`, `MPI_INC_PATH`, `MPI_LIB_PATH`). No `MPK_*_DBG` / `*_PROBE` / `*_GUARD` /
   `FASTFWD` / campaign perf toggles survive into a merged PR — a perf lever is either
   **hard-wired to its chosen production value** (as a named constant) or absent, and a
   debug/diagnostic knob is deleted with its code path. (This norm is about landed code; an
   *in-flight exploration* branch keeps levers env-gated default-OFF — see `mpk-lever-cleanup`
   for the collapse step.) Audit isn't limited to `persistent_kernel.py`: check the builder,
   `runtime.cc`, and C++ `getenv` debug hooks for the same residue.
4. **Runtime changes are separate, coherent PRs** — not bundled inside a model or kernel PR.
   `#411` (Split persistent kernel) touched exactly 2 files. If your model work needs a runtime
   fix, split it into its own PR so the maintainer can take/defer it independently.
5. **One PR = one coherent topic.** PRs are squash-merged (one commit each). A focused bugfix is
   often a single file (`#719`). Don't fold a de-cruft, a perf lever, and a new kernel into one
   diff. Commit granularity mirrors this even pre-squash: each commit is one reviewable idea.
6. **Comments are sparse and functional.** ~5% comment lines in kernels, ~10% in a builder —
   they say *what a non-obvious line does*, never a perf-campaign diary, a "we tried X" history,
   or narration of the obvious. No commented-out code.
7. **Tests are part of the change shape.** A new kernel/task ships its
   `tests/runtime_python/.../test_*_testmode.py` (usually + a `pytorch_reference.py` and, if
   needed, a wrapper/`setup.py`). A PR that adds a kernel with no test is off-norm.
8. **Registration/ABI is coherent.** A new task ID updates `runtime_header.h`, the `src/kernel`
   registration, the wrapper, and TMA/runtime glue **together** — never a dangling enum with no
   register/graph handler, never a handler for a deleted enum. (Fail-loud: rebuild after any
   enum edit — a stale enum silently mis-dispatches.)
9. **Format + no artifacts.** Run `bash scripts/format.sh` (clang-format-15, CI-enforced) before
   pushing. Never stage generated/local material: `scratch/`, `outputs/`, `_results/`, weight
   caches, generated `test.cu`/`.so`, perf logs, `PR_DESCRIPTION`/campaign notes, `.claude/`
   (except the sanctioned `.claude/skills/**` + `.claude/agents/**` on a skills PR).

## 3. PR-shape checklist (run before you open/push)

- [ ] Every changed file is the **owner** of its concern (ownership map). No shared-file reach
      for a single-model need.
- [ ] `git diff --stat <merge-base>..HEAD` — is the shared-surface footprint
      (`persistent_kernel.cuh`, `persistent_kernel.py`, `multigpu.py`, `runtime.cc`) as **small
      and generic** as the closest exemplar PR? Any `<model>_*` method in `persistent_kernel.py`?
- [ ] Env-var count in `persistent_kernel.py` back to the 5 build-path vars (no campaign
      toggles anywhere in the diff)?
- [ ] Runtime/scheduler changes split into their own commit/PR?
- [ ] Each commit one coherent topic; message states mechanism + (for perf) measured Δ; ends
      with the required `Co-Authored-By` line?
- [ ] New kernels/tasks carry their test-mode test + reference?
- [ ] Registration coherent (enum ⇄ register ⇄ graph ⇄ wrapper), rebuilt clean?
- [ ] `scripts/format.sh` clean; no generated/local artifacts staged; sensitive-grep before push?

## References
- `references/exemplar-prs.md` — the cited merged PRs per category, with their file-touch tables
  (the empirical basis for every claim above). Mirror the closest one.
