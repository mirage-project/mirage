# MPK change-shape review checklist

Review a diff against the norms below. This file is self-contained: paste it into a review
tool (e.g. `codex exec`) together with the diff and the instruction *"review this diff against
this checklist; for each item cite file:line and give KEEP / CHANGE / REMOVE with a one-line
reason."* The repo is Mirage Persistent Kernel (MPK): a compiler + runtime that fuses an LLM
inference pass into one persistent CUDA megakernel. `mpk` is the integration branch changes land
on; diffs are taken against the PR's merge-base.

Work through each section item by item. Be adversarial and specific; default to the leaner diff.
Most items are judged from the diff alone; items that say "check the tree" / "rebuild" / "run the
formatter" need repo + tool access (which `codex exec` has) — treat them best-effort if you were
handed only a diff.

## 1. Change lands in the file that OWNS the concern

| Change | Belongs in | Must NOT touch |
|---|---|---|
| New GPU op / kernel | `include/mirage/persistent_kernel/tasks/<arch>/<op>.cuh` + `tests/runtime_python/.../sm100_<op>/` | `multigpu.py`; a model builder |
| Wiring a new task type | C++ only: `runtime_header.h` enum → `src/kernel/{task_register,graph,runtime}.cc` → `tma.cuh` if TMA — all-or-nothing | — |
| Generic Python op-API | ONE `<operation>_layer` in `python/mirage/mpk/persistent_kernel.py`, named by the operation/algorithm, **never** by the model | — |
| Model bring-up | `python/mirage/mpk/models/<model>/builder.py` + `demo/<model>/` | `persistent_kernel.py` beyond generic ops; `persistent_kernel.cuh`; `multigpu.py` |
| Runtime / scheduler change | `persistent_kernel.cuh` / `runtime_header.h` / `src/kernel/runtime.cc` — as its OWN change | a model dir; unrelated kernels |
| Multi-GPU / collectives | `python/mirage/mpk/multigpu.py` | model builders |

- [ ] Every changed file owns its concern. No shared file (`persistent_kernel.cuh/.py`,
      `multigpu.py`, `runtime.cc`) reached into to serve one model.
- [ ] No `<model>_*`-named method in `persistent_kernel.py` (e.g. `deepseek_mla_rope_layer`,
      `dsv3_router_gate_gemv_layer`). Shared APIs are named by operation, not model.
- [ ] Runtime/scheduler changes are a separate, coherent change — not bundled into a model or
      kernel diff.

## 2. Comment density — sparse and functional

- [ ] Comments say *what a non-obvious line does*; they are NOT a perf-campaign diary, a
      "we tried X / I attributed the regression to Y" history, dated log entries
      (`C20 (2026-05-17): ...`), box-verification stories (`verified on a clean B200, n=3/3`),
      or narration of the obvious.
- [ ] No commented-out code (dead `// printf(...)`, `// cudaMemcpy(...)`, old alternatives).
- [ ] Rough density ceiling: ~5% comment lines in a kernel, ~10% in a builder. A 30-line
      narrative block on one statement is the smell.
- [ ] A comment that documents a real, non-obvious constraint (a hardware ordering requirement,
      a layout contract, a footgun) is KEPT — trim the campaign framing, keep the "why".

## 3. No experiment / debug env-vars in landed code

- [ ] In `persistent_kernel.py` the ONLY `os.environ` uses are build-path/infra vars
      (`MIRAGE_HOME`, `NVSHMEM_INC_PATH`, `NVSHMEM_LIB_PATH`, `MPI_INC_PATH`, `MPI_LIB_PATH`).
- [ ] No `MPK_*_DBG` / `*_PROBE` / `*_GUARD` / `FASTFWD` / campaign perf toggles survive. A perf
      lever is either **hard-wired to its chosen production value** (a named constant) or absent;
      a debug/diagnostic knob is deleted together with its code path.
- [ ] Audit is not limited to `persistent_kernel.py`: check the model builder, `runtime.cc`, and
      C++ `getenv` debug hooks for the same residue.
- [ ] Diagnostic-only logging added for a past debug session (`fprintf(stderr, "[SOMETHING] ...")`
      that only prints and changes no behavior) is removed.

## 4. No gratuitous assertions / error-throwing

For EACH added `assert` / `raise` / `throw` / `abort` / fail-loud check, apply this test:

1. Did upstream (the merge-base) have it? If yes, leave it.
2. Is it necessary? Does *omitting* it have a **correctness** consequence — a silently-wrong
   result — or merely a later *natural* error (a `KeyError`, a dtype/shape error, the next CUDA
   call's error code)?
3. If omitting it only defers to a natural error, or if it restates a condition already checked
   upstream on the path that reaches it → **REMOVE it.** Default to removal.

- [ ] No config guard that merely **restates a predicate the caller already checked** (e.g. a
      task body re-asserting `world_size==8 and num_workers==136` that its selection predicate
      already gated). Delete; if the caller's predicate is *incomplete*, fix the predicate
      instead of duplicating it as a body assert.
- [ ] No check that only pretty-prints an error the very next line raises anyway (a missing-key
      `raise` immediately followed by the indexing that would `KeyError`).
- [ ] No new host launch/return-code `abort` wrapper where upstream launched bare — a failed
      launch surfaces on a subsequent CUDA call regardless, so the wrapper is house-style surface,
      not a correctness guard (dropping it changes no valid-run behaviour).
- [ ] No assertion that can fire on a **valid** state. (A real failure this catches:
      `assert(params.size()==0||3)` guarding a reader written as
      `process_dim = params.size()==1 ? params[0] : default` — the assert rejects the valid
      1-param call.) Such a "defensive" throw misleads debugging worse than no check.
- [ ] KEPT only if it guards a real, DEMONSTRATED failure or a **silently-wrong** path
      (wrong-kernel selection, a BF16-vs-FP8 weight fork, a buffer-capacity overflow that would
      corrupt adjacent memory) — and even then it uses the existing/upstream idiom (a plain
      `assert` in the house style), not a new multi-line fail-loud `abort` + narrative.

## 5. No gratuitous renames / type-descriptors on working code

- [ ] No existing symbol renamed (function, parameter, enum symbol) unless the rename is the
      point of the change — a rename breaks source/API references (external callers, imports).
- [ ] No task-type enum **renumbered** (its integer value changed). `task_type` is serialized
      numerically, so changing a value breaks already-serialized task graphs — a surviving
      TP8-only reducer keeps its upstream ID, it is not re-slotted into a deleted variant's
      number. (A pure symbol rename with the value unchanged is the weaker source-only concern.)
- [ ] No type annotation / `TYPE_CHECKING` import / descriptor field added purely for
      annotation (inert at runtime, adds a dependency edge for nothing) when the code ran
      upstream without it.
- [ ] No accept-then-discard "API-parity" params (`del eps, epsilon`; a `group_size` arg that
      exists only to be rejected when `!= 128`). Unused surface that only exists to be validated
      away — remove the param and its guard.
- [ ] If a name/shape/ID ran upstream, the diff reverts to it rather than introducing a new one.

## 6. PR shape

- [ ] One PR = one coherent topic (PRs squash-merge to one commit). Not a de-cruft + a perf lever
      + a new kernel folded together.
- [ ] Registration/ABI coherent: a new task ID updates `runtime_header.h` enum, the `src/kernel`
      registration, the wrapper, and TMA/runtime glue TOGETHER — no dangling enum without a
      handler, no handler for a deleted enum.
- [ ] New kernel/task ships its `tests/runtime_python/.../test_*_testmode.py` (+ `pytorch_reference.py`).
- [ ] `git diff --stat <merge-base>..HEAD`: the shared-surface footprint
      (`persistent_kernel.cuh`, `persistent_kernel.py`, `multigpu.py`, `runtime.cc`) is minimal
      and generic — a single-model change reaches into no shared file beyond a generic op-API.
- [ ] Formatter clean (`bash scripts/format.sh`, clang-format-15). No generated/local artifacts
      staged (`scratch/`, `outputs/`, `_results/`, weight caches, generated `test.cu`/`.so`, perf
      logs, `PR_DESCRIPTION`/campaign notes).
- [ ] Behaviour bar for a cleanup/refactor: the emitted megakernel source (the generated
      `test.cu`) for an unrelated model is byte-identical pre/post, or a token-identical run
      proves the shared path is unchanged.
