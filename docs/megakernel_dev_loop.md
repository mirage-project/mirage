# The megakernel loop: design

**Goal:** turn one model part into a static megakernel part (graph, task templates, plan, post-processing, plug-in) that beats
SGLang in the server and end to end. The loop runs with few human decisions, and repeats for every new part.

The K3 MoE layer was built by hand through this loop (09-16 → 10-07); KDA earlier, on the old tree. This doc is the mechanism.
What we learned is in `docs/megakernel_loop_rules.md`: rule ids like `C11` and `F6` below point there.

---

## 1. Principles

1. **The thing measured is a candidate,** not a plan: (template variants, split, assignment, post-processing choices). Every
   stage's output is a file, and every measurement names the candidate it measured.
2. **Post-processing is part of evaluating a candidate,** not a final step. A pass changes task times (F1, F5) and its
   feasibility depends on the assignment (F3, F4). So it runs on every candidate before measurement, and its choices are context
   in the cost records.
3. **The cost model is learned and its error is tracked.** The search is trusted per phase only while its error is under 1 µs on
   measured candidates (E3). Otherwise the loop spends the round on measurements that reduce the error, not on exploiting the
   search.
4. **Knowledge is enforced, not only written.** A rule becomes an automatic check where possible (section 6), a review item where
   not, or a gate. The loop fails a round on a check, not on a reader noticing.
5. **Cheap filters first.** Prediction, then harness, then server window, then server, then end to end. A candidate is dropped at
   the first stage where it loses beyond noise (rule H3).
6. **Nothing is retried without a new reason.** The ledger and the C2 "lost" list are read before every proposal.
7. **Human decisions are batched** at fixed points (section 13). The loop keeps working on everything not blocked.

---

## 2. Artifacts

Code that is the same for every part lives in the repo:
- `python/mirage/mpk/loop/`: driver, lints, passes, evaluation, triage;
- `static_harness.py`, `search.py`, `compiler.py`, `static_schedule.py`: exist.

Part-specific code lives in `demo/<model>_<part>/`. Loop state lives outside the repo, in `~/mk_loop/<part>/`.

```
demo/<model>_<part>/
  part.yaml            the part's definition (below)
  graph.py             build(tokens, gpus, tensors) -> StaticMegakernel        (like demo.py build_graph)
  plans/<bucket>.json  the incumbent plan per token bucket                       (like solver_plan.json)
  sglang/              the plug-in glue (shadow / run hooks)
include/mirage/static_megakernel/tasks/<name>.cuh   one task type, with its variants as template parameters
~/mk_loop/<part>/
  state.json           round number, incumbent per bucket, budget used, open user questions, model error per phase
  ledger.jsonl         one line per experiment (below)
  costs/<machine>_<cuda>.json   cost records (cost_records.py format + context fields)
  cand/<id>/           one candidate: candidate.json, schedules/, layer.cu, layer.so md5, gates.json, times/, results.json
  refs/<bucket>/<input>/        reference files (x, weights link, y_ref, partial refs)
```

**part.yaml:**
- name, model, layer ids;
- the server boundary: the kernel names just before and after the part, used by the trace parsers;
- TP size, machine, CUDA version;
- token buckets (e.g. 1, 2, 4, 8, 16);
- input sets per bucket and how each was made (server p10/p50/p90 of the statistic that changes the work: expert count for MoE);
- error limits and how they were derived (A8);
- the bound per bucket and its derivation (A4);
- SGLang's in-server time per bucket and its phase split (A5).

**Contract** (declared in the task type's `*_layer` in `static_megakernel.py`; `lint_contract` checks it is complete):
- maps and units (B1); minimum work per task (B2);
- output slice (B3);
- `combine` options and legality (B4);
- the smem formula + minimum (B5);
- release points and sentinel form (B6);
- dry pass yes/no; one-per-SM;
- cluster pair params;
- allowed hand-off kinds per input;
- **variants:** the template parameters the kernel agent exposes (stages, issuers, tile M, …) with their legal values;
- the task's own bound formula (bytes, MMA count), used by triage.

**candidate.json:** part, bucket, template variant per node, plan (grids + lists), pass choices (smem plan, dry/warm-up,
cluster, hand-off kind per edge, prefetch, PDL trigger, buffer order), code tag, parent candidate, and why it was made (which
triage class or exploration target).

**ledger.jsonl line:**
- round, time, candidate id, box, code tag;
- what was run: harness / window / server / e2e / bench / diagnostic build;
- the gate results;
- the numbers with conditions (cold L2, reps, rounds, min/median/max), vs the incumbent in the same session;
- the verdict: WIN / LOSS / NULL / FAIL-GATE / DIAG;
- the rule ids touched, and the reason it was run.

NULL and LOSS lines are kept.

---

## 3. One round

A round starts from `state.json` and ends with the ledger updated and either a new incumbent or a recorded reason why not. It
has a GPU budget (default 2 box-hours) and a wall budget.

```
 1. preflight      lints on the current tree (section 6); box preflight (md5, co-runners, fabric, clocks)
 2. triage         on the incumbent's latest timing-build trace → one or two targets (section 9)
 3. generate       candidates for the targets:
                     C target → kernel sub-loop (section 10) → new variant → candidates = incumbent with that variant
                     E target → search (section 7) from the incumbent, keep best K = 5 distinct
                     F target → incumbent with the pass's other choices (each choice one candidate)
                     D target → exploration candidates (section 5)
                     B target → contract change → needs a user decision only if it changes meaning (section 13); else proceed
 4. passes         every candidate through the post-processing passes (section 8)
 5. build + lint   generated code, SASS and plan lints (section 6); a failed lint drops the candidate with a ledger line
 6. gates          correctness gates G1–G7 on every candidate (cheap: harness, 2 inputs); full set on promotion
 7. evaluate       the ladder (section 7): harness → window → (server, e2e at milestones); stop each candidate as soon as it loses
 8. record         timing build of each surviving candidate → cost records with context; ledger lines; model error update
 9. promote        a candidate becomes the incumbent only with a window win ≥ noise (H3) in ≥ 3 of 4 rounds, all gates passing
10. decide         next round, next bucket, or stop (section 12)
```

---

## 4. The option space

For each node, the search's options are (grid, template variant). Before, they were grids only.
- A new variant from the kernel sub-loop adds options; it never replaces the old ones silently.
- A variant is retired only when the records show it no faster in any recorded context. That keeps a variant that wins only at
  a small split, as stage depth did on KDA (W9).
- `combine` becomes part of a node's option once B4 is built (todo 7). Until then it stays fixed per template.
- An option without measured records is not selectable (D6). New options are first measured by exploration (section 5).

---

## 5. The cost model: learned, checked, explored

**Record:** cost_records.py's record plus context:
- first on SM, previous node, cold/warm code, tiles in flight on the GPU at the task's start;
- launch mode, pass choices, bucket, input set, machine, CUDA (D2).

**Duration** as D1. Pre/post fitted per (task type, option) as D3. Quantile 0.75, not the median (D4).

**Model:**
- per (task type, option, variant): a table keyed by the context features that the records show matter. Start with first-on-SM,
  previous node kind, and cold/warm.
- A context with no records falls back to the nearest key with records. That fallback is flagged in the prediction, so a
  candidate built on fallbacks is exploration, not exploitation.
- Bandwidth-sharing terms stay out until a context table can't explain the error (D5).

**Model error, every round:**
- For every measured candidate, predict it with the current model (`demo_tools.py --predict`) and compare per phase. The phases
  come from part.yaml, e.g. front / experts / tail. Compare per node the first start and last end (I3).
- `state.json` keeps the leave-one-candidate-out error per phase over the last 10 candidates.

**Explore when the error is high** (above 1 µs in a phase):
- The next round's D target makes candidates that put the worst-predicted nodes into the contexts with the fewest records:
  - first on SM vs behind each kind of producer;
  - many vs few tiles in flight;
  - with and without each pass.
- Generated by perturbing the incumbent (move one node's tasks; change one node's option), plus `random:SEED` plans for coverage.
- Measured with the harness timing build only (`--record`). They aren't compared with the incumbent.
- This is how 09-30's "route 4.19 / 7.10 / 8.58" would have been learned before the search relied on it.

**Search trust:**
- Only phases under the error limit are optimized by the search.
- For the others, the start plan's placement is kept fixed: the search gets the incumbent's lists for those nodes as fixed.
- This stops the 09-30 failure: a 72.37 µs prediction measured 92.3.

**Known gaps the model must grow** (open rules E4, E7, D10):
- cross-GPU arrival (tail about 8 µs low);
- pool end imbalance;
- the pool's fixed start.

Each is a context or a term added when triage shows the error lives there.

---

## 6. Automatic checks

Each check is a script in `loop/lints/`. A failure drops the candidate (or fails the round for tree-level checks) and writes a
ledger line with the rule id.

- **`lint_contract`** (B1, B4, B5, contract completeness): every field present; candidate grids derivable; combine legal; the smem
  formula evaluates for every option.
- **`lint_source`** on task templates and generated code:
  - no static `__shared__` (C6);
  - `__threadfence` inside a task loop flagged (C18);
  - G/Maps `__grid_constant__`, bodies `__forceinline__` (C11);
  - `globaltimer` asm with a memory clobber (H7);
  - no part constants in the generator (B14).
- **`lint_build`** (C6, C13, C20, C24): smem ≤ 227 KB with the `.nv.shared` size; no `-use_fast_math`; the arch flags;
  registers per thread reported and compared with the parent.
- **`lint_sass`** (C7, C11, C14, C15, C23), per task body using `-lineinfo`:
  - LDL/STL = 0;
  - no fp64 ops;
  - no CALL into bodies;
  - R2UR count in loader loops;
  - instruction and MMA-site counts vs the parent (a rise > 5% is flagged for review);
  - the default build's SASS identical to the baseline's when only flags changed (registers normalised).
- **`lint_plan`** (B2, D6, F3, F4):
  - `check_deadlock_free`;
  - per-SM cap;
  - min work per task;
  - no option without records;
  - cluster pairs fit;
  - the smem plan fits with no space deadlock.
- **`lint_records`** (D2, D8): every record has the context fields, the current code tag, and a passing gate run.
- **`lint_rename`** (B12): after a rename, every changed line in every file under the work tree, tracked or not.
- **`code_tag`** (D9): the md5 of the templates' source without formatting (`static_schedule._code_tokens`: comments gone,
  whitespace kept only between two words), so a clang-format run or a comment edit keeps the tag. Before 10-07 it was a plain
  source md5, and a whitespace change invalidated 434 records. Not a SASS hash: a task body is inlined into `layer_kernel` with
  the plan's template arguments, so its SASS differs per plan, not per template version. `lint_sass` keeps each build's SASS
  summary (`sass_summary.json`) for comparison with its parent instead. Layout changes (B13, C22) are in the generated code,
  not the templates, so they are context of the candidate, not part of the tag.
- **`preflight_box`** (A7, H1, H5, H6):
  - md5 manifest match;
  - no other process on the GPUs;
  - fabric state;
  - clocks;
  - the L2 flush by reading;
  - free disk;
  - the data disk mounted.
- **`check_trace`** (H5): our kernel name present in the server trace; the variant's stamp present.

---

## 7. Search and evaluation ladder

**Search** (`search.search_plan`):
- always from the incumbent, solved fixed first, handed in complete (E1);
- ≥ 60 s, and report status, objective and bound (E2);
- untrusted phases fixed (section 5);
- the best K = 5 distinct solutions kept (CP-SAT solution callback, deduplicated by the per-SM node sequence).

**Ladder** (a candidate moves up only by winning):

| step | what | rounds | moves up if | stops if |
|---|---|---|---|---|
| predict | model, with fallback flags | — | in the best K, or an exploration target | — |
| harness | `demo.py`, cold L2, 20 launches, alternated with the incumbent | 2 | median ≤ incumbent + 0.3 µs | worse by > 0.3 in both rounds |
| window | `test_plugin.py --prev-us 5`, alternated | 4 | better by ≥ 0.3 in ≥ 3 of 4 | not better in 2 |
| server | `sgl_mks_server.sh run`, per-layer blocks | 1 | at a milestone (section 12) | worse than the window said by > noise → investigate |
| e2e | `sgl_e2e_server.sh` + `compare_answers.py`, ≥ 20 requests per arm, neighbours unchanged (F6, H4) | 1 | at a milestone | any answer differs from the gate's expectation, or a neighbour slows |

A candidate that passes the window becomes the incumbent (step 9 of the round). Server and e2e run at milestones, not every
round, because they cost 30–60 min each.

---

## 8. Post-processing passes

**Protocol.** Each pass is a module with:
- `apply(candidate, graph, costs) -> candidate | Infeasible(reason)`: deterministic; changes only its own choices and the list
  numbering;
- `features(candidate) -> dict`: the context it adds to cost records;
- `constraints(graph) -> optional search constraints`: soft, with a bonus equal to its current estimate; e.g. "at most one
  r_norm per list" so pairing stays possible;
- `estimate(candidate) -> µs`: from the ledger's ablations (below); none until there are ≥ 3 ablations in similar contexts.

**Order** (each pass sees the earlier ones' output):
1. smem plan per SM (F4): {owner, offset, size, early} per buffer. Today it is one maximum for all SMs.
2. warm-up / dry pass (F1, F2): which first tasks run a dry pass; moving a dry-pass task behind an empty slot.
3. same-SM consumer placement (a consumer behind its last producer).
4. cluster pairing (F3): exists (`compiler.pair_clusters`).
5. hand-off kind per edge (F7–F12): from edge attributes (counts, byte ownership, cross-GPU, release points, skew).
6. prefetch (F5, F6): smem prefetch of the next task's weights from the leftover smem; L2 prefetch only in windows where HBM is
   idle.
7. PDL trigger (F15), buffer and exchange order (F19).

**Ablation.**
- When a candidate is promoted, the next round also measures it with each pass turned off (one harness pair per pass).
- The difference is the pass's effect in that context. It goes to the ledger, and from there into `estimate`.
- This is how the loop learns what a pass is worth under different assignments, and why the search gets the pass's constraints
  with a bonus instead of hard rules.

**The loop between E and F:**
- The search ranks with costs that already include pass effects as context, plus pass constraints with their estimates.
- The ladder measures post-processed candidates.
- The records from those measurements change the next search.
- Nothing in F is a one-time final step.

---

## 9. Triage: choosing the next target

**Input:** the incumbent's latest timing build (`Harness.task_times`, all GPUs) and its prediction.

**Steps:**
1. **Critical path.** The latest-ending GPU's last task, walked back through either the previous task on the SM or the latest
   input (whichever ended later).
2. **Attribute** every µs on the path to one class:
   - `work(T)`: a task's duration minus its own bound (contract bound formula), summed per task type T → **C** on T;
   - `cold`: a first-on-SM task's duration minus that option's warm quantile → **F** (dry pass, warm-up, order);
   - `handoff`: input wait minus the hand-off floor for its kind (F7) → **F** (hand-off kind, polling F10, multicast F12);
   - `start`: kernel start → first task start, beyond the measured floor → **F** (prologue F18, PDL F15, start barrier F16);
   - `imbalance`: latest SM end minus mean SM end, for pools and waves → **E**;
   - `coarse`: waits on producers whose output the consumer doesn't read (from slice overlap) → **B** (finer maps, B3);
   - `model`: phases whose error is over 1 µs → **D** (explore).
3. **Pick** the largest one or two classes. Skip any target whose ledger history has ≥ 2 LOSS/NULL lines with no new reason, and
   any change on the C2 "lost" list.
4. **Write a proposal** for each target: the step it removes (with the stamp that shows it), the predicted gain, and the test
   that would refute it (I1).

When stamps are too coarse to attribute, the first action is a stamp build for that stretch (I1), not a change.

---

## 10. The kernel sub-loop (C)

For one task type T with a target from triage:
1. **Bound and current state:** T's own bound; its SASS summary; its stage stamps in the incumbent; the competitor's
   implementation of the same work if one exists (A6).
2. **Plan:** one change from the C2 record or the skills' ladders (`b200-gemm-optimization-ladder`, `b200-tma-pipeline-designer`,
   …), not on the lost list; or a structural port from the competitor.
3. **Implement as a variant** (a template parameter), with the default unchanged (C23).
4. **Unit gates:**
   - correctness vs an fp32/fp64 reference with asymmetric and sentinel data;
   - `lint_source`, `lint_sass`, `lint_build`;
   - the barrier ledger review for any barrier change (C4);
   - unit time at the production grid inside the megakernel (`mpk-faithful-gate`: slowest CTA), alternated with the parent
     variant.
5. **Keep it** if the unit time is better above noise; otherwise a ledger LOSS line, and the reason is added to C2 if it is general.
6. **The variant goes to the round** as candidates (the incumbent with that variant, plus the search from it if the variant changes
   T's time a lot).

The test writer (section 13) owns the unit reference and freezes it before step 3.

---

## 11. Gates

- **Cheap gates on every candidate:**
  - G1 on 2 inputs;
  - G2;
  - G3 (`--dump-cmp`, 20 launches);
  - G7.
- **Full gates on promotion:**
  - G1 on all input sets and buckets;
  - G4 chained (reps ≥ 2);
  - G5 mutation;
  - G6 contention and repeated;
  - G8 at server milestones.
- **G11:** no performance verdict for a candidate that failed any gate.

---

## 12. Milestones, stopping, budgets

- **Milestone:** after 3 promotions, or a total gain ≥ 1% of the part's server time since the last server run. Then a server run
  and an e2e run of the incumbent.
- **Bucket done:** two rounds in a row with no promotion while the triage classes left are each < 0.5 µs or blocked; or the
  incumbent within 5% of the bound. Then the next bucket, starting from the nearest bucket's incumbent.
- **Part done:** every bucket done, the gates full, the server and e2e milestones passed. Then the cleanup for the PR (section 15).
- **Budgets:** 2 box-hours per round, a daily cap from the user. A round that hits its budget writes what it didn't finish into
  `state.json`.
- **Box:**
  - every run detached with a waiter;
  - md5 proof;
  - the decisive check first;
  - stop on the first lost connection and report;
  - stop the instance when the queue is empty.

---

## 13. Agents and the user

**Roles,** each started with the rule ids of its stages plus the matching skills:
- **orchestrator:** `state.json`, the ledger, triage, budgets, the box. The only agent that touches the box.
- **spec:** part.yaml, references, bound, contracts. Rules A*, B*. Skills `b200-kernel-roofline-triage`,
  `b200-scope-layout-dispatch`, `add-mpk-task`.
- **kernel:** the C sub-loop for one task type at a time. Rules C*, W*/L*. Skills `static-mk-kernel-lessons` and the `b200-*` set.
- **test writer:** unit references and the gates; frozen before the kernel agent starts. Never the same agent as kernel or search
  (`ferret-kernel-system`). Rules G*.
- **compiler:** costs, model, search, passes. Rules D*, E*, F*.
- **analyst:** reads stamps and traces, writes the triage attribution and the proposals. Rules H*, I*.

Agents exchange files (candidate dirs, ledger, proposals), not conversation. A proposal without its measurement and refuting test
is rejected by the orchestrator (I1).

**User touchpoints** (only these; everything else runs):
1. **Part start:** part.yaml (buckets, inputs, error limits, bound, SGLang's time) and the contracts. One review.
2. **Meaning changes,** collected and asked together at the end of a round, while the loop works on other targets:
   - a new task type;
   - a contract's meaning;
   - numerics (accumulate dtype, error limits);
   - the shared runtime (`core.cuh`, `runtime.cuh`, `host.cuh`);
   - dropping a gate or feature;
   - a departure from the reference harness's design.
3. **Milestones:** a short report — incumbent, server and e2e numbers vs SGLang, what changed, what was tried and lost.
4. **Part done:** the PR.

---

## 14. Several parts

- **Shared limits** checked by `lint_build` across all parts in one launch: `MAX_BUFS` 24, `MAX_MAPS` 32, counter lines, 227 KB smem,
  16 named barriers, registers.
- **One hand-off interface** between parts: the part's input and output tensors, with their sentinel or counter form declared like
  any contract.
- **Regression:** a change to the shared runtime reruns every finished part's full gates and a window A/B of its incumbent.
- **Later:** one graph with several parts and one search, so tasks of different parts can overlap.
  Collective seams gate run-ahead traffic (FIR: push spread 1.5 → 2.6 µs).

---

## 15. Running KDA through the design (on paper)

KDA's work, as the hand-written kernel before this compiler did it: proj GEMM ring, conv, recurrence, o_proj,
cross-GPU exchange, attnres. Walking it through the loop shows what the design needs beyond MoE:

- **B, persistent state:** recurrent state and conv windows live across launches (B11). The contract needs a `state` kind of
  tensor: written at the end of launch n, read in launch n+1, never inside one launch. G4 must chain launches (reps ≥ 2): the
  conv-roll bug was invisible at reps 1.
- **B, split legality:** the recurrence can't be split along time, only by head (B4). Its options are per-head groups.
- **C, existing bodies:** the old tree's bodies (gemm_ring, oproj128, recurrent, rolemajor) become task types with variants. The
  W8/W12 numbers say which variants to keep (dual issuer, tensor-core recurrence).
- **D, cold code:** cold code and tf32 first use are large for KDA (3.0 vs 1.6 µs; 3.2 µs/layer), so cold/warm context is needed
  from the first round, not later.
- **F, hand-offs:** KDA's best hand-offs were Lamport data-as-flag with triple buffers (F8) and an aggregate counter with backoff.
  The hand-off pass must offer these kinds, not only MoE's counters and multicast.
- **F, prefetch:** KDA's L2 prefetch is the F6 example, so the prefetch pass for KDA is gated end to end from the start.
- **H, server:** the plug-in fork faulted at MK_NB=10 with MK_TASKS_GLOBAL (open). The generic plug-in path replaces the fork, so
  that fault becomes a G9 hang/crash investigation on the new path, not a port.
- **A, buckets:** KDA is linear attention: its per-step work doesn't grow with context length (fixed state, fixed conv window),
  so its input sets are by token bucket only.

Gaps this adds to the build list: the `state` tensor kind in contracts and gates; the Lamport and aggregate-counter hand-off kinds
in the pass; cold/warm context from day 1.

---

## 16. What to build, in order

Each item says what it unlocks.

1. **The generic part harness and plug-in** (part.yaml, `graph.py` interface, references by path, the boundary kernels as
   parameters). `demo.py`, `test_plugin.py`, `plugin_host.cuh` and the trace parsers are MoE-only. Unlocks any second part.
2. **Lints** `lint_contract`, `lint_source`, `lint_build`, `lint_sass`, `lint_plan`, `preflight_box`, and `code_tag` from the
   templates' source without comments and formatting (D9). Unlocks automatic rule enforcement; stops the stale-tag problem.
3. **Context in records + the model with error tracking** (section 5), plus the exploration generator. Unlocks a search that is
   trusted per phase.
4. **The round driver:** `state.json`, ledger, candidate dirs, the ladder with early stops, triage attribution. Unlocks running
   rounds without me in the middle.
5. **The pass protocol,** with the existing passes ported (dry pass, cluster pairing, PDL), then the smem plan, then the smem
   prefetch, then the hand-off kind per edge, with ablation measurement.
6. **Template variants in the option space** and the kernel sub-loop's unit gate runner.
7. **Token buckets.**
8. **KDA** through the loop, with the gaps from section 15.

### Status 10-07: items 1-7 built, not yet run on the box

Code in `python/mirage/mpk/loop/`; tests in `tests/static_loop/` (plain scripts, run without a GPU on fakes and simulated stamps:
all pass). Nothing below has run on the box yet.

1. `part.py` (part.yaml + hooks), `run.py` (the generic harness run), `candidate.py`; the MoE part is `demo/kimi_k3_moe/part.yaml`
   + `part.py`. The window step is generic through part.yaml `window` commands, but the plug-in itself (`make_plugin.py`,
   `test_plugin.py`, `plugin_host.cuh`) and the server trace parsers are still MoE-only.
2. `lints.py` (contract, source, build, SASS, plan, records), `preflight_box.sh`, `code_tag` from normalized template source
   (10ccb3d9ead3; local costs retagged). `lint_baseline.json` holds the tree's 25 findings at the start (12 C6 static
   `__shared__` in moe_experts / sum_rmsnorm, C11 `drain_acc`, H7 two globaltimer reads, ...): known ones flag, new ones fail.
3. `cost_records.py` adds p75_us and ctx_stats (cold_frac, in_flight_median) without changing record grouping (search.Costs
   unchanged); `model.py` Model (p75, nearest-key fallbacks listed), predict, phase_error, ErrorTracker; `explore.py`
   perturb_plan, exploration_targets. Open: the prediction's non-final phases are not earliest-start (the solver minimizes the
   end only), which biases their error toward "untrusted".
4. `driver.py` (state, ledger, rounds, ingest, promotion, milestone), `round_job.py` (box side: lints, generate, ladder with
   early stops, timing builds, triage), `box.py` (sync with md5 proof, push, detached launch, waiter that stops on the first
   lost connection, fetch), `triage.py` (critical path, class totals).
5. `passes.py`: dry pass, cluster pairing, PDL trigger ported, with ablations and estimates from the ledger; smem plan, consumer
   placement, hand-off kind, prefetch are protocol slots that raise Infeasible until the templates support them.
6. Variants: `_add_node(variants=...)`, `compiler.candidate_variants / apply_variants`, search options = grid x variant,
   plan "variants"; `unit.py` is the unit gate. No template declares a variant yet. `search_plan(fixed_nodes=...)` keeps the
   untrusted phases' placement.
7. Buckets: per-bucket incumbents in state.json (`driver init --bucket`); the MoE part has only the 8-token inputs.
