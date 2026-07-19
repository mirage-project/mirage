# Runtime-V2 wiring recipe — the 8-file checklist (+ footguns)

The v2 twist on the v1 7-file lifecycle (`/add-mpk-task`): role variants, smem_info regions,
the §1.1 dep-prefix, the task_offset=bid.x list, and the skip_after_step0 contract.
Work through the files IN THIS ORDER; each step names its footgun.

## 1. `include/mirage/persistent_kernel/runtime_header.h` — TaskType enum

Add `TASK_<OP>_V2 = <next-free>` (examples: `TASK_LINEAR_SM100_V2 = 242`,
`TASK_DSV3_FFN_MEGA_V2 = 348`, `TASK_TENSOR_INIT_V2 = 352`,
`TASK_ATTN_BLOCK_MEGAKERNEL_V2 = 353`).
FOOTGUN: v2 ids were renumbered in the merge — never hardcode numeric ids anywhere else
(profiler buckets switch on enum NAMES, runtime_v2.cuh:74-77).
FOOTGUN 2 — **ID ranges have semantics** (upstream cba02075): `create_tma_desc_by_task`
auto-creates TMA descriptors for the SM100 TMA range **231-256**. A task that consumes
`input_tma_desc_ptrs` must sit inside that range OR be added to create_tma_desc_by_task
explicitly; a non-TMA task must stay OUT of it (upstream parked its non-linear v2 ids at
224-229 for exactly this; ours live at 326+). Check both directions when allocating an ID and
at every upstream merge — upstream renumbered to 244/245 + 224-229, ours diverge.

## 2. `tasks/blackwell_v2/<op>_v2.cuh` + `<op>_v2_spec.h`

Write per house-style.md. Then add the include to
`tasks/blackwell_v2/task_header.cuh` (alphabetical, with the `// kernel::<ns>` comment —
task_header.cuh:14-32). That header is pulled into every generated test.cu via
`persistent_kernel_v2.cuh:19`, so a .cuh edit needs NO library rebuild (runtime nvcc JIT).
FOOTGUNS: `extern __shared__ __align__(1024)` (see house-style §4); own namespace
`kernel::<op>_v2` (no collision with v1 `kernel::`); spec.h must be host-safe (no CUDA types
beyond what task_register.h brings in).

## 3. `include/mirage/kernel/task_register.h` — declare
`int register_<op>_v2_task(threadblock::Graph const &bgraph, std::vector<int> const &params);`

## 4. `src/kernel/task_register.cc` — the registration function

Include the spec at the top (spec includes live at task_register.cc:19-25). Body skeleton
(full pipeline example :1959-2160; consumer-only example :7480-7521):

```cpp
int TaskRegister::register_<op>_v2_task(tb::Graph const &bgraph,
                                        std::vector<int> const &params) {
  // 1. params + bgraph.operators split (inputs first, then outputs — new_input order)
  // 2. dims/strides: row stride = dtensor.stride[0] (P1/P2 invariant, :85-88), NOT dim[1]
  int const num_tasks = (int)bgraph.grid_dim.x;   // grid-wide ops bake this into the body

  auto emit_body = [&](mirage::transpiler::CodeKeeper &c) {
    c.e("kernel::<op>_v2::<op>_task_impl(");
    c.e("    task_desc,");
    c.e("    static_cast<int>(task_desc->task_metadata.task_offset),");  // logical CTA/tile id
    c.e("    $, ...);", num_tasks);                // iter_num, instruction_index as needed
  };

  mirage::transpiler::CodeKeeper code;             // plain variant string = dedup identity
  code.inc_indent(); emit_body(code);
  int variant = register_task_variant(TASK_<OP>_V2, code.to_string());

  mirage::transpiler::CodeKeeper consumer_code;
  consumer_code.inc_indent();
  emit_dep_wait_consumer_prefix(consumer_code);    // §1.1 — MANDATORY, FIRST LINE
  emit_body(consumer_code);

  TaskRoleVariantCode role_code{/*init_semaphores=*/"", /*loader=*/"",
                                /*launcher=*/"", /*consumer=*/consumer_code.to_string(),
                                /*storer=*/""};
  register_v2_task_role_variant(TASK_<OP>_V2, variant, role_code);
  register_variant_smem_info(TASK_<OP>_V2, variant, ::kernel::<op>_v2::make_smem_info());
  return variant;
}
```

- **§1.1 LETHAL INVARIANT**: the consumer body MUST begin with `emit_dep_wait_consumer_prefix`
  (emits `consumer_dep_prefix(runtime_config, task_desc, runtime_smem, instruction_index,
  iter_num);`, :62-66). It does the cross-SM event spin AND arrives the per-slot
  `SEM_DEP_READY` — skip it and the NEXT task on that ring slot deadlocks silently
  (runtime_v2.cuh:609-656). Grep-check: every `register_*_v2_task` has it (:2887, :7506, :9708…).
- Pipeline kernels: also fill `init_semaphores` (mbar_init every SEM ordinal with its count +
  `fence.mbarrier_init.release.cluster`, :2035-2059), `loader`, `launcher`
  (launcher/consumer get the dep prefix, loader does it inline — :2120-2140), and set
  `role_code.auto_consumer_finish = false` when a non-consumer role releases pages (:2145-2151).
- Multi-role MAC-fold (FFN 7w pattern): emit the SAME body into loader/launcher/storer WITHOUT
  the dep prefix — their first action is an acquire on a consumer-released tag, which chains the
  dep transitively (:9716-9727). Pass the salted sync_tag
  `((unsigned long long)instruction_index + 1ull) * 0x9E3779B97F4A7C15ull` (:9835-9836);
  `0ull` selects the consumer-only compile-out.
- FOOTGUN: variant dedup keys on the PLAIN code string — a param that changes behavior but not
  the emitted string (e.g. only smem_info differs) silently aliases variants. Make every
  behavioral param appear in the emitted string (tensor_init's skip guard does this, :2854-2866).

## 5. `src/kernel/graph.cc` — name dispatch

```cpp
} else if (name == "<op>_v2") {
  int variant_id = task_register->register_<op>_v2_task(customized->bgraph, params);
  task_config[op] = std::make_tuple(NUM_IN, NUM_OUT, TASK_<OP>_V2, variant_id);
```
(v2 examples at graph.cc:595-600, :726-727, :774-776, :1146-1147.) The tuple's
NUM_IN/NUM_OUT must equal the registration's split — mismatch = "Invalid __global__ read"
at runtime, not compile time.

## 6. `src/kernel/runtime.cc` — TWO stops

a) **task_offset list** (~:571-635): if the body indexes by tile/logical-CTA id, add the enum to
   the `task.task_metadata.task_offset = bid.x;` chain (per-tile linears :579-585, FFN chain
   :593-605, attn chain :616-624; 3D grid-wide ops use
   `bid.x + bid.y*gx + bid.z*gx*gy`, :631-635). **Omitting this is the deadliest v2 footgun**:
   task_offset stays garbage → grid-barrier deadlock / wrong tiles, SILENT (comment :609-610).
   Note v1 megas read `merge_task_offset` (:522-542) — the v2 bodies read `task_offset`; don't
   conflate the union members.
b) **task_type_to_name** (~:2108-2236): add the string or the perfetto trace mislabels the task
   and per-position tooling filters it out (the FFN-324 "missing task" incident).

## 7. `python/mirage/mpk/persistent_kernel.py` — the layer method

```python
def <op>_v2_layer(self, ..., grid_dim, block_dim=(128, 1, 1)):
    assert self.use_v2_runtime, "<op>_v2 is v2-only"
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(x, (-1, -1, -1), ...)   # v2 convention: often unpartitioned; the
    ...                                        # body slices by task_offset itself
    self.kn_graph.customized([...], tb_graph)
    self.kn_graph.register_task(tb_graph, "<op>_v2", params)
```
- block_dim is the LOGICAL body width (consumer-only = (128,1,1)); the v2 worker always
  launches 256T (persistent_kernel.py:2957-2959 comment).
- **Grid-wide fused op ⇒ the co-residency HARD GATE**:
  `assert num_tasks == self.num_workers` — the per-SM queues are strict FIFO; two same-op
  tasks on one worker deadlock the in-op grid barrier (:2949-2955 attn, :4341-4345 ffn_mega).
- linear-family v2: output dim0 ≤ 16 (single BLOCK_N=16 A-tile — rows 16+ silently uncomputed;
  asserted since the M=128 catch, tests/runtime_python/blackwell_v2/README.md).

## 8. `python/mirage/mpk/models/<model>/builder.py` — the use_v2 branch

Gate on `self.mpk.use_v2_runtime` (builder.py:1781, :2545, :3056, :3873). Keep the v1 path
byte-identical — the v2 path is opt-in (`--use-v2`).

**MONOTONIC-BARRIER CONTRACT (M3 root cause, 2026-07-07)**: any scratch whose head bytes hold a
monotonic grid-barrier counter (`need = num_tasks*(iter+1)`, never reset —
attn_block_megakernel_v2.cuh:120-140, dsv3_ffn_v2.cuh:1817-1818) MUST be seeded with
`tensor_init(..., skip_after_step0=True)`:
- persistent_kernel.py:2233-2304 → `v2_params=[1]` → task_register.cc:2776-2900 wraps ONLY the
  memset in `if (runtime_config.step[0]==0)`; the dep-prefix / FINISHED-arrive / event-trigger
  stay UNGUARDED (the task participates in the protocol every step).
- builder call-sites: attn scratch ~builder.py:1803, FFN `_ffn_bar` ~builder.py:3038.
- Dropping it = **iter-0-fine / iter-1-hang** (counter re-zeroed while need grows;
  discriminator: `num_tasks*1 ≥ num_tasks*(it+1)` iff `it ≤ 0`). See validation-debug.md.

## 9. Cross-cutting rules

- **Default-build byte-identity**: new task types are additive; levers/variants are env- or
  param-gated default-OFF. A default flip needs a measured Δ in the commit body
  (mpk-commit-reviewer enforces).
- SEM budget ≤ 31 op-private (SEM_OP_BASE=1 of MAX_DYNAMIC_SEMAPHORES=32).
- SMEM: ≤16 regions (MAX_SMEM_REGIONS_PER_TASK), total ≤ 224256 B (= planner CAPACITY_BYTES =
  spec PLANNER_CAPACITY_BYTES; PAGE_SIZE=16KB × 14 pages), static_assert in spec.h.
- Rebuild matrix: body-only `.cuh` edits → JIT only (re-run). Any `*_v2_spec.h` field the
  planner/registration sees (task_register.cc:19-36 #includes the spec headers — regions,
  ordinals, make_smem_info geometry) is Class B: it REQUIRES the C++ library rebuild, or the
  planner's geometry silently skews vs the JIT-compiled kernel. `task_register.cc`/`graph.cc`/
  `runtime.cc`/headers → `pip install -e . -v`; if Python doesn't reflect it,
  `rm python/mirage/core.*.so && touch python/mirage/_cython/*.pyx` and reinstall.
- `bash scripts/format.sh` before any commit; never stage `scratch/`, `outputs/`, or
  `.claude/` session state (once `.claude/skills/GITIGNORE_PATCH.txt` is applied,
  `.claude/skills/**` + `.claude/agents/**` are tracked-by-design — but a KERNEL commit
  still shouldn't mix skill edits in).

## 10. Ship checklist (tick all)

- [ ] enum + name map + (if indexed) task_offset list entry
- [ ] task_header.cuh include; `__align__(1024)`; namespace
- [ ] spec.h: ordinals, make_smem_info, capacity + drift static_asserts
- [ ] register fn: §1.1 prefix first line of consumer; role bodies; smem_info; params in string
- [ ] graph.cc tuple arity == registration split
- [ ] py wrapper: use_v2 assert; grid-wide ⇒ num_tasks==num_workers assert
- [ ] builder: v2 branch opt-in; barrier scratch ⇒ skip_after_step0=True
- [ ] test-mode case added (validation-debug.md §1) + in-MPK 0-3 probe + iter≥1 run
