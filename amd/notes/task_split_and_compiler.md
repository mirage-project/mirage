# Task split and the compiler on MI355X (K3, 2026-10-16)

In one sentence: a layer is a graph of nodes; I choose each node's **task split** (its grid) and each CU's **task list** by
hand-made plan passes guided by a per-CU read budget and decided by same-session A/B runs, and the compiler turns that plan
into one schedule per GPU and then into one generated HIP kernel (`layer.cu`) that every CU runs.

Code: `python/mirage/mpk/` (`static_megakernel.py` the graph API, `compiler.py` plan -> schedules, `static_schedule.py`
schedules -> `layer.cu`, `search.py` / `solver.py` the cost-model search), `amd/tools/plan_one_gpu.py` (plan passes),
`amd/tests/test_*_layer.py` (harnesses that build the graph, the plan, and run it), `amd/tools/gen_rank_lib.py` (server
libraries). Measurements cited as `kdaNN` / `srvNN` are in `amd/notes/kda_mi355x.md`, the MoE ones in
`amd/notes/layer_math_mi355x.md`.

## 1. Terms

- **node**: one operator of the layer graph, made by a layer call (`gemm_tile_layer`, `topk_route_layer`,
  `moe_experts_layer`, `short_conv_silu_layer`, ...). It has a task type (the HIP function `static_mk::run_<name>`), params,
  input / output tensors, and a **grid** (x, y, z).
- **task**: one grid position of a node. A task reads and writes **slices** of the node's tensors: for each grid axis that
  the node's tensor map ties to a tensor dimension, that dimension is cut into `grid[axis]` equal blocks and the task gets
  its block (`compiler.box`). For `gemm_tile`: grid x cuts the weight's rows (**row block**, 16 / 32 / 64 / 128 rows), grid y
  cuts K (**K part**; the K parts' partial sums are added by the reader or by the last part).
- **task split** = the node's grid: how many tasks, how big each is.
- **CU list**: the ordered tasks one CU (one workgroup of 256 threads, the whole 160 KB LDS) runs, one after another.
- **plan**: `{"grids": {node: grid}, "lists": per GPU, per CU, [(node, position), ...]}` (`compiler.plan_from_file`).
- **schedule**: one JSON file per GPU (`schedule_gpu<g>.json`): the nodes, every task with the tasks it waits for, the CU
  lists as task ids. **layer.cu**: the generated kernel + host code.

## 2. How the compiler works

```
graph      StaticMegakernel(num_gpus=8); layer calls          each call: _add_node(name, grid, tensors, maps, params,
                                                              buffers, counters, changeable grid dims, ...)
plan       a plan file + plan passes (section 3)              {"grids", "lists"}
schedules  compiler.compile_plan(mpk, plan, out_dir)          schedule_gpu<g>.json per GPU
kernel     static_schedule.generate_code / compile_static     layer.cu -> hipcc -> .so (harness) or
           (+ amd/tools/gen_rank_lib.py)                      layer_rank.cu -> lib.so (one process per GPU, the server)
run        static_harness.Harness / test_rank_lib.py / amd/serving/sglang_k3.py
```

### 2.1 What a layer call declares

`static_megakernel.py` records per node: its tensors and how each grid axis maps onto each tensor's dimensions
(`input_maps`), which grid axes the compiler may change and in what units (`changeable_grid_dims`; `gemm_tile` on HIP: x in
units of 16 rows with 1, 2, 4 or 8 units per task, y in units of 128 K columns), whether the rows are split over the GPUs
(`rows_split_over_gpus`: latent_down, latent_up, sum_quant_send, sum_send compute only their GPU's row blocks), whether the
node runs exactly one task per CU (`one_task_per_sm`: `moe_experts`, `residual_add`), which tasks must run at the same time
(`concurrent_along_axis`), its output buffer (reset value 0 / 0xFF per launch, or the cross-GPU exchange region), its
counters, its buffer and map slots.

### 2.2 `compiler.compile_plan` (plan -> schedules)

1. **apply_grids**: each node the plan gives a grid is regridded; the grid must be one of `candidate_grids(node)` (on each
   changeable axis every size that cuts every tensor dimension the axis touches into whole units, within the declared
   min / max / allowed units per task). Other nodes keep their layer's grid.
2. Per GPU:
   - **gpu_tasks**: every grid position of every node; a `rows_split_over_gpus` node only this GPU's row blocks
     (`owned_positions`: block range [B g / 8, B (g + 1) / 8) of its declared blocks, so every split node gives a GPU the
     same rows whatever its grid now).
   - **add_dependencies**: a task waits for every task of another node that writes a slice overlapping a slice it reads.
     Nothing else: the dependency graph comes from the slices.
   - **lists_of_gpu**: the plan's lists as task ids, checked: every task exactly once, a one-per-CU node's task x on CU x,
     the tasks of a concurrent group on different CUs, at most 63 tasks per CU (`MAX_TASKS_PER_SM` 64 minus the end entry).
   - **write_schedule**: a deadlock check (`check_deadlock_free`: an edge for every dependency and for every two neighbours
     in a CU's list, a concurrent group counted as one task; the graph must have no cycle), then the file.
   (Cluster pairing, `pair_clusters`, is B300 only: HIP has no clusters.)

### 2.3 `static_schedule.generate_code` (schedules -> layer.cu)

Five parts:
- **header**: tokens, GPUs, CUs; the exchange region's layout (offsets and sizes of every cross-GPU buffer, two sets);
  `StaticTask {node, x, y, z}`; per node its grid, counter, output buffer, params, buffer and map slots as template constants;
  `core.cuh` and the used tasks' headers.
- **smem**: the launch's dynamic LDS = the most any node's task type needs (`smem_<name><...>()`).
- **task tables**: per GPU, every CU's list as `StaticTask` entries, then an end entry (it also carries, for the arena
  resets, whether the CU holds a task of the graph's last node).
- **kernel**: `kernel_begin` (launch number, resets of the exchange's next set), a loop over the CU's entries with one
  `case` per node calling `static_mk::run_<name><node, params, slots, inputs' producers>(maps, g, L, task)`, `kernel_end`
  (the arena's next copy reset). The task waits for its inputs itself (counters or data-as-flag polls, inside the task
  type); the schedule's dependencies are not in the kernel, they were used for the checks and to order the lists.
  `MPK_SPLIT_LOOP=S`: two loops, nodes < S then the rest (a one-kernel layer pair, so the two parts' registers are
  allocated apart).
- **host**: allocation of the buffers / maps / exchange regions, launch, wait, the instrumentation buffers (harness). The
  server library (`gen_rank_lib.py`) keeps the kernel and the tables and replaces this part by `rank_host.cuh`.

Build flags (`-DSTATIC_...`) select the variants that are still flags (`amd-kernel-development` skill, section 8).

### 2.4 The search (exists, not used for the final plans)

`search.search_plan(mpk, cost)` builds the solver's input: every node's candidate grids, every task's measured duration from
cost records (`cost_records.py`, from `--record` runs), dependencies from the slices, `moe_experts` as one pool of work;
`solver.solve` picks one grid per node and a CU and start time per task, minimizing the layer's end. The MI355X plans
below were not made with it (it was not run with MI355X cost records for them). What is known about it comes from B300:
with no starting plan it finds no plan in 300 s (10-05); with a starting plan it found nothing faster for KDA / MLA, and
its MoE plan, predicted 1.1 us faster than the hand plan, measured 1.0 us slower (10-14). The model also lacks two things
that decided plans on MI355X: a task's cost depends on its place in its CU's list (a CU's first task is slower) and on the
previous task's remote stores (they share vmcnt with the next task's loads). So the plans below are hand-made and
A/B-tested.

## 3. How I chose the task split on MI355X

### 3.1 The budget that drives it

One workgroup per CU reads ~17-31 GB/s (`l2_carry.hip`, `read_bw.hip`); B300's SM reads ~54. A weight stream on the
critical path therefore needs ~200+ CUs at once to reach ~5-6 TB/s, and a task's weight slice should be small enough that
the node has a task on most CUs: ~100-250 KB per task. But every task also pays fixed costs (its activation round trip,
its stores, its counter, the next list entry), so tasks cannot be arbitrarily small. Per node I wrote this budget first
(bytes, CUs, the chain it sits on), then measured.

### 3.2 Rules, each from a measurement

1. **Weight-streaming GEMMs on the critical path: ~64-row tasks, K split until ~100-200 KB per task, on most CUs.**
   KDA projection (6400 x 7168): `[100,7,1]` = 700 tasks of 64 rows x 1024 K (128 KB). Coarser `[50,4]` 51.1-51.5,
   `[100,2]` 52.0-52.2, `[200,1]` 52.2-52.4 us, finer `[200,7]` (32-row tasks) 57.6-57.7, vs `[100,7]` 51.0-51.4 (kda186,
   kda206); 100x8 57.7, 100x14 59.8, 50x14 57.8 vs 100x7 ~56 in an earlier build (kda_mi355x.md, the first "Lost / no change" list). Router `[7,14,1]` (128 rows x 512 K) beat `[14,14,1]` by ~5
   us (kda206). latent_down `--regrid 28:56,8,1` (64-row tiles, 48 tasks per GPU instead of 24) -1.5 to -2 us; `56,14` +4
   (kda80).
2. **K parts are cheap only where the reader adds them anyway.** gemm_tile's `slots` mode leaves the K parts in a buffer
   that the consumer adds (topk_route, short_conv_silu); `finish` mode has the last part add and push. More K parts =
   more tasks without rows below 64, but each part is one more slice to poll: latent_up with 14 K parts +3.5 (kda194),
   latent_down 56x14 +10 (kda201).
3. **Work off the critical path goes on few CUs.** The shared expert's down GEMM (56 tasks) is needed only at the
   residual: `--pack 34:4` (4 tasks on each of 14 CUs) keeps 42 more CUs for the routed experts; pack 2 the same, pack 8
   +18, after the experts on their CUs +6-7 (kda148, kda207).
4. **Dynamic work is not split statically.** `moe_experts` has one task per CU; inside it, per-die queues hand out the
   W13 units and W2 items at run time (a static deal left late-starting CUs with a full share: median 67 vs longest 93
   us, 10-10).
5. **A task behind a task that stores to other GPUs starts late**: the previous task's remote stores share vmcnt with the
   next task's loads. attn_residual tasks behind o_proj started ~6.4 us after it -> o_proj grid 112 (not 224) and
   `--apart 31,30` (the residual on CUs without o_proj) (kda16, kda19, kda66); MLA o_proj 112: 51.5-52.0 vs 224 53.4-54.0
   (kda69). Same for the MoE: `--apart 39,37,38` (latent_up apart from the R send / S sum).
6. **A node that waits for the slowest CU's previous tasks goes to the light CUs**: `--light 28:27` (KDA's conv off the
   CUs with the most projection tasks: their conv started ~5 us after its input, kda58).
7. **Order inside a die matters when tasks share data**: MLA attention, `--die-group 35` (the head groups of a KV split on
   one die: one HBM read through its L2) and `--rank 35,29,34` (the low KV splits, which hold rows at any context length,
   on the CUs with the fewest tasks before them) (kda35, kda73, kda149).
8. **KV splits for MLA**: 24 (was 40): MLA 46.9 vs 48.7 us per layer in the server (srv17); grouped by 2 heads.
9. **A slice that fits in LDS and waits for its activation copies its weights into LDS while it waits** (`GEMM_LDSW=3`,
   polled GEMMs: o_proj, q_b): 47.8 / 48.2 vs all GEMMs that way 52.6 / 52.4 (kda36).
10. **One kernel per layer pair** (MoE of layer L + agg1 + attention of L+1): `moe_first_plan` = per CU its MoE tasks,
    then its attention tasks, so each CU starts the projection when its MoE work is done, during the MoE's tail. The KDA
    projection there runs as 128-row **hybrid** tasks `[50,7,1]` (each wave's first row tile copied into LDS, its second
    held in registers, all before the activation's wait): -3.2 us vs `[100,7]` (hy1), + `GEMM_L2WARM=256` (the task
    pre-reads its CU's next projection task into the L2): -0.7 (hy2).

### 3.3 The procedure

1. Start plan: the MoE from B300's solver plan widened to 256 CUs (`plan_for_cus.py`: the per-CU nodes get one task on the
   new lists, every old task stays on its list); KDA / MLA from `simple_plan` (each node's tasks round-robin over the CUs in
   graph order) with the grids given on the command line (`--grids`).
2. Profile (timing build, `amd/tools/trace/`), find the node or CU whose end is furthest from its budget.
3. Change one thing with a plan pass (`regrid_plan`, `pack_plan`, `apart_plan`, `light_plan`, `rank_plan`, `die_plan`,
   `first_plan` / `dedicate_plan`; each keeps every list in node order, so no CU meets a task before one it waits for, and
   the compiler's checks still run) or a grid (`--grids`).
4. Same-session A/B on the box, >= 3 rounds, e2e with 1 GB flushed; keep it only if it wins by more than the spread
   (~1-2 us at 85 us); record wins and losses with numbers.

## 4. The final grids and passes (10-16 server libraries)

| library | grids (node: grid) | passes |
|---|---|---|
| KDA | projection 27: `[100,7,1]`, o_proj 30: `[112,1,1]` | `--light 28:27 --apart 31,30` |
| MLA | latent 29: `[34,7,1]`, q_b 31: `[144,1,1]`, gate 32: `[24,7,1]`, o_proj 37: `[112,1,1]`; KV splits 24, groups of 2 | `--die-group 35 --rank 35,29,34` |
| MoE | from the widened solver plan; latent_down 28: `[56,8,1]`, R send 37: `[64,1,1]` after 36, residual 40: `[64,1,1]` (`[8,1,1]` with agg1) | `--fold --pack 34:4 --apart 39,37,38` |
| MoE + agg1 + next KDA | MoE as above, KDA projection `[50,7,1]` hybrid, o_proj `[112,1,1]` | `--order moe` (`moe_first_plan`), KDA's `--today` passes, `MPK_SPLIT_LOOP` |
| MoE + agg1 + next MLA | MoE as above, MLA's grids | `--order moe --attn mla`, MLA's passes |

Exact command lines: `amd-megakernel-development` skill section 4 and `~/mpk-amd/logs/job_fin1.sh`.

## 5. What this does not do yet

- The plans are hand-made per layer; the cost-model search has not been run on MI355X, and its model lacks a task's place
  in its CU's list and the previous task's stores (section 2.4).
- Grids are chosen per node, one at a time; no pass trades one node's CUs against another's except `pack` / `apart` /
  `light` by hand.
- The split of `moe_experts`' work is dynamic (queues), so its imbalance at the tail (the last W13 / W2 items) is not
  something the plan can fix.
