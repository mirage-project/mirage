# Segmented μGraph compilation

A prototype that compiles selected model regions with Mirage's **ordinary
μGraph compiler** (`KNGraph.superoptimize()` / `KNGraph.compile()`) instead of
lowering them to the MPK task/event graph. The existing MPK task-graph
implementation is left completely untouched and used only as a baseline.

---

## 1. μGraph regions vs. MPK task graphs

|                        | MPK task graph                                                  | Segmented μGraph                                                |
| ---------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Entry point            | `PersistentKernel.compile()`                                     | `KNGraph.compile()` / `KNGraph.superoptimize()`                   |
| Lowering               | KNGraph → `generate_task_graph()` → task/event graph → megakernel | KNGraph → transpiler → a plain `.so` with `execute_mugraph`        |
| Kernels                | Registered custom task kernels (`register_task`, `*.cuh`)        | Transpiler-generated cuBLAS + elementwise kernels                  |
| Execution              | One persistent megakernel, device-side scheduler                  | Ordinary stream launches driven from Python                        |
| Scope                  | Whole model                                                       | One region at a time; everything else stays in PyTorch             |
| Shapes                 | Fixed at task-graph construction                                  | Fixed per compiled region; other shapes fall back to PyTorch       |

`PersistentKernel.compile()` *inherently* calls `generate_task_graph()` and
depends on registered task kernels — that is its architecture, not an
accident. So this prototype does **not** try to make it bypass that step.
Instead it is a separate runner that reuses the same model and weights but
sends chosen regions down the μGraph path. Nothing in
`experiments/segmented_mugraph/` calls `generate_task_graph()`,
`register_task()`, `PersistentKernel.compile()`, or any MPK task registration;
`no_task_graph_guard()` enforces this at run time by patching
`KNGraph.generate_task_graph` into a tripwire and diffing the tree for
newly written `task_graph*.json`.

## 2. Region boundaries

The Qwen3 dense MLP is cut into two regions:

**Region A** — `silu(x @ Wg^T) * (x @ Wu^T)`
inputs: activation `[T, H]`, gate weight, up weight → output `[T, I]`.

**Region B** — `a @ Wd^T` (+ `residual`)
inputs: Region A output, down weight, optionally residual → output `[T, H]`.

The cut sits where the dataflow narrows to a single tensor, so the boundary
costs one `[T, I]` tensor and nothing else. Region A ends after the multiply
rather than after the two GEMMs, so the SiLU and the multiply are compiled
together with the projections instead of round-tripping through PyTorch.

**Residual placement.** Region B takes the residual *only* in the Stage-1
microbenchmark, where it mirrors MPK's `linear_with_residual_layer` so the
three implementations compute the same thing. In the Stage-2 hybrid model
Region B is the **down projection alone**, because `Qwen3MLP.forward` is
`down_proj(act(gate_proj(x)) * up_proj(x))` — the transformer residual is added
by the *decoder layer*, outside the MLP. Folding it into Region B would only be
semantics-preserving if the region also owned the residual bookkeeping, so it
is left exactly where the original model applies it.

## 3. Weight layouts

PyTorch stores `nn.Linear` weights as `[out_features, in_features]`; Mirage's
`matmul(A, B)` wants `B` as `[in, out]`. The runner passes `w.t()` — a strided
**view**, never a per-iteration contiguous copy — and declares the μGraph input
as `dims=(in, out), strides=(1, in)`, which is exactly `w.t()`'s stride pattern
for a row-major `w`. `KNGraph.compile()` asserts the declared shape/stride
against the tensors it is given, and `CompiledRegion.validate()` re-checks every
runtime input against the recorded `TensorSpec` on each call.

Region A's output is column-major (`strides=(1, T)`). Region B declares its
input with those exact strides, so the intermediate flows from A to B with no
repacking.

## 4. Graph reuse across layers

Compiled regions are cached by `RegionKey`, which covers: region kind, token
count, hidden size, intermediate size, dtype, the full shape/stride/dtype spec
of *every* input, GPU compute capability, and compiler options. Weight
*values* are deliberately absent — weights are runtime pointers. Consequently
all 28 structurally identical Qwen3-0.6B layers share **two** compiled graphs
(one per region), which the Stage-2 run confirms: 2 variants compiled, 1736
μGraph calls, 3472 cache hits, 2 misses.

Launches reuse cached scratch/output buffers and call the compiled entry point
directly, so the steady-state path performs no per-call allocation. The
returned tensor aliases a region-owned buffer; `HybridQwen3MLP` clones it (a few
KB at decode) before handing it back to the model.

## 5. Dynamic-shape fallback

There is no symbolic dynamic-shape codegen. Each region is compiled for a fixed
token count, and `HybridQwen3MLP` routes by token count:

* token count in the enabled bucket set → μGraph;
* anything else (prefill, unseen shape) → the original `Qwen3MLP.forward`.

By default only the decode bucket (1 token) is enabled; `--extra-buckets 2,4`
compiles additional fixed buckets. In the Stage-2 run this produced 1736
μGraph calls (28 layers × 62 decode steps) and 56 PyTorch fallback calls
(28 layers × 2 prefills), exactly as intended.

### The `tokens=1` padding workaround

A single-row output has strides `(1, 1)`, where row-major and column-major
coincide. `kn::gemm` (`include/mirage/transpiler/runtime/kernel/matmul.h:99`)
decides layout with `trans_C = (stride_n_C == 1)`, takes the row-major branch,
swaps `m`/`n`, and ends up passing `ldc=1` where cuBLAS requires `ldc >= 2048`.
cuBLAS rejects it with `CUBLAS_STATUS_INVALID_VALUE` and the kernel aborts.

This is a pre-existing Mirage limitation, not something the prototype
introduces. Rather than patch a shared transpiler runtime header, the runner
compiles the next larger bucket (`MIN_TOKENS = 2`) and pads. The cost is nil:
these GEMMs are weight-bandwidth bound, so M=1 and M=2 take the same time.
A proper upstream fix belongs in the transpiler's output-layout assignment (or
in that `trans_C` test) so degenerate single-row tensors get a non-degenerate
leading dimension.

## 6. Limitations and fair-comparison caveats

**The Hopper/Blackwell threadblock transpiler backends were broken; Blackwell is
now partially fixed.** This was the single root cause behind Region A's
fallback, and it is independent of this prototype.

Originally *no* graph containing a `KNCustomizedOp` compiled on sm_90 or
sm_100. Six defects have since been fixed in the Blackwell backend (see
`tests/experiments/test_blackwell_codegen.py` for the regression net):

1. `elect_one_cta` was declared only inside the matmul branch but referenced
   unconditionally by the accumulator/epilogue paths.
2. `tmem_allocator` / `tmem_base_ptr` were allocated conditionally but released
   unconditionally.
3. `kernel_ptr` was only declared when TMA params existed, yet the B200 cluster
   launch always uses it.
4. `cluster_dim` was hardcoded `{4,4,1}` (`threadblock/graph.h:200`) and never
   derived from `grid_dim`; for a grid it does not divide, the cluster launch
   fails *silently* and the output is never written. Now clamped to divide the
   grid.
5. The single-CTA/single-warp selector — which exists so one warp issues the
   tcgen05 MMA — was wrapped around *every* in-loop op, leaving only 32 of the
   required 128 threads running cooperative kernels.
6. `NUM_THREADS`/`CONSUMER_NUM_THREADS` were pinned to one warp group even when
   no warp specialization was emitted, so surplus threads raced on the tile.

Current state, single-tile custom op (`grid=(1,1,1)`, `forloop_range=1`) — the
shape a fused MPK task body has:

| `target_cc` | 80 (Ampere) | 90 (Hopper) | 100 (Blackwell) |
| ----------- | ----------- | ----------- | --------------- |
| nvcc + numerics | **OK**  | still fails | **OK** (max abs err ≈ 0.01) |

Verified correct at `block_dim` 128/256/384, and for an unpipelined multi-CTA
grid. **Still broken:** the pipelined/TMA warp-specialized path and any matmul,
because the TMA input atoms reference guid-suffixed `tiled_mma`/`mma_tiler`
symbols the TB backend never declares (A2/A3) and the CUTLASS 4.2.1 signatures
do not match (A4). Hopper is untouched — same class of defects, no sm_90
hardware here to validate against. Region A's superoptimizer candidates are all
matmul graphs, so they still fall back.

A secondary, independent constraint sits on top of it: the Hopper/Blackwell
backends require `num_threads == num_warp_groups * 128` when `forloop_range > 1`
(`transpiler_tb_blackwell.cc:370`). The search emits 128-thread blocks, so only
`num_warp_groups = 1` satisfies it — yet `KNGraph.superoptimize()` sweeps
`num_warp_groups_list = [2, 3, 4]` for `target_cc >= 90`
(`python/mirage/kernel.py:607`). With 2/3/4 the transpiler returns *empty* code;
with 1 it returns code that then fails nvcc as above. Fixing the sweep alone
would therefore not help — the codegen bug has to be fixed first.

A third, unrelated robustness bug: `Graph::create_customized_op` **segfaults**
when a threadblock graph's output does not pass through `forloop_accum`. Every
no-accumulator variant crashes; inserting a `forloop_accum` (even with
`forloop_range = 1`) fixes it. Under `NDEBUG` this is a hard crash, not an
assertion.

Two further traps make all of this easy to misdiagnose, and both cost time here:

* `TranspilerConfig::num_consumer_wgs`, `num_producer_wgs` and
  `pipeline_stages` have **no default initializers**, and
  `generate_cuda_program` only assigns them when *both* `num_warp_groups` and
  `pipeline_stages` are passed. Call it without them and the Cython stack
  struct holds garbage, which makes Hopper/Blackwell codegen fail spuriously.
* The build is Release, so `NDEBUG` compiles out the transpiler's
  `assert(false && "compiler assertion failure")` guards; a rejected config
  returns empty `code` instead of aborting. `TranspileResult.error_type` exists
  in C++ but is not exposed through Cython, so Python cannot distinguish
  "transpiler rejected the graph" from "nvcc failed".

The runner catches the failure, reports the region as `direct` with an explicit
`fallback_reason`, and compiles the high-level KNGraph instead — never a
hand-written CUDA kernel. Region B *does* superoptimize successfully at 8
tokens (its graph reaches the transpiler by a path that avoids the constraint),
so the fallback is per-region, not global. **The Stage-1 numbers below were
measured with the stock `superoptimize()` and therefore reflect this bug.**

**What the μGraph path actually executes.** The direct lowering is not a fused
kernel: Region A becomes two `cublasGemmStridedBatchedEx` calls plus a SiLU and
a multiply elementwise kernel (4 launches, `max_smem_size = 0`). So the
Stage-1 μGraph numbers measure "cuBLAS + generated elementwise", not a fused
megakernel.

**Stage 1 is apples-to-apples; Stage 2 is not.** Stage 1 runs the same MLP on
identical inputs and weights, each implementation in its own process. Stage 2
compares *runtimes*: `torch` and `hybrid-mugraph` are driven by PyTorch/HF
Python orchestration (per-op launches, HF `DynamicCache`, a Python sampling
loop), while `mpk` runs the entire decode inside one persistent megakernel with
its own scheduler, attention and sampling kernels. MPK's ~13× end-to-end
advantage there is overwhelmingly the runtime, not the MLP kernels — Stage 1
shows the μGraph MLP is actually *faster* than MPK's MLP tasks at this size.

**Other caveats.**
* MPK cannot be split into per-region measurements the way the μGraph path can;
  Stage 1 builds a separate `PersistentKernel` per scope, which charges each one
  a full megakernel launch and inflates the small-region rows.
* `demo.py --max-new-tokens` only caps the PyTorch branch, so the MPK run
  decodes to EOS; token agreement is compared over the overlapping prefix.
* The timed PyTorch baseline is native bf16 `F.linear`. The FP32-accumulated
  helpers are the correctness oracle only — timing them would have measured an
  upcast no real model performs.
* Greedy decoding amplifies small numeric differences, so token agreement is
  *reported*, not asserted exact, beyond the first-step logits check.
* Single GPU, bf16, batch size 1 only. No TP/EP, no MoE, no quantization.
