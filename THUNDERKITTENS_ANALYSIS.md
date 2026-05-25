# What ThunderKittens does that we don't (NVFP4 GEMM)

Comparison reference: TK's `nvfp4_gemm` kernel vs Mirage's `linear_mxfp4_1d2d_sm100_2cta_task_impl` in [linear_mxfp4_1d2d_sm100.cuh](include/mirage/persistent_kernel/tasks/blackwell/linear_mxfp4_1d2d_sm100.cuh).

Concrete measurement (M=N=K=4096, ncu `--set full`, 5 launches each):

| Metric | Mirage 2CTA | FlashInfer | Ratio |
|---|---:|---:|---:|
| Executed instructions | 8.05M | 3.35M | 2.4× |
| Block size | 224 (7 warps) | 192 (6 warps) | 1.16× |
| Grid size | 1024 | 148 | 6.9× |
| Total threads | 229,376 | 28,416 | 8.1× |
| Waves per SM | 6.92 | 1.00 | 6.9× |
| Register spilling | 0 | 0 | — |

The instruction-count gap matches the block-count gap almost exactly. TK uses the same persistent design as FlashInfer.

---

## 1. Persistent grid sized to SM count

```cpp
__host__ inline dim3 grid() const {
    return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
}
```

Grid is `min(num_clusters_needed, num_sms())`. For 4096³ this is `min(512, 148) = 148` blocks.

Mirage's grid is `(BATCH/MMA_M, OUTPUT/MMA_N, 1)` = 1024 blocks for the same shape. We launch 6.9× more thread-blocks than TK, and that ratio explains essentially all of the 2.4× instruction gap (per-tile setup overhead is paid 1024× vs 148×).

## 2. In-kernel tile scheduler (persistent loop)

Every warp's main loop:

```cpp
for (int block_idx = cluster_id; block_idx < num_blocks;
     block_idx += gridDim.x / C::CLUSTER_SIZE) {
    int supergroup_idx        = block_idx / num_blocks_per_supergroup;
    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
    int rows_in_supergroup    = min(C::SUPERGROUP_SIZE,
                                    num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
    int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
    int row_block_idx         = supergroup_idx * C::SUPERGROUP_SIZE
                              + row_within_supergroup;
    int col_block_idx         = idx_within_supergroup / rows_in_supergroup;
    ...
}
```

Three things:

1. **`block_idx += gridDim.x / CLUSTER_SIZE`** — each cluster strides through the global tile space. One cluster handles `ceil(num_blocks / num_clusters)` output tiles back-to-back. Setup costs (TMEM allocator, mbarrier init, TMA descriptor prefetch, cluster handshake) are paid **once per cluster**, not once per tile. **This is the single biggest reason TK has fewer instructions than us.**

2. **Supergroup tile order** — Hilbert-curve-like grouping that walks `SUPERGROUP_SIZE × num_col_blocks` rectangles before stepping rows. Adjacent tiles share more A or B rows → higher L2 / DRAM reuse. Doesn't reduce instructions but reduces memory pressure.

3. **One mbarrier-init per kernel** at the top, before the persistent loop. Mirage re-initializes mbarriers on every block; TK initializes once and reuses the same 64-bit mbarrier objects across all of one cluster's tiles via phase-bit toggling.

## 3. TMA descriptor prefetch (one-time)

```cpp
if (threadIdx.x == 0) {
    g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
    g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
    g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
    g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
    g.D.template prefetch_tma<typename G::D_tile>();
}
```

Once per kernel, not once per tile. Mirage rebuilds the per-stage TMA setup inside every block.

## 4. Producer warp split, even finer than ours

In the producer warpgroup (warps 0–3):
- `warp_id == 3` → A/B tile loader (`tma::cluster::load_async`).
- `warp_id == 2` → scale-factor loader.
- `warp_id == 0` → MMA launcher (only on `cta_id == 0`).
- `warp_id == 1` → idle.

Matches the split we just did (`TMA_WARP_ID`, `SF_TMA_WARP_ID`, `MMA_WARP_ID`). The interesting bit is that the MMA launcher's `inputs_finished[stage]` mbar is **passed as an argument to the MMA op**:

```cpp
mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
        A_sc_tm.template subtile<...>(...),
        B_sc_tm.template subtile<...>(...),
        inputs_finished[stage]);   // <-- hardware completion arrives on this mbar
```

So `tcgen05.mma`'s hardware completion directly signals operand-empty without a separate `umma_arrive` instruction. Trims a few instructions per K-tile but is small relative to the persistent-loop savings.

## 5. Single accumulator, single `outputs_arrived`/`outputs_finished` mbar pair

```cpp
__shared__ semaphore outputs_arrived;
__shared__ semaphore outputs_finished;
```

Per cluster, **one** accumulator-empty / -full mbar pair, used across all the cluster's output tiles via phase-bit cycling (`phasebits` for stage 0).

Mirage: 1CTA uses a `NUM_TMEM_ACC_STAGE`-long array of `acc_full_mbar_ptr` / `acc_empty_mbar_ptr`; 2CTA has no acc mbar at all and relies on `cluster_sync()`. TK's design naturally extends to multi-output-tile-per-block; ours is single-tile-per-block by construction.

## 6. Multi-warp-group "consumer" epilogue with warp-local sync

```cpp
} else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
    ...
    for (int block_idx = cluster_id; block_idx < num_blocks;
         block_idx += gridDim.x / C::CLUSTER_SIZE) {
        wait(outputs_arrived, ...);
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
            warpgroup::load_async(D_reg, out_tm.template subtile<...>(0, ...));
            ...
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
            warpgroup::sync(1);
            warpgroup::store(output_tiles.D[i % NUM_D_TILES], D_reg);
            warpgroup::sync(1);
            warpgroup::tma::store_async<...>(g.D,
                output_tiles.D[i % NUM_D_TILES], {...});
        }
    }
}
```

Two differences from ours:

1. **`warpgroup::sync(1)` instead of a `NamedBarrier`** — a 128-thread warpgroup-local sync, cheaper than the cross-warp named barrier we use (`epilogue_wg_barrier.arrive_and_wait()` × 2 per subtile in the inner loop).

2. **`outputs_finished` signaled once per output tile, after the final TMEM load** — exactly the ThunderKittens "release acc_empty as early as possible" idea, which we just implemented in the 1CTA path. So the per-tile epilogue cost is comparable to ours now, *within* a single tile. The remaining gap is all from the persistent loop.

The `OVERLAP_EPI` template flag controls a sub-tradeoff: when true, each subtile is loaded → `sync(1)` → stored as it's ready; when false, all subtiles are drained first (matching our new structure), then stored in a second loop. TK's benchmark configs flip it to `false` for N ≥ 4096. So at large shapes their epilogue structure is essentially the same as our new one.

## 7. NVFP4 vs MXFP4 — small but real

TK's example uses **NVFP4** (16-element scaling vector, FP8 E4M3 scales) plus a global scale factor `g.A_sc_global * g.B_sc_global` applied as one FP32 multiply during dequant. We're MXFP4 (32-element vector, FP8 UE8M0 scales, no global scale). Their epilogue does an extra `warp::mul(D_reg, D_reg, global_scale)` per subtile that we don't — a tiny instruction *advantage* for us, swamped by the persistent-loop disadvantage.

## 8. Smaller details

- **`pdl::wait()` / `pdl::arrive()`** — Programmatic Dependent Launch handshake at producer entry / consumer exit, allowing two consecutive launches of the same kernel to overlap. Useful for back-to-back GEMMs in a graph, not relevant for our single-launch perf.
- **`cache_policy::EVICT_FIRST`** on the output TMA store — hints L2 to drop the output line right after the store, since nothing else in this kernel reads it back.
- **`tensor_after_thread_sync()` / `tensor_before_thread_sync()`** — TMEM-side fences around the consumer/producer handshake, similar to our `fence_view_async_tmem_load()` placement.
- **Kept the consumer warpgroup at 4 warps × 32 threads = 128 threads.** This matches our `EPI_WARP_COUNT = 4` decision; their epilogue isn't structurally bigger than ours.

---

## Ranked: what closes the gap

1. **Persistent grid + in-kernel tile scheduler.** Single biggest win. Recovers most of the 6.9× block-count gap.
2. **One-shot TMEM allocator and mbarrier init** at start-of-kernel rather than start-of-block.
3. **Supergroup tile ordering** for L2/DRAM reuse. (Memory, not instructions.)
4. **`warpgroup::sync(1)` rather than cross-warp `NamedBarrier`** in the epilogue inner loop.
5. **MMA op consumes the operand-empty mbar directly** instead of a separate `umma_arrive` after the gemm.

Already adopted this session:
- Epilogue early `acc_empty` release after the final TMEM load (1CTA).
- Separate A/B and SFA/SFB TMA producer warps.
- `NO_BIAS` compile-time specialization.
- Phase-A drain-then-fence epilogue structure.

---

## What the next change has to be

To close the instruction-count gap, the lever is a **persistent kernel with an in-kernel tile scheduler**. That's a structural rewrite:

- `block_dim` stays the same; `grid_dim` becomes `(min(num_clusters_needed, num_sms()), 1, 1)`.
- Move all one-shot setup (TMEM alloc, mbar init, TMA prefetch, `cluster_sync` handshake) above a new outer loop.
- Wrap the TMA producer / SF producer / MMA launcher / epilogue in
  `for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / CLUSTER_SIZE) { ... }`.
- Adopt phase-bit cycling on a **single** `acc_full` / `acc_empty` mbar pair (1CTA) or a single `outputs_arrived` / `outputs_finished` pair (2CTA) — reusing the same mbar across all tiles by toggling phase.
- Optionally reorder tiles via a supergroup index for L2 reuse.
- Replace `epilogue_wg_barrier.arrive_and_wait()` calls in the per-subtile loop with warpgroup-local syncs where the participating threads are all in the same 128-thread WG.

Wrapper changes (in [runtime_kernel_wrapper_sm100.cu](tests/runtime_python/blackwell/sm100_linear_mxfp4/runtime_kernel_wrapper_sm100.cu)):
- `dim3 grid_dim(min(NUM_TILES_M * NUM_TILES_N, num_sms()), 1, 1)` for both 1CTA and 2CTA launches.
- `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount, ...)` to size the persistent grid at runtime.

Expected outcome: ~2× instruction-count reduction at M=N=K=4096, putting us within striking distance of FlashInfer.
