# MXFP4 2CTA Bottleneck Analysis

## Short Answer

The 2CTA MXFP4 kernel is still latency-bound, not bandwidth-bound and not tensor-core-saturation-bound.

Before the accumulator-barrier refactor, the dominant bottleneck was the per-tile `cute::cluster_sync()` handoff between MMA and epilogue. After replacing that with explicit `acc_full` / `acc_empty` mbarriers, the large cluster-barrier stall dropped substantially, and runtime improved. The remaining bottleneck is now the serialized single-TMEM-accumulator handoff into the epilogue: the epilogue waits for MMA completion, drains TMEM to registers, and the MMA warp cannot reuse the only accumulator buffer until those TMEM loads are fenced.

## Evidence

Focused timing on `CUDA_VISIBLE_DEVICES=4`, shape `M=N=K=4096`:

| Kernel | Event Time | Graph Time |
| --- | ---: | ---: |
| 1CTA | 53.48 us | 53.49 us |
| 2CTA before refactor | ~84-85 us | ~83-84 us |
| 2CTA after refactor | 72.48 us | 70.24 us |

The refactor helped, but the 2CTA kernel is still slower than 1CTA.

Post-refactor NCU report:

`tests/runtime_python/blackwell/sm100_linear_mxfp4/profile/results/mxfp4_4k_2cta_accbar_20260502_1329.ncu-rep`

Key metrics:

| Metric | Value |
| --- | ---: |
| Duration under NCU | 87.58 us |
| Compute throughput | 26.63% |
| DRAM throughput | 2.70% |
| Memory throughput | 17.68% |
| Issue slots busy | 9.46% |
| No eligible warp cycles | 89.72% |
| Active warps / scheduler | 2.01 |
| Eligible warps / scheduler | 0.11 |
| Registers / thread | 240 |
| Local spilling requests | 0 |

Warp-stall sampling after the refactor:

| Stall reason | Not-issued samples |
| --- | ---: |
| Long scoreboard | 1896 |
| Barrier | 1206 |
| Wait | 499 |
| Membar | 94 |

Before the refactor, `stall_barrier_not_issued` was about `3409`, with most of the attribution in `cute/arch/cluster_sm90.hpp`. After the refactor it is about `1206`. That confirms the per-tile cluster barrier was a real bottleneck and that the accumulator-barrier handoff removed a large part of it.

## What Is Happening Now

The 2CTA kernel uses a 2-SM UMMA instruction, but it still has only one physical TMEM accumulator stage:

```cpp
static constexpr int NUM_TMEM_ACC_STAGE = 1;
```

The current post-refactor pipeline is logically:

```text
MMA waits acc_empty
MMA consumes AB/SF stages
MMA writes one TMEM accumulator
MMA signals acc_full
epilogue waits acc_full
epilogue loads four TMEM subtiles into RF
epilogue fences TMEM loads
epilogue signals acc_empty
epilogue converts and TMA-stores output
```

This is correct, but it still serializes MMA reuse of the accumulator behind the epilogue's TMEM drain. Since there is only one accumulator buffer, the next output tile's MMA cannot begin accumulating until the epilogue has loaded the previous accumulator into registers and issued `fence_view_async_tmem_load()`.

That explains the new profile:

- Low memory throughput means DRAM is not the limiter.
- Low compute throughput means the tensor cores are not continuously fed.
- Very low eligible warps per scheduler means the SMs are usually waiting on dependencies.
- Long scoreboard is now the largest sampled stall, pointing at TMEM/global/shared dependency latency rather than raw synchronization overhead.
- Remaining barrier samples are expected from the new fine-grained `acc_full` / `acc_empty` waits and the final setup/deallocation cluster sync, but they are much smaller than before.

## Why 1CTA Still Wins

The 1CTA kernel has less cross-CTA coordination and already uses the accumulator mbarrier handoff. It does not pay the 2CTA peer synchronization cost, and its epilogue/accumulator protocol is simpler. For this shape, the theoretical 2-SM MMA advantage is not enough to overcome:

- single accumulator-stage serialization,
- peer-CTA accumulator handoff overhead,
- epilogue TMEM drain latency,
- low occupancy from register/shared-memory limits.

The 2CTA kernel also has only about two active warps per scheduler, so it has limited ability to hide those waits.

## Current Bottleneck

The current bottleneck is:

```text
single-stage TMEM accumulator reuse + epilogue TMEM-drain latency
```

More concretely, the slow path is the dependency chain from 2CTA UMMA completion to epilogue TMEM loads to `acc_empty` release. The kernel is waiting for data and synchronization dependencies, not for DRAM bandwidth.

## Recommended Next Changes

1. Reduce epilogue live range and register pressure.
   The epilogue currently materializes four RF accumulator tensors before conversion/store. Rework it to drain, convert, and store fewer subtiles at a time if that does not delay `acc_empty` too much. The goal is lower register pressure and better scheduling, not changing math.

2. Evaluate a true multi-stage TMEM accumulator.
   Add a second accumulator stage only if TMEM capacity permits it alongside SFA/SFB TMEM fragments. This would allow MMA for tile `N+1` to overlap with epilogue work for tile `N`.

3. Move or relax per-tile store waiting.
   The epilogue currently does a final `tma_store_wait<0>()` per tile. If safe, make stores tail-waited or more deeply pipelined so output store latency overlaps with future work.

4. Keep 1CTA as the production choice for this shape until 2CTA beats it.
   The refactor improved 2CTA, but measured 1CTA remains faster for `4096^3`.

## Bottom Line

The original issue was real: the 2CTA kernel was dominated by coarse cluster-wide barriers. That part is improved. The next limiter is the fine-grained accumulator/epilogue dependency chain around a single TMEM accumulator buffer. To make 2CTA competitive, the next step is to create overlap between MMA and epilogue or reduce the epilogue latency enough that the single-stage handoff is no longer exposed.
