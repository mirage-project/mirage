# Primitive cost microbenchmarks

What the megakernel runtime's synchronization primitives actually cost, measured
rather than assumed.

Every dependency edge in the task graph is one of these operations: a worker
finishing a task bumps an event counter with `atom.add.release.gpu.u64`, the
scheduler bumps each worker's ready counter the same way, both wait by
polling with `ld.acquire.gpu.u64`, and stages hand off through fences and
barriers. The
kernels under test are copied verbatim from
`include/mirage/persistent_kernel/mpk_atoms.cuh`, so these are the instructions
the megakernel issues, not approximations of them.

## Running

```bash
make run                      # print both benchmarks' tables
make run JSON=b200            # also record b200_sync.json and b200_tma.json
make run SM=90a               # build for a different target (default 100a)
```

Plain CUDA binaries with no PyTorch dependency (they link NVML for the
shared-GPU check), so they build against whatever toolkit is installed. The
extension-based harnesses under `tests/runtime_python/blackwell/` currently
cannot build when the CUDA toolkit and PyTorch disagree on version.

**Run it on an idle GPU.** Every run ends with a shared-GPU check, and prints
`WARNING ... PROVISIONAL` if another process was on the device or any SM was
interrupted. Provisional numbers are not comparable with clean ones; see
[Shared GPUs](#shared-gpus).

## Measured: NVIDIA B200, 148 SMs, driver 13020

SM clock measured at runtime: **1.964 GHz**. Clean run: no other process on
the device, ALU sentinel spread 1.0000 and SM clock unchanged at start and
end.

Each primitive is measured on every one of the 148 SMs against each of 8
addresses spread across 1 GiB — 1,184 samples. Atomics and fences fall into
two separated groups, *near* and *far* (see below), reported apart:

| primitive | median | p10 | p90 |
|---|---|---|---|
| `atom.add.release.gpu.u64`, result discarded, near | 250.5 ns | 238.9 | 265.3 |
| `atom.add.release.gpu.u64`, result discarded, far | 432.1 ns | 416.7 | 446.5 |
| `atom.add.release.gpu.u64`, result used, near | 312.5 ns | 295.4 | 334.8 |
| `atom.add.release.gpu.u64`, result used, far | 595.5 ns | 572.8 | 618.3 |
| `__threadfence()` marginal cost, near | 216.6 ns | 205.0 | 231.4 |
| `__threadfence()` marginal cost, far | 396.8 ns | 381.3 | 411.6 |
| `ld.acquire.gpu.u64`, idle | 149.1 ns | 137.6 | 163.7 |
| `ld.acquire.gpu.u64`, 147 SMs writing 6.8 TB/s (89% of HBM peak) | 219.6 ns | 201.6 | 239.9 |
| runtime poll loop, one iteration, 1 queue | 254.5 ns | 241.7 | 267.9 |
| runtime poll loop, one iteration, 2 queues | 255.3 ns | 243.2 | 264.8 |
| `st.relaxed.gpu.u64` (baseline, fire-and-forget) | 0.99 ns | 0.99 | 0.99 |

`__syncthreads()` never leaves the SM, so one SM stands for all of them (7
repetitions, spread under 0.5%):

| threads | 128 | 256 | 512 |
|---|---|---|---|
| `__syncthreads()` | 10.6 ns | 14.7 ns | 22.8 ns |

Contended atomics — one block per SM, all incrementing **one** address, the
shape of a phase barrier. The slowest SM is reported, since that is what a
barrier waits on:

| contending SMs | slowest ns/op | aggregate ops/µs |
|---|---|---|
| 1 | 233.3 | 4.3 |
| 2 | 234.0 | 8.5 |
| 8 | 235.5 | 34.0 |
| 32 | 432.9 | 73.9 |
| 74 | 440.3 | 168.1 |
| 148 | 445.6 | 332.1 |

A second clean run on another B200 in the same node agrees with every median
above to within 2% (its far group is up to 2% faster). The rows for 1–8
contending SMs depend on which SMs those few blocks land on: there they were
far ones, at 433 ns.

## What these say

**The machine is two halves for atomics and fences.** The SMs split into two
fixed groups, and every address is *near* one group and *far* from the other.
A device-scope atomic costs ~250 ns from the near group and ~430 ns from the
far one, with an empty gap between (on the GPU above, nothing between 280 and
409 ns); fences split the same way (~217 vs ~397 ns). The partition is
checked on every run: for each address, the set of near SMs must be exactly
the first address's set or its complement, and all 8 addresses fit. The
group sizes differ from chip to chip: 74 + 74, 72 + 76 and 70 + 78 SMs on
three B200s in the same node. *Interpretation, not tested here:* B200 is two
dies, the groups are each die's SMs (so their sizes vary with which SMs are
disabled on a given chip), and the far case is crossing to the other die's
L2. Either way, a counter's cost depends on where it lives relative to the
SMs that touch it, by 1.7–1.9×, and a single measurement from one block is
one arbitrary side of that.

**Polling costs the same from every SM, but not under heavy HBM traffic.**
Acquire loads do not split: ~149 ns from every SM and address (p10–p90
138–164 ns). With the other 147 SMs writing to HBM at 6.8 TB/s, 89% of its
peak, the same load takes ~220 ns, 47% longer. (An earlier, lighter
generator — one writer warp per SM, through a buffer only twice L2's size —
moved it by 3 ns, which is why an unmeasured load is not evidence.)
One iteration of the runtime's actual poll loop — the acquire load, the move
to the next queue and `__nanosleep(10)` from `persistent_kernel.cuh`, with
its queue state in shared memory as the worker keeps it — takes ~255 ns on an
idle machine, whether the worker polls one queue (one GPU) or rotates between
two (several GPUs). That period is a floor on how soon a waiting worker
notices new work, and a memory-bound neighbour raises it.

**Contention on one counter costs nothing measurable; distance does.** With
1–8 contending SMs, all of which happened to be near this counter, the
slowest pays ~234 ns per atomic — the near number. From 32 SMs up, a far SM
is among them and the slowest pays ~440 ns — the uncontended far number (p90
447 ns). Aggregate throughput grows linearly to ~332 ops/µs. This is the
phase-barrier shape — one thread per SM, each with one atomic in flight at a
time — so it says splitting such a counter to relieve contention is not worth
it; it does not say one address could absorb a much higher offered load.
What such a counter costs is set by where it lives relative to the SMs on its
critical path.

**Fences are expensive relative to stores.** A relaxed store is ~1 ns
fire-and-forget; making it visible device-wide costs ~217 ns (near) or ~397 ns
(far) on top. Any handoff that can prove readiness without a fence —
data-as-flag, where the consumer polls the payload itself — avoids a real
cost, not a notional one.

**Using an atomic's result costs 1.25–1.4× more than discarding it** (near
313 vs 251 ns, far 596 vs 432 ns). Either way `.release` puts a full fence
before every atomic, so consecutive atomics from one thread never overlap;
using the result adds the wait for the returned value. Which number applies
depends on the edge: in `persistent_kernel.cuh` the worker's per-task event
trigger and the schedule-queue enqueue use the result, so every task
completion pays the result-used cost; the scheduler's ready-counter bumps
discard it.

**Intra-CTA barriers are cheap.** `__syncthreads()` runs 11–23 ns for 128–512
threads, an order of magnitude below any device-scope operation. Synchronizing
inside a task is close to free; crossing tasks is not.

## Method

**Every SM, several addresses.** Device-scope costs depend on the issuing SM
and the address, so a single block's number is one arbitrary sample. The
benchmark forces exactly one block per SM (each block requests more than half
an SM's shared memory) and has the blocks take turns on a ticket, so each SM
measures with the rest of the machine idle. Waiting blocks back off with
`__nanosleep(2000)`; without a backoff their ticket polling loads the memory
system enough to inflate the SM being measured several-fold, and quadrupling
it changes no median by more than 0.3%. Each run checks, via `%smid`, that
every SM was measured exactly once. This is repeated for 8 addresses 128 MiB
apart. Each SM's sample is the median of 3 repetitions.

**Near and far are found, not assumed.** The pooled samples are split at their
widest relative gap if that gap is at least 1.25× and leaves at least 5% of
the samples on each side; otherwise they are reported as one group. Reporting
one median over two separated groups would be misleading: with half the
samples in each, it is just the fastest far sample, and it flips between
groups with a single sample. Compare medians, p10s and p90s between runs; the
min and max pick up rare interruptions and do not reproduce.

**Results are consumed or discarded explicitly.** Where an operation's cost
depends on whether its result is read, both numbers are reported; they are
not interchangeable. No artificial dependency is manufactured to force
serialization — threading a result into the next address through `& 0` does
nothing, because the compiler folds it.

**The poll loop is the runtime's.** Beyond the raw acquire load, the benchmark
times the worker's loop from `persistent_kernel.cuh` with `__nanosleep(10)`
and its queue positions and queue ids in shared memory, as the worker keeps
them, since that — not a bare load — is what sets how soon a worker sees new
work. (Keeping that state in registers instead measured ~232 ns.)
Both of its shapes are measured: one queue, as on a single GPU, and rotation
between a local and a remote queue, as when `num_gpus > 1`.

**Timing uses `clock64()`, not `%globaltimer`.** Measured on this B200,
`%globaltimer` only advances every **32 ns**, which cannot resolve a single
operation; `clock64()` has 2-cycle (~1 ns) granularity. Cycles are converted
using the SM clock measured at runtime, not a nominal value. This also bounds
what `profiler.h` can see — it timestamps with `%globaltimer_lo`, so it cannot
resolve stage events shorter than 32 ns.

**Work is verified.** The contended case checks the counter equals
`blocks × iterations` or the run fails, so a timing is never reported for
atomics that did not happen. Blocks rendezvous on-device before the timed
window so launch skew is excluded.

**The load is measured, not assumed.** Every lane of each background block
stores 16 bytes, so each warp writes 512 contiguous bytes per step, four
writer warps per SM, through a 2 GiB buffer, 16× the size of L2. Each writer
warp records how many bytes it wrote and over what `%globaltimer` window, and
the run reports the write bandwidth the background reached while the poller
was timed. The buffer size matters: at twice L2's size the same generator
reported more than the HBM peak, because L2 was absorbing the writes; the run
now says so if that happens.

## Shared GPUs

Another process on the same GPU is time-sliced with the benchmark, and
`clock64()` keeps counting while the benchmark is switched out. Every number
inflates — by 20–140% in the runs that exposed this, including purely on-SM
ones — with nothing in the output to show it. So each run checks twice, at
start and end:

- **NVML** lists compute processes on the device other than this one.
- **An ALU sentinel** times a fixed chain of dependent integer instructions on
  every SM. On an idle GPU it takes exactly the same number of cycles
  everywhere (max/min spread 1.0000); an SM that was time-sliced shows up at
  once (spreads of 1.9–5.9 were seen on shared GPUs).
- **The SM clock** is measured again at the end; cycles are converted with the
  start value, so a drift of more than 1% would skew every number.

If any check fails, the run is marked `PROVISIONAL` in the output and in the
JSON (`"provisional": 1`). The checks run only at the start and end, so a
process that comes and goes in between is caught only by the numbers
themselves; and inside a container whose PID namespace differs from the
host's, NVML's process IDs do not match the benchmark's own, so every run is
marked provisional.

## TMA loads (`tma_bandwidth`)

What a TMA load costs as the megakernel issues it. The instruction is copied
from `tasks/hopper/tma_2d.cuh`, which the SM100 linear task uses; the tensor
map is encoded as `tma.cuh`'s `fill_tma_desc` builds the runtime's
descriptors; the mbarrier helpers are `tasks/hopper/barrier.cuh`'s, which
issue the same instructions as the CuTe helpers the linear task calls. That
is a 5-D tile-mode `cp.async.bulk.tensor` into `shared::cluster` memory from
a bf16 tensor map in global memory, 128B swizzle, no L2 promotion, a box 64
elements (128 B) wide.
Tiles walk a row-major matrix 7168 elements wide (the DeepSeek-V3 hidden
size) along K, as the linear task's loader does. Tile heights run from 8 rows
(1 KiB, decode-sized activations) to 256 rows (32 KiB, the TMA box limit).

### Measured: NVIDIA B200, 148 SMs, driver 13020

L2 126 MiB; HBM peak 7,672 GB/s from memory clock × bus width. Clean run; a
second clean run on another B200 in the same node agrees with every median
to within 1%.

Latency of one tile in flight, from issuing the load to the mbarrier wait
returning, on every SM in turn (2,368 samples per row):

| tile | from HBM, median (p10–p90) | from L2, median (p10–p90) |
|---|---|---|
| 8 rows, 1 KiB | 526 ns (511–727) | 179 ns (172–188) |
| 16 rows, 2 KiB | 536 ns (519–762) | 183 ns (175–192) |
| 32 rows, 4 KiB | 590 ns (528–776) | 191 ns (183–199) |
| 64 rows, 8 KiB | 687 ns (549–791) | 208 ns (201–216) |
| 128 rows, 16 KiB | 785 ns (650–824) | 240 ns (233–249) |
| 256 rows, 32 KiB | 843 ns (748–883) | 307 ns (299–315) |

One SM streaming from HBM through a ring of stages with one loader warp,
every SM in turn (GB/s per SM, median; p10–p90 within ±5%):

| tile \ stages | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| 1 KiB | 1.6 | 3.2 | 6.2 | 8.4 |
| 2 KiB | 3.1 | 6.2 | 12.0 | 16.4 |
| 4 KiB | 5.9 | 11.7 | 22.8 | 31.3 |
| 8 KiB | 11.1 | 21.6 | 41.7 | 57.7 |
| 16 KiB | 19.6 | 38.5 | 74.2 | 105.8 |
| 32 KiB | 36.5 | 71.5 | 136.3 | 191.9 |

The same with 4 stages per loader warp and 1, 2 or 4 loader warps:

| tile \ loader warps | 1 | 2 | 4 |
|---|---|---|---|
| 2 KiB | 12.0 | 23.7 | 46.9 |
| 8 KiB | 41.7 | 82.0 | 160.1 |

Aggregate from HBM, every participating SM streaming its own slice of a
4 GiB matrix (GB/s, median of 7; min–max within ±0.6%):

| SMs | 16 KiB × 4 stages | 32 KiB × 6 stages | 2 KiB × 6 stages | 2 KiB × 4 stages × 4 warps |
|---|---|---|---|---|
| 1 | 76.9 | 206.4 | 16.4 | 46.9 |
| 2 | 154.1 | 413.3 | 32.7 | 93.1 |
| 8 | 619.3 | 1,602.5 | 129.7 | 375.6 |
| 32 | 2,331.3 | 5,142.6 | 525.4 | 1,498.5 |
| 74 | 4,641.6 | 6,894.7 | 1,206.8 | 3,176.7 |
| 148 | 6,520.0 (85% of peak) | 6,876.0 (90%) | 2,332.8 (30%) | 5,305.8 (69%) |

### What these say

**One SM's TMA throughput is set by how much it has in flight.** Throughput
grows almost in proportion to the number of stages — 1.6, 3.2, 6.2, 8.4 GB/s
for 1 KiB tiles at 1, 2, 4, 6 stages — because each tile takes ~0.5–0.9 µs to
arrive from HBM and a loader can only have as many tiles moving as it has
stages. Bigger tiles carry more bytes per trip: six 32 KiB stages give one SM
192 GB/s, six 1 KiB stages give 8.4.

**It does not matter how the in-flight tiles are split.** With 4 stages per
warp, one loader warp gets 12.0 GB/s from 2 KiB tiles, two get 23.7 and four
get 46.9 — in proportion to the tiles in flight, as more stages in one warp
give. So there is no per-warp limit in the wait-and-reissue loop; a loader
needs more in flight, not more loaders.

**Small tiles cannot use HBM without a lot of them in flight.** The most any
configuration reached is 6.9 TB/s, 90% of the computed peak, with 32 KiB
tiles and 6 stages (192 KiB in flight per SM) — and 74 SMs already reach it;
all 148 get the same. With 16 KiB tiles and 4 stages (64 KiB in flight) 148
SMs reach 6.5 TB/s, 85%. With 2 KiB tiles and 6 stages (12 KiB in flight)
they reach 2.3 TB/s, 30% — every SM is waiting on latency, not bandwidth —
and 16 tiles in flight per SM bring that to 5.3 TB/s. A memory-bound decode
task loading narrow tiles is limited by its pipeline depth long before HBM.

**From L2, a tile lands in ~180–310 ns**, 3–4× sooner than from HBM, so a
tile another task just wrote or read is much cheaper to load.

### Method

**Data is checked, not assumed.** Before timing, tiles of every height —
aligned, and at an odd row deep in the matrix — are loaded through the same
path and every element is compared with where the 128B swizzle must put it
(the 16-byte chunk `c` of row `r` lands at chunk `c ^ (r & 7)`). Every
element within a tile holds a distinct value, so a misplaced element cannot
pass by coincidence. After that, every latency load checks its tile's first
element; each streaming loader checks the first element of the last tile in
each stage, and that its incrementally advanced coordinates end exactly where
the tile index says they should. The run fails on any mismatch.

**HBM means HBM.** Each HBM measurement reads tiles nobody has read since L2
was flushed, from a matrix 32× larger than L2. The flush reads twice L2's size
through L2, so it also leaves no dirty lines whose write-back would add
traffic to the timed reads. The aggregate is checked against the HBM peak: a
result above it would mean data came from L2, and fails the run.

**Two clocks agree.** Per-SM numbers use `clock64()`. The aggregate uses
`%globaltimer`, which all SMs share, from the first SM starting to the last
finishing, and is cross-checked against CUDA events around the launch; the
event window contains the device window, so it may only be slower (it is, by
up to about 1.5%), never faster.

**The loop is a real loader's.** Stage index, phase and coordinates advance
incrementally, as in the linear task's loader. A first version computed them
with 64-bit division by a runtime stage count, and that alone cut the 2 KiB
aggregate from 2.3 to 0.9 TB/s and made throughput look capped at a fixed time
per tile, independent of tile size. The tensor map's location (global memory,
as MPK passes it, vs. a `__grid_constant__` parameter or a prefetched
descriptor) was checked separately and makes no difference.

## Scope

Synchronization primitives and TMA loads (issue #771). MMA issue rate is not
covered here.
