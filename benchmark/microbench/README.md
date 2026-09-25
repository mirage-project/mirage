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
make run                      # print the tables
make run JSON=b200            # also record the numbers in b200_sync.json
make run SM=90a               # build for a different target (default 100a)
```

A plain CUDA binary with no PyTorch dependency (it links NVML for the
shared-GPU check), so it builds against whatever toolkit is installed. The
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

## Scope

Synchronization primitives (issue #771, phase 1). TMA bandwidth and MMA
issue rate are not covered here.
