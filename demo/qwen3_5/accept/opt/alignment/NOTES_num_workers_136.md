# Megakernel WIDTH is a tunable, and upstream already raised it for B200

Source: upstream PR mirage-project/mirage#743 ("Deepseek v3 support", author bill810975 = this
project's user), head `df63784a`. Pointed at by the user 2026-07-31.

## The one-line change

`python/mirage/utils.py::get_configurations_from_gpu` — the `sm_cnt >= 144` tier:

```
        worker = 128     ->     worker = 136
```

On our 148-SM B200 that is **128 -> 136 workers (+6.25% width)**, and since
`get_scheduler = 4 * (sm_cnt - worker)`, schedulers go **80 -> 48**: it trades 8 scheduler SMs for
8 worker SMs. The PR's own summary pairs it with "updated the num_worker schedule for qwen3 demo,
so the performance and correctness consist" — i.e. worker count and the demo's geometry are coupled
and must move together.

## Why it matters here more than +6% suggests

M4-I8's floors are `max(cp_exact, total_task_work / num_workers)`:

| bs | cp_exact | work_bound | binding | vLLM | floor/vLLM |
|---|---|---|---|---|---|
| 1 | 4130.7 | 2039.7 | cp | 3503.0 | 1.179 |
| 8 | 5275.7 | 4280.2 | cp | 4727.0 | 1.116 |
| 16 | 5638.7 | **5976.0** | **work** | 5301.0 | 1.127 |

**bs16 is the ONE batch size where the work bound binds.** Scaling it by 128/136:
- bs16 work bound 5976.0 -> **5624.5**, so the binding term flips to cp (5638.7):
  floor **1.127 -> 1.064x**
- bs1 / bs8 unchanged — cp still binds there, so width buys nothing
- M4-I9's hard cap ("every non-GEMM stage free still leaves 1.030x at bs16") -> **~0.969x**

The qualitative part is the point: the work bound is a SUM, exactly additive, and immune to latency
tricks — which is precisely why BOTH scheduling (M4-I8) and fusion (M4-I9) were refuted at bs16.
Widening the machine is the only lever that moves a work bound. It flips bs16 back to
critical-path-bound, where the levers we already have start working again.

## Adoption notes

- Our `qwen3_5` builder is already width-agnostic: argmax partial buffers are
  `(mbt, pk.num_workers)` and its `grid_dim=(pk.num_workers, 1, 1)`, and `mpk_engine_run.py`
  derives from `get_configurations_from_gpu(0)`. So nothing hardcodes 128 except a stale docstring.
- num_workers CHANGES THE GENERATED CODE (those buffers and that grid), so a kernel dir must not be
  reused across values — the M3-I7 / M3-I9 compile-knob trap again.
- `MPK_NUM_WORKERS` was added to `mpk_engine_run.py`'s adapter (shared by `profile_wave.py`, which
  reuses `MPKOfflineAdapter` verbatim) so an A/B arm NAMES the knob instead of inheriting it.
- Being a floor, 1.064x is a BOUND, not a measurement — more workers can also cost per-worker
  efficiency (smaller scheduler pool, different occupancy). It must be measured, and correctness
  re-checked, before it counts.

## MEASURED 2026-07-31 — the perf is real, the correctness is NOT. Do not adopt yet.

**Perf (bs16, 3 paired reps, shipped fusion config, per-arm kernel dirs):** 128w = 3228.5 / 3243.6 /
3266.3 ms; 136w = 3070.1 / 3068.6 / 3086.5 ms. **-5.27% mean, no crossovers** (max 136 < min 128),
tracking the predicted 5.6% floor movement closely — good evidence the floor model is sound.

**Correctness: 136 workers is BROKEN and must not ship.** `gate_ac3_stable` at bs {1,16}, 2 cold
reps each: fingerprint divergence **0.0%** (both reps share a state_sig, zero quarantines, zero
errors — it is perfectly deterministic) but token divergence **100%**, with
`first_divergent_position = 0` on **9 of 10 prompts**. Decoded, the output is fluent, coherent text
— just a different continuation, and p06 shows one spurious leading token (`.nextSibling`) followed
by the baseline's own text. So this is a real defect, not a tie-flip: a deterministic wrong token
from step 0. (The low 1.6-4.7% "agreement" is a POSITION-SHIFT artifact of comparing a shifted
sequence — do not read it as incoherence.)

**ROOT CAUSE FOUND — the argmax partition does not tile the vocabulary at 136.** Read straight off
the generated TUs (compile-only, no GPU):

```
  128 workers:  argmax_{partial,reduce}_sm100_kernel<bfloat16, 16, 1940, 128>
  136 workers:  argmax_{partial,reduce}_sm100_kernel<bfloat16, 16, 1825, 136>
```

`padded_vocab_size = 248320 = 970 * 256` (weight_loader.py; it pads the vocab up to a multiple of
256 for `grid_for_rmsnorm_linear_layer`, and 248320 needs none).

- 248320 / 128 = **1940 exactly** -> 1940 * 128 = 248320. Perfect tiling, every id scanned once.
- 248320 / 136 = 1825.88 -> codegen emits **floor 1825** -> 1825 * 136 = **248200**. The last **120
  token ids are never scanned**, and each task's slice offset is shifted against the true row
  stride (the kernel also strides batches by `CHUNK_SIZE * NUM_PARTIAL_TASKS` = 248200 != 248320).

So the defect is structural in the partitioning, which is why the two argmax fixes could not touch
it. THREE hypotheses were tested before this: (a) a degenerate chunk leaving the -1 sentinel —
porting upstream's `local_idx = 0` + row-loop `wg_barrier` changed nothing at 136; (b) chunk-0
reconstruction, predicting the one passing prompt would have its argmax in chunk 0 — REFUTED,
p07-format's first token id is 90700; (c) "vocab 151936 doesn't divide by 136" — I had the wrong
vocab entirely; it is 248320, and the point is not raggedness per se but that 128 tiles it EXACTLY
while 136 does not.

## RESOLVED AND SHIPPED 2026-07-31 — bit-exact, -6.2% at bs16

Fix taken: **the argmax split must DIVIDE padded_vocab_size, not equal num_workers.**
`Qwen35Builder._argmax_split` picks the largest divisor of 248320 that is <= num_workers. At 128 it
returns 128, so the shipped path is untouched BY CONSTRUCTION; at 136 it returns 128 (248320 = 2^9 *
5 * 97, whose largest divisor <= 136 is 128). The argmax is a tiny slice of the step, so
under-using a few workers there costs far less than the width buys everywhere else — much cheaper
than padding the vocab to lcm(256,136)=8704 and padding lm_head to match.

Validated end to end:
- `gate_ac3_stable` @136, bs {1,16}, 2 cold reps each: rc=0, STABLE, **0.0% token divergence**,
  byte-identical to `results/dumps_final`. (Before the fix: 100% divergence.)
- bs16 perf, 3 paired reps, shipped fusion config: 128w 3239.1/3253.1/3275.8ms vs 136w
  3038.7/3051.6/3070.7ms = **-6.21%**, no crossovers, **tokens byte-identical across arms**.

Shipped as `56a8eaa8` (divisor-safe split) + `ee300d5e` (136 default). The default was put in the
qwen3_5 ENGINE, not in shared `utils.get_configurations_from_gpu`: that helper serves every model
and widening is only safe once a model's argmax divides its padded vocab. qwen3's builder pairs
num_workers with a hardcoded 153600, which 136 does not divide — flipping the shared helper would
have broken it exactly as 136 broke us. Upstream's PR pairs its own worker bump with a qwen3 builder
change for the same reason ("so the performance and correctness consist").

**Superseded fix directions** (kept for the reasoning): (1) make `padded_vocab_size` a multiple of BOTH 256 and
num_workers — for 136 that is lcm(256,136)=8704, so 8704*29 = 252416 (+4096 entries, which must be
filled -inf so they can never win the argmax, and the lm_head weight padded to match); (2) make the
codegen use ceil with a bounds guard in the scan loop; (3) restrict num_workers to divisors of the
padded vocab (128 is one, 136 is not) — cheapest, but forfeits the 5.27%.
LESSON: when a width knob changes behaviour, read the GENERATED template arguments before theorising
— two of my three hypotheses died on numbers a one-line grep of the TU would have supplied.

**Banked regardless:** the two upstream argmax fixes are genuine latent-bug fixes for us (the
row-loop smem race is live at mbt=16) and were validated clean at 128 workers — AC-3 STABLE,
byte-identical to baseline. Shipped as `ad57af48`, KEEPING our lowest-index tie-break, which the PR
reverts and which M2-I9 added for a real AC-3 mismatch. Worth probing whether that barrier also
bears on M4-I0's open cold-compile nondeterminism residual.
