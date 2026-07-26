# M3-I1 — decode-step profiling baseline

Where the Qwen3.5-35B-A3B-FP8 megakernel's time actually goes, at batch sizes 1/2/4/8/16, on the
AC-3 workload. Everything here is derived from saved profiler buffers; nothing is estimated.

## The three results that matter

**1. The 4.4–12.9× gap to vLLM was two different things stacked on top of each other.**
Comparing MPK's *steady decode step* against vLLM's decode throughput gives a flat **3.7–4.4×**
at every batch size. The rest of the M2 headline gap is prefill and drain time inside the AC-3
wave, and it is a scheduling problem, not a kernel problem.

| bs | vLLM tok/s | M2 wave tok/s | M2 gap | MPK decode step | MPK decode tok/s | decode gap | gap from prefill+drain |
|---:|-----------:|--------------:|-------:|----------------:|-----------------:|-----------:|-----------------------:|
| 1  | 285.5  | 65.0  | 4.39×  | 15.26 ms | 65.5  | 4.36× | 0.03× |
| 2  | 529.8  | 126.2 | 4.20×  | 15.65 ms | 127.8 | 4.15× | 0.05× |
| 4  | 934.4  | 203.8 | 4.58×  | 15.65 ms | 255.7 | 3.65× | 0.93× |
| 8  | 1692.5 | 240.5 | 7.04×  | 18.62 ms | 429.7 | 3.94× | 3.10× |
| 16 | 3018.1 | 233.9 | 12.90× | 22.01 ms | 681.7 | 4.43× | 8.48× |

Caveat that binds every row: MPK is measured at the AC-3 geometry (24–68 input tokens,
`max_seq_length=132`) and the vLLM baseline at 256/1024. Attention and KV traffic grow with
context, so **3.7–4.4× is a lower bound** on the AC-4 gap. Backlog rank 11.

**2. At bs16 the wave never reaches a decode steady state at all.** An exact replay of
`prepare_next_batch` (`schedule_sim.py`, validated against the trace's own iteration count at all
five batch sizes — 109/109/109/111/203 predicted, 109/109/109/111/203 observed) shows 108 of the
203 iterations are a mixed prefill+decode regime and the other 95 are draining. The first request
retires at iteration 101 while another is still prefilling. `max_num_batched_tokens=16` equals the
batch size, so decode saturates admission and prefill starves one token at a time.

**3. The step has a ~10 ms batch-independent serialization floor.** Worker-idle-while-others-work
is 9.6/9.4/8.8/9.9/9.9 ms at bs 1/2/4/8/16 — flat. It is *not* dispatch stalling: total all-idle
time (no worker in any task) is 0.50 ms at every batch size, and `prepare_next_batch` costs
4–36 µs. It is graph **width**. Stages that emit few tasks occupy few of the 148 SMs for their
whole duration: attention 2 tasks/layer (concurrency 2.0), GDN conv 8 (8.0), norm / gate / MoE
combine 16 (15–16), quantize 96 (34.6).

## The step audit

Per steady decode step, µs. `Σtask/128 + all-idle + worker-idle = step` by construction; the
independent closure test is trace span vs the CUDA-event wall clock, in the last row.

| | bs1 | bs2 | bs4 | bs8 | bs16 |
|---|---:|---:|---:|---:|---:|
| regime (live, prefill, decode, tokens) | 1,0,1,1 | 2,0,2,2 | 4,0,4,4 | 8,0,8,8 | 16,1,15,16 |
| step | 15264 | 15648 | 15645 | 18618 | 22005 |
| Σ task time, all 128 workers | 664324 | 729604 | 816883 | 1048214 | 1483761 |
| Σ task / 128 (perfect packing) | 5195 | 5705 | 6387 | 8194 | 11597 |
| scheduler gap (nobody working) | 501 | 505 | 502 | 499 | 502 |
| — of which `prepare_next_batch` | 4.2 | 6.9 | 11.1 | 20.9 | 35.6 |
| worker idle (some busy, not all) | 9568 | 9438 | 8757 | 9925 | 9906 |
| occupancy | 0.34 | 0.37 | 0.41 | 0.44 | 0.53 |
| mean concurrency (of 128) | 43.5 | 46.5 | 51.9 | 55.2 | 68.3 |
| µs at ≤16 workers busy | 7570 | 8187 | 7528 | 9075 | 6749 |
| **closure: trace span vs CUDA event** | **−0.93%** | **−0.94%** | **−0.94%** | **−0.88%** | **−0.44%** |

The residual closure error is negative and nearly constant because the trace span runs from the
first `BEGIN_TASK_GRAPH` to the last, while the CUDA event brackets the whole launch including
kernel entry and exit.

## Where the step's wall time goes

Summing each task type's *wall span* (the union of the time it is executing) accounts for
109–114% of the step at every batch size — the 40 layers really are sequential, so a stage's wall
span is a fair estimate of the step time it costs. bs1 / bs16, µs:

| task type | layer | wall bs1 | conc bs1 | wall bs16 | conc bs16 |
|---|---|---:|---:|---:|---:|
| `QUANTIZE_FP8_SM100` | quantize | 4540 | 34.6 | 4244 | 33.2 |
| `MOE_W13_FP8_BLOCKSCALE` | MoE w13 | 3084 | 74.7 | 5009 | 89.8 |
| `LINEAR_FP8_BLOCKSCALE` | dense proj | 2936 | 70.2 | 2973 | 69.6 |
| `MOE_W2_FP8_BLOCKSCALE` | MoE w2 | 1702 | 75.1 | 2641 | 86.8 |
| `GDN_RECURRENT_SM100` | GDN recurrent | 1217 | 32.0 | 4979 | 128.0 |
| `LINEAR_SM100` | dense proj | 838 | 88.3 | 921 | 88.9 |
| `MOE_TOPK_SOFTMAX` | router | 565 | 9.0 | 632 | 9.0 |
| `ATTN_SM100` | attention | 513 | 2.0 | 461 | 32.0 |
| `SIGMOID_GATE_MUL_ADD` | GDN gate | 428 | 15.3 | 432 | 15.3 |
| others (silu, combine, conv, norm, argmax, embed) | | ~1300 | | ~1650 | |

Three of the top four are **flat across batch size** — `quantize` 84.1→76.1 ms of worker time,
`LINEAR_FP8_BLOCKSCALE` 86.2→87.0 ms, `LINEAR_SM100` 29.3→31.1 ms from bs1 to bs16. They compute
`max_num_batched_tokens = 16` rows whether or not 16 tokens exist. The MoE is padded the same way:
the GEMM grid is `min(num_experts, mbt*topk) = 128` groups regardless of batch, and the measured
live (>1 µs) expert groups per layer are 56.4 / 59.4 / 60.2 / 70.1 / 86.7 at bs 1/2/4/8/16, against
the 8 that top-8 routing on one token needs. bs1 streams roughly 7× more expert weight than it
uses.

## Backlog

`backlog.json`, ranked by expected decode-throughput delta. Headlines:

1. fuse/widen `quantize_fp8` — +11–42%
2. right-size MoE expert activation to live tokens — +37% at bs1, 0 at bs16
3. widen the narrow task stages — +18–35%
4. fix mbt=16 admission — +44% wave-level at bs16
5. GDN recurrent kernel (M3-I3) — +2.4–7.3%, pays at bs8/16 not bs1
6. dense fp8 GEMM (M3-I4) — +4.2–6.1%

Two entries are explicit rejections: `prepare_next_batch` (≤0.16% of the step) and MoE dead-task
dispatch (0.3%). Two are re-sequencing recommendations: the attention pass-size sweep (M3-I6a) is
worth ≤3.5% at this geometry and should be re-measured at 256/1024 first, and GDN prefill WY/UT
(M3-I6b) should follow the admission fix, because a mixed iteration costs only 1.16–1.30× a decode
iteration — bs16's 2754 ms mixed phase is 108 iterations doing 36 iterations' work.

## Method

- **Capture.** `profile_wave.py` reuses `accept/mpk_engine_run.py`'s `MPKOfflineAdapter` verbatim
  and injects a 48 M-slot profiler buffer by scoped monkeypatch, then detaches
  `mpk.profiler_tensor` so `PersistentKernel.__call__` skips its own exporters. No MPK source is
  modified. One wave per process (HAZARD-WAVE-RESET). Per batch size the prompt set is the first
  `bs` reference prompts by ascending length — exactly wave 0 of a full AC-3 sweep.
- **Export.** The stock exporters walk the buffer one element at a time in Python and hand every
  event to `tg4perfetto`; a full wave is 9.3–24.7 M events, which is unusable. `trace_lib.py` is a
  vectorised numpy reader over the same bytes, including the 32-bit `%globaltimer` unwrap the
  1.6–4.7 s waves require. Perfetto traces are exported for a 3-iteration window only.
- **Validity.** 3 reps per batch size, event counts bit-identical across reps. Profiled vs
  unprofiled wave wall time differs by 2.85–3.59%; dispersion within a set is ≤0.22% (bs16
  profiled 5.27%, one slow rep, median used). Exclusive GPU under a 3-sample idle guard and
  `.gpu-locks/M3-I1.lock`. Zero dangling profiler events at every batch size, so no trace was
  truncated.
- **AC-3 non-regression.** All 25 (prompt, batch size) token sequences from the profiled runs are
  byte-identical to the committed `results/run_report_all_bs.json` at `e51cb86`.

## Files

| file | what |
|---|---|
| `attribution.csv` | the step audit above, all five batch sizes |
| `pertask_by_bs.csv` | per task type: count, µs, wall span, concurrency, live/dead split |
| `layer_type_by_bs.csv` | the same rolled up to layer types |
| `gap_vs_vllm.csv` | the gap re-derivation |
| `backlog.json` | ranked levers with evidence, expected delta, risk, verify command |
| `predictions.md` | predictions written before measurement, with outcomes |
| `tables/bs<N>_attrib.json` | full per-task table + stall structure per batch size |
| `tables/bs<N>_concurrency.json` | concurrency profile + worker gap histogram |
| `tables/bs<N>_iters.csv` | per-iteration timeline with schedule labels |
| `meta/`, `meta_noprof/`, `tokens/` | run metadata and generated token ids |
| `profile_wave.py`, `parse_profile.py`, `trace_lib.py`, `schedule_sim.py`, `concurrency.py`, `analyze.py`, `run_m3i1.sh`, `gpu_guard_m3i1.sh` | the pipeline |

Raw profiler buffers (111–298 MB each) and the Perfetto traces (5.7–8.1 MB each) stay on the B200
box under `~/mpk-qwen35/m3i1/{prof,tables}/` — too large for the repo, and every number here is
reproducible from them with `parse_profile.py`.

## Reproduce

```bash
# on catalyst-B200
cd ~/mpk-qwen35/m3i1
bash opt/gpu_guard_m3i1.sh 6,0,1 -- bash -c "bash opt/run_m3i1.sh prof; bash opt/run_m3i1.sh noprof"
for BS in 1 2 4 8 16; do
  python opt/parse_profile.py --raw prof/raw_bs${BS}_rep0.npz --meta prof/meta_bs${BS}_rep0.json \
      --names prof/task_names.json --out-prefix tables/bs${BS} --warm-iters 2 --steady-iters 60
  python opt/concurrency.py prof/raw_bs${BS}_rep0.npz prof/meta_bs${BS}_rep0.json \
      prof/task_names.json tables/bs${BS}_concurrency.json
done
# then, in the repo
python3 opt/analyze.py
```
