# M3-I9 — the matched-geometry re-measure (backlog rank 11)

M3-I1's decode gaps (4.36 / 4.15 / 3.65 / 3.94 / 4.43× at bs 1/2/4/8/16) compare **MPK at the
AC-3 geometry** (24–68 input tokens, `max_seq_length = 132`) against **vLLM at 256/1024**. They
are a lower bound, not a comparison. This is the protocol that makes M4's number like-for-like.

It reuses `docs/qwen35/bench-protocol.md` unchanged wherever the rule is engine-agnostic, and
says explicitly where MPK needs a different mechanism for the same *definition*.

## 1. What must be identical (inherited verbatim)

| bench-protocol | what MPK reuses |
|---|---|
| §2 pinned workload | `input_len = 256`, `output_len = 1024`, identical at every batch size. |
| §2 prompt content | the same synthetic ids: uniform from `[0, tokenizer.vocab_size)` = `[0, 248044)`, `seed = 20260726 + batch_size*1000 + rep_index` — **confirmed** against every committed baseline rep's own `seed` field (bs1 20261726/7/8 … bs16 20276726/7/8). Fresh content per rep, per §2. `make_synthetic_prompts.py` reproduces the sampler; `--verify` cross-checks the seed formula and `input_len` against a committed baseline and fails closed. Caveat stated plainly: **the baseline artifacts do not persist prompt ids**, only `output_ids_sha256`, so this establishes the sampler *inputs*, not byte-equal prompts. |
| §3 batch sizes / decoding | `{1,2,4,8,16}`, greedy, exactly `output_len` tokens, no spec-decode (`MPK_SPEC_DECODE` off). |
| §6 warmup / reps / dispersion | 1 discarded warmup rep per batch size, ≥3 measured reps, `(max−min)/median ≤ 5%`, the §6 escalation rule (second boot, merged IQR/median ≤ 5%, boot medians within 3%) on a miss. |
| §6 co-tenant re-check | per rep — and for MPK it is **correctness, not etiquette**: one co-tenant block deadlocks the megakernel (M3-I2a SM-residency law). `opt/gpu_guard_m3i1.sh` semantics, 3 idle samples, exclusive lock. |
| §8 artifact schema | same envelope: env, versions, GPU id, clocks before/after, per-rep array, `dispersion_pct`. |

Three MPK-side fields are added to the envelope, because they are the things that can silently
make the comparison unfair: `max_num_batched_tokens`, `slot_order` /
`per_request_token_cap`, and the `compaction` record (`live_slot_migrations`,
`straddling_slots`) that `mpk_engine_run.py` now writes per wave.

## 2. Geometry changes on the MPK side

- `max_seq_length = 1280` (= 256 + 1024), pinned via `--max-seq-length`, so one compiled kernel
  serves every wave at a batch size (`pinned_max_seq_length` already exists for this).
- `page_size = 256` unchanged ⇒ 5 pages per request, `max_num_pages = 16*5 + 4 = 84`.
- `max_new_tokens = 1024`.
- One compiled kernel per `(mbt, batch_size)`: 10 compiles for the base sweep at
  `mbt = 16`, 1–10 min each (M2 B200 rules). Compile once, `--reuse-kernel` thereafter.
- **`total_num_requests == batch_size`** still, one wave per measurement, so
  `assert_no_rolling_admission` holds. With all 16 prompts the same length there are no
  duplicate padding slots, so the bs16 wave is 16 *distinct* requests — which is what the vLLM
  baseline measures and what the AC-3 wave never was.

## 3. The decode window — the one place the definition needs re-deriving

bench-protocol §5.1 defines vLLM's AC-4 number as

```
decode_window_start = max over requests of first_token_ts    # last request to enter decode
decode_window_end   = min over requests of last_token_ts     # first request to finish
decode_tokens       = batch_size * (output_len - 1)
```

i.e. **the largest interval in which all `batch_size` requests are simultaneously decoding**.
MPK has no per-request timestamps, and — this is the trap — at bs16 that interval is currently
**empty**: M3-I1 measured zero iterations in which all 16 requests decode. Applying §5.1's
formula naively to MPK either divides by a negative number or silently reports a mixed regime as
"decode". Both would be a fabricated comparison.

The engine-agnostic *definition* is preserved by measuring iterations, not timestamps:

```
D = { iterations i : every live slot took exactly 1 token, and n_live == batch_size }
decode_wall_seconds = sum of the wall time of the maximal contiguous run of D
decode_tokens       = batch_size * len(that run)
decode_tokens_per_second = decode_tokens / decode_wall_seconds
```

`D` is exactly §5.1's interval: all `batch_size` requests simultaneously in decode, no request
finished, no request still prefilling. Two rules that make it un-gameable:

1. **`D` is computed by the schedule replay before the run** (`protocol_sim.py`,
   `regimes()["decode_full"]`), not fitted to the trace afterwards, and the predicted iteration
   count is checked against the profiler's own `BEGIN_TASK_GRAPH` count — the same falsifiable
   test M3-I1 used.
2. **If the run reports `decode_full == 0`, the protocol reports NO decode number for that
   configuration.** It reports the wave-level number and the regime histogram, and says so. A
   configuration that never reaches steady decode does not get to quote a steady-decode figure.

Good news, pre-registered: at 256/1024 the replay says bs16 **does** reach steady decode —
a single contiguous run of **175** all-16-live, all-1-token iterations — where the AC-3 wave has
zero. So the matched geometry can quote a decode number at every batch size, and rule 2 above is
a guard rather than a blocker. The rule still has to be written down, because it is what stops
the AC-3-geometry mistake from being repeated silently.

**The wave-level number is reported too, always, at both geometries**, computed identically for
both engines: `batch_size * output_len / e2e_wall` — this is bench-protocol §5.2's AC-5 bracket
and it is the number that carries the prefill/drain cost the decode figure excludes.

## 4. The `tokens_per_s` field must be re-derived, not reused

`mpk_engine_run.py:385` computes `tokens_per_s = len(wave) * max_decode_steps / wall`. At the
AC-3 bs16 wave that is `10 * 107 / 4.5686 = 234.2`: **10 distinct prompts** (six slots are
duplicates and excluded) and **107 steps** (the `max_seq_length` tail, not the 64 reported
tokens). It is a self-consistent wall-clock proxy for a correctness harness, and it is not
comparable to vLLM's 16-request decode throughput. At the 256/1024 geometry there are no
duplicate slots and every request produces exactly `output_len` tokens, so the field becomes
`batch_size * output_len / wall` — the same quantity vLLM reports. The re-measure harness must
assert that (`len(wave) == batch_size` and `max_decode_steps == output_len`) rather than assume
it, and M4 must quote the re-derived field, never M3-I1's 234/240/204/126/65 row.

## 5. Predicted regime split at 256/1024 (pre-registered)

From `protocol_sim.py`, today's admission, `mbt = 16`:

| bs | iterations | prefill | mixed | decode_full | draining | live-slot migrations | straddling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1039 | 16 | 0 | 1023 | 0 | 0 | 0 |
| 2 | 1057 | 16 | 18 | 1005 | 18 | 1 | 1 |
| 4 | 1094 | 16 | 55 | 968 | 55 | 6 | 3 |
| 8 | 1193 | 16 | 154 | 869 | 154 | 28 | 7 |
| 16 | **1887** | 16 | **848** | **175** | **848** | **120** | **15** |

The bs16 row is the whole point of the issue: **45% of the wave is mixed and 45% is draining**,
leaving 9% clean decode, at a geometry whose floor is 1279 iterations. The pathology is *worse*
at 256/1024 than at AC-3 because the harmonic blow-up scales with prompt length — the `j`-th
slot to finish prefill only gets `mbt − j` tokens per iteration, so prefill costs
`≈ input_len · H_16 = 256 · 3.381 = 865` iterations instead of `input_len · bs / mbt = 256`.

Note the last two columns. 15 of the 16 requests at bs16 have their reported output window still
being written when compaction migrates their slot. At the AC-3 geometry only the six duplicate
padding slots were exposed, and they are not reported — at 256/1024 **every request is
reported**, so the same hazard lands squarely on the benchmark. The re-measure must not be run
under a policy with `live_slot_migrations > 0` and reported-window straddles, or its numbers are
produced by a known-corrupting schedule. `per_request_token_cap = 1` takes both columns to zero
(§`README.md`); that is the gating dependency, not an optimisation.

## 6. Predicted MPK decode step at 1280 context (pre-registered, falsifiable)

MPK's decode step should be **nearly context-independent**: 30 of 40 layers are GDN with a
fixed-size recurrent state, and `ATTN_SM100`'s entire wall span is 461 µs of the 22005 µs bs16
step (2.1%). Scaling only the attention term by the context ratio (mean context over the decode
window ≈ 768 vs ≈ 100 at AC-3, ~7.7×) gives

```
step(1280 ctx, bs16) ≈ 22005 + 461*(7.7 − 1) ≈ 25.1 ms      (+14%)
```

with the KV-page walk adding a little more. **Prediction: the bs16 decode step at 256/1024 is
23–28 ms, i.e. the true decode gap is 4.6–5.6× and not more than 6×.** If it lands above 6× the
"gap is a scheduling problem, not a kernel problem" conclusion of M3-I1 needs revisiting, and
the attention pass-size sweep (backlog rank 7, M3-I6a) is re-ranked upward, exactly as backlog
rank 11 said it might be.

## 7. Execution recipe

```bash
# on catalyst-B200, exclusive GPU, one wave per measurement
python3 opt/m3i9/make_synthetic_prompts.py --batch-size 16 --input-len 256 --rep 0 \
    --verify baselines/vllm-0.25.1-20260725/full/bs16.json --out syn256_bs16_rep0.jsonl

bash opt/m3i9/gpu_guard_m3i9.sh 6,0,1 -- python3 accept/mpk_engine_run.py \
    --batch-size 16 --max-seq-length 1280 --max-new-tokens 1024 \
    --mbt 16 --page-size 256 --kernel-dir kernels/bs16_msl1280 --reuse-kernel \
    --prompts-file syn256_bs16_rep0.jsonl --out-dir m3i9/bench256 \
    --per-request-token-cap auto
```

The prompts are dumped as `{"id": ..., "input_ids": [...]}` so the adapter consumes ids directly
and no tokenizer sits between the two engines (bench-protocol §2: only *length* carries
fairness-relevant signal, and here the sampler is shared too).

`--per-request-token-cap auto` is not optional at this geometry: §5's table says 15 of 16
requests otherwise have their reported output window written across a slot migration. Running
the benchmark without it produces numbers from a schedule already known to corrupt.

The vLLM side is **not re-run**: `baselines/vllm-0.25.1-20260725/` is already at this geometry
and is pinned. If it is re-run for any reason it must be the same commit and the same
`--language-model-only` ruling, or the comparison is void.
