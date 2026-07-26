# P8 — prefill-iteration cost vs decode-iteration cost

**Provenance addendum (coordinator-requested fix cycle):** the run below was
originally taken on the shared `~/mpk-qwen35/mirage` clone, which — unrecorded
at the time — had another agent's in-progress edits to shared runtime-path
sources. Re-run clean-room in a verified-empty `git worktree` (detached at
`origin/qwen3-5_support`, same HEAD `79c0073`) in a fresh venv (`venv-mpk2`).
**Clean r = 1.578 vs dirty r = 1.559 (1.25% delta, within this probe's own
~1.4% cross-pair noise) — same band, same pin verdict. The dirty-tree run is
retro-validated, not overturned.** `p8_verdict.json`'s top-level fields now
report the clean run (authoritative per instruction); both runs' full
provenance ({head_sha, dirty file list, venv, gpu}) and the delta are in its
`provenance` key. Clean raw evidence: `p8_raw_result_clean.json`,
`p8_raw_result_xcheck_clean.json`, `p8_verdict_clean_purified.json`,
`clean_run.log`, `clean_setup_excerpt.log`, `clean_head_sha.txt`.

---

Tests v1-architecture.md S8.2's load-bearing assumption `t_pf(16-token prefill
iteration) <= t_dec(16-request decode step)` on the shipped Qwen3-8B
MODE_OFFLINE path (no Qwen3.5 code), per S14 P8. Owner: M2-I11. Prerequisite
gate for M2-I9.

## Construction

`probes/runtime/p8_prefill_iter_cost.py` (adapted from
`tests/ci-tests/run_batch_perf.py`, single request, `use_cutlass_kernel=True`,
B200/SM100) replaces the fixed 1-token `"."` prompt with an exact-length
synthetic prompt (repeated filler text tokenized with
`add_special_tokens=False`, sliced to `input_len`) so `--input-len` can be
swept while `--output-len` (decode length) stays fixed. For a pair
`(lo, hi)` of input lengths at fixed `mbt` and `output_len`:

```
N_pf(L)     = L / mbt                                    (exact; L chosen a multiple of mbt)
t_pf        = [T(hi,128) - T(lo,128)] / (N_pf(hi) - N_pf(lo))
t_dec       = decode-iteration cost "from the same run" (see below)
r           = t_pf / t_dec
```

Run command (exactly the doc's P8 spec): `--model Qwen/Qwen3-8B --mbt 8
--input-len 32 --input-len 512 --output-len 128 --ignore-eos`, on
`catalyst-B200` GPU 7 (claimed via `~/mpk-qwen35/.gpu-locks/M2-I11.lock`,
verified idle before/after via `nvidia-smi`), `venv-mpk`,
`HF_HOME=/raid/catalyst/models` (same cache CI uses; Qwen3-8B already
resident, no download).

## Root-caused correction: which "t_dec from the same run" to trust

The doc's construction says to reuse `run_batch_perf.py`'s own
`latency_ms_per_token` from the low-input-len run as `t_dec`. That field is
`total_time_ms / sequence_length`, i.e. **blended** across every token
position including the prefill ones — and a prefill iteration advances `mbt`
tokens at once, so it contributes only `t_pf/mbt` ms to that per-token average
(far below a real decode token's cost), pulling the blended figure down.

This was caught, not assumed: the (32, 512) pair gave `r_blended = 1.858`.
Because a single `(lo, hi)` pair always exactly fits a 2-parameter linear
model (tautological — it can't tell a good model from a bad one), an
**independent, disjoint pair (64, 256)** was run as a non-tautological check
(GPU compiles here run ~40s, not the 1-10 min worst case, so this cost
~80s). It gave `r_blended = 2.132` — a 27% swing on a quantity that should be
near-constant, i.e. the blended reading is unreliable exactly as suspected.

The purified estimate — back out the pair's own `N_pf(lo)*t_pf` from
`T(lo)` to isolate `t_dec = (T(lo) - N_pf(lo)*t_pf) / output_len` — agrees to
**0.045%** between the two disjoint pairs (`r=1.5589` vs `r=1.5596`), and
`t_pf` itself agrees to 0.53% (6.610 ms vs 6.575 ms). This matches how
`t_dec(B)` is used everywhere else in S8.2 (a pure per-iteration cost, not a
token-blended one) and is what P8's official `r` reports.

| | spec pair (32,512) | independent pair (64,256) | spread |
|---|---|---|---|
| t_pf (ms/prefill-iter) | 6.610 | 6.575 | 0.53% |
| t_dec purified (ms/decode-iter) | 4.240 | 4.216 | 0.57% |
| r purified | 1.5589 | 1.5596 | **0.045%** |
| r blended (rejected) | 1.858 | 2.132 | 14.7% |

## Result

**r = 1.559, band `1.0 < r <= 2.25`, `workload_pin_stands = true`.** The
strict assumption `t_pf <= t_dec` is false (prefill iterations cost ~56% more
wall time than a decode iteration at this mbt=8, single-request config), but
this is comfortably inside the absorption band the doc derives from AC-5's
minimum-acceptable 25% B=16 decode win (`k <= 2.25`). The (256, 1024)
workload pin **stands**, with quantified margin `2.25 - 1.559 = 0.69`.

One nuance worth flagging plainly: the doc's own **point prediction** was
`r <= 1.5` ("the headroom covers the GDN-chunk compute adder Qwen3-8B cannot
exercise"); the measured r=1.559 sits just above that point estimate (by
~4%), i.e. `point_prediction_1.5x_held: false` in the verdict JSON even
though the pass/fail **band** (which is what actually gates the pin) is
comfortably satisfied. Re-run on the real Qwen3.5 graph once M2-I9 stands, as
the doc directs — Qwen3.5's GDN chunk-loop compute (absent from Qwen3-8B) is
expected to push real `t_pf` up further, so this margin (0.69 band-width, or
only 0.04 versus the literal 1.5x point prediction) is not enormous headroom.

## Artifacts

- `probes/runtime/p8_prefill_iter_cost.py` — the probe script (also at
  `~/mpk-qwen35/probes/p8_prefill_iter_cost.py` on B200).
- `probes/runtime/p8_finalize_verdict.py` — deterministic verdict derivation
  from the two raw-result JSONs (no new measurement; pure arithmetic).
- `probes/runtime/p8_verdict.json` — `{r, band, workload_pin_stands}` plus
  full cross-validation and raw evidence, machine-checkable.
- `probes/runtime/p8_raw_result.json` — the doc-mandated (32,512) pair's raw
  per-run numbers.
- `probes/runtime/p8_raw_result_xcheck.json` — the independent (64,256)
  cross-check pair's raw per-run numbers.
- `probes/runtime/p8_run.log` — full stdout (compile + run) for both pairs.
