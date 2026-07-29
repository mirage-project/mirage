# M3-I7 — the M3 integration gate

Everything M3 landed, measured together on one tree, at integrated HEAD `c80ebd68`
(`origin/qwen3-5_support` was at `21381336`; the two commits in between are M3-I11 evidence
and docs — `git diff --stat c80ebd68 21381336` touches no source, no kernel, no builder).

Run in an isolated clone (`~/mpk-qwen35/mirage-i7`) with its own freshly built C++ extension,
because a stale editable `.so` produces a convincing fake regression at every batch size
(`.memory/main/b200-env.md`, STALE-EXTENSION TRAP). Provenance in `raw_meta/provenance.txt`;
per-run device state in `raw_meta/gpu_before.txt`; drivers in `scripts/`.

## 1. Correctness — AC-3 at integrated HEAD

**Unchanged from the adjudicated M2 state, per case, at every batch size.**

- Full sweep at bs 1/2/4/8/16: all **50 (prompt, bs) dumps byte-identical** to the committed
  `results/dumps_final` (`gates/bytediff_dumps.json`).
- The harness verdict is identical to the committed `results/run_report_all_bs.json` field for
  field: the same five waiver-request records, `p06-poem` position 60 at every bs, reference
  top-1 `31000` and top-2 `81316` both at logit 21.0 (margin 0.0), engine argmax `40581` which
  is the reference's own top-3 at 20.875 — the numeric-precision tie M2 adjudicated. Nine of
  ten exact plus that one tie, at all five batch sizes. Nothing new appeared and nothing that
  was passing regressed.
- The capped bs16 build (the pinned benchmark policy) and the newly-capped bs4/bs8 builds are
  also 10/10 byte-identical (`gates/bytediff_cap.json`, `gates/bytediff_bs4.json`,
  `gates/bytediff_bs8.json`), so the admission cap is bit-transparent wherever it is on.

**On the concurrent bs2/p08-science question.** M3-I11 attributed it before this sweep
finished (`opt/m3i11/CAMPAIGN2.md` §6: a 2×2 factorial over fence × pass-size, 3 reps per arm,
120 trajectories all baseline — the fence and `a86b1eb1` are both exonerated, the cause is the
open nondeterminism). This gate is an independent data point and agrees: **3 reps at bs2, all
10/10 byte-identical** (`gates/bytediff_dumps.json`, `gates/bytediff_dumps_bs2_rep{1,2}.json`).
The gate verdict is therefore unconditional on the correctness side. What stays open is
M3-I11's own acceptance — root cause and a ≥5-rep stable demonstration — which is carried to
M4-I0, not closed here.

## 2. Performance

Three geometries, because they answer three different questions. Medians over 3 reps, full
range quoted, per-rep values in `tables/perf_raw.json`. Every rep's device state was audited
before the run against the foreign floor the guard recorded at claim time.

**Discarded reps.** Three, all in the prefill-only arm, all rep 0 of their config: bs1
(9364 MiB resident at start), bs4 (1866 MiB), bs16 (1061 MiB) against a 1032 MiB limit. Those
prefill medians are therefore n=2, not n=3; the full-run arm is n=3 everywhere. The effect is
small and in the conservative direction — bs1's discarded rep read 325.7 ms against the 322.1
ms median, which moves the decode slope by 0.04 % — but it is below the pinned ≥3-rep rule and
is called out rather than averaged in. bs16's full-run arm also has a 6.19 % range (one rep at
16111 ms against 15169/15217); its median is the lower cluster and the cap A/B in §2c, which
was captured in a separate window, reproduces it at 15210 ms.

### 2a. AC-3 geometry, M3-I1's exact shape — how far M3 moved

One wave per process, `msl=132`, the ascending-length reference subsets: the only shape whose
wave wall is comparable to the committed M3-I1 baseline. (`tables/geomA1_ac3_shape.csv`)

| bs | wave ms | range | M3-I1 | speedup | step µs | M3-I1 | decode tok/s | M3-I1 | ×    |
|----|--------:|------:|------:|--------:|--------:|------:|-------------:|------:|-----:|
| 1  | 1049.2  | 3.58% | 1616.9| 1.54×   | 9532    | 15264 | 104.9        | 65.5  | 1.60 |
| 2  | 1084.4  | 1.60% | 1670.2| 1.54×   | 9697    | 15648 | 206.3        | 127.8 | 1.61 |
| 4  | 1123.4  | 0.71% | 1686.5| 1.50×   | 9943    | 15645 | 402.3        | 255.7 | 1.57 |
| 8  | 1359.3  | 0.39% | 2117.0| 1.56×   | 10501   | 18618 | 761.8        | 429.7 | 1.77 |
| 16 | 3305.3  | 1.71% | 4566.5| 1.38×   | 11252 † | 22005 | 1421.9 †     | 681.7 | 2.09 |

† bs16's steady window at this geometry is `[115,116)` with regime (7 live, 0 prefill) — a
two-iteration drain tail, not a bs16 step. Its wave wall (1.38×) is sound; its step µs is not.

### 2b. Pinned 256/1024 benchmark geometry — the M4-binding table

The geometry the binding vLLM baseline was captured at. Decode throughput is the
prefill-subtracted slope `bs·(D_full − D_pre)/(wall_full − wall_pre)`, which is vLLM's own
tokens-÷-decode-window definition. bs16 runs the pinned capped policy; bs1–8 uncapped.
(`tables/geomM_matched_256_1024.csv`)

| bs | e2e s | range | prefill s | decode tok/s | vLLM  | gap  | vLLM e2e | e2e gap |
|----|------:|------:|----------:|-------------:|------:|-----:|---------:|--------:|
| 1  | 10.27 | 0.25% | 0.32      | 102.6        | 285.5 | 2.78×| 3.60     | 2.85×   |
| 2  | 10.81 | 0.57% | 0.62      | 200.4        | 529.8 | 2.64×| 3.89     | 2.78×   |
| 4  | 11.59 | 0.93% | 1.19      | 392.7        | 934.4 | 2.38×| 4.45     | 2.60×   |
| 8  | 14.41 | 0.80% | 2.94      | 712.4        | 1692.5| 2.38×| 4.95     | 2.91×   |
| 16 | 15.22 | 6.19% | 3.06      | 1343.7       | 3018.1| 2.25×| 5.57     | 2.73×   |

The previously recorded matched-geometry gaps (3.84/3.63/3.26/3.36/4.17×) are **not
comparable** and should be retired — see §4.

### 2c. Admission cap, re-validated with per-arm kernels

`--per-request-token-cap` is a compile-time define. Two arms sharing a `--kernel-dir` under
`--reuse-kernel` run one binary — which is what this gate's own first pass did, reporting the
arms identical to 0.05% while the CPU-side admission replay still claimed 203-vs-131
iterations. With a kernel directory per arm (`tables/cap_policy.json`):

| bs | prefill  | decode tok/s | e2e     |
|----|---------:|-------------:|--------:|
| 4  | 1.45× faster | +0.7 %   | +3.9 %  |
| 8  | 1.73× faster | +4.8 %   | +14.0 % |
| 16 | 2.71× faster | +64.6 %  | +86.0 % |

bs16 at the AC-3 geometry: 3307.9 → 1573.1 ms = **2.103×**. Both are larger than M3-I9's
recorded +84.2 % / +14.1 %. Mechanism, from the adapter's own admission replay: uncapped, the
whole `mbt` budget goes to the lowest live slot, so requests prefill nearly serially —
1887 wave iterations at bs16 versus 1279 capped. The same arithmetic predicts bs4 and bs8, and
they measure as predicted. `docs/qwen35/bench-protocol.md` is amended accordingly.

## 3. Re-derived per-stage comparison

`stage/armL_m3i10/comparison_by_stage.csv`, regenerated through the committed pipeline
(`parse_profile.py` → `concurrency.py` → `analyze_armA.py` → `build_comparison_armA.py`) from
the archived profiler buffers. Overall MPK/vLLM step ratio **2.75× / 2.23× / 2.36×** at
bs1/8/16 — consistent with the independently measured e2e gap of 2.2–2.9×, which the previous
basis (4.28/3.86/4.10×) was not.

Targets, ranked by bs1 absolute gap:

| # | stage | ratio bs1/8/16 | was (M3-I10) | gap µs/step bs1 |
|---|-------|----------------|--------------|----------------:|
| 1 | MoE routed GEMM w13 | 6.96 / 2.85 / 2.56 | 7.09 / 2.82 / 2.23 | 2092 |
| 2 | dense projections (fp8) | 2.07 / 1.98 / 2.08 | 2.17 / 2.09 / 2.19 | 1788 |
| 3 | MoE routed GEMM w2 | 4.37 / 2.00 / 2.03 | 4.48 / 2.08 / 1.92 | 1090 |
| 4 | MoE router top-k/softmax | 5.70 / 4.93 / 4.12 | 3.36 / 3.06 / 2.54 | 731 |
| 5 | full attention | 4.50 / 4.63 / 4.98 | 8.09 / 9.15 / 10.15 | 501 |
| 6 | dense (bf16 + lm_head) | 1.37 / 1.43 / 1.44 | 1.29 / 1.40 / 1.43 | 375 |
| 7 | GDN conv1d | 1.93 / 2.13 / 2.06 | 2.06 / 2.19 / 2.17 | 106 |
| 8 | GDN recurrent | 1.35 / 2.70 / 3.14 | 7.44 / 9.12 / 10.67 | 98 |

MPK is **ahead** of vLLM on norms/RoPE/glue (0.28–0.33×), the shared-expert gate (0.59–0.80×),
MoE/shared SiLU-mul (0.71–0.93×) and quantize (0.88–0.96×) — the megakernel's fusion advantage,
unchanged and real.

### What moved, and why

- **GDN recurrent 7.44 → 1.35×** (bs1) is M3-I3's ferret kernel. Real.
- **Attention 8.09 → 4.50×** is partly M3-I6a's pass-size change and partly the corrected
  basis; the two are not separable from this capture alone.
- **MoE router 3.36 → 5.70×** got *worse*. Some of that is the corrected window exposing it;
  M3-I5c also knowingly cost this task +51–61 % to fix a compaction race. Never re-costed →
  new M4 lever.
- Everything else moved by a few percent.

### The basis moved from msl=353 to msl=897, for a structural reason

The MPK column is a one-step measurement inside `schedule_sim`'s steady window. At msl=353
(M3-I10's arm A) **no prefill-free regime wider than five live requests exists at bs8 or
bs16** — the first request retires at iteration 112, the last finishes prefilling at 128 (bs8)
or 256 (bs16). The committed arm-A capture in fact fell through `parse_profile`'s `hi <= lo`
guard into the last-eight-iterations fallback and recorded regime (1 live, 0 prefill, 1 decode,
1 token) with `tokens_per_step = 1` at both — a single-request step labelled bs8 and bs16.

msl=897 gives a genuine full-width 8-live decode step at bs8 (`[170,656)`, 486 iterations) and
12-of-16 at bs16, and it samples the context band (556–896) the vLLM reference was itself
captured at — the argument M3-I10 already accepted for the attention row, applied to every
stage. `scripts/window_plan.py` prints the regime enumeration the choice is derived from and
emits the per-bs `--warm-iters` that centres `parse_profile`'s window on `concurrency.py`'s own
midpoint iteration, so the per-stage wall spans and the step µs they are normalised against
come from the same iterations.

**Anchor QC** (`stage/qc/anchor_qc_summary.json`), run before publication:

| bs | verdict | max_frac_err | window | note |
|----|---------|-------------:|--------|------|
| 1  | PASS | 0.0000 | [288,384) | 1 live, clean |
| 8  | PASS | 0.0000 | [365,461) | 8 live, clean |
| 16 | **FAIL** | 0.4437 | [720,733) | 16 task types mismatched; the replay predicts 1360 iterations, the trace has 1004 |

bs16's per-stage row is retained and flagged, not used for ranking (the rank is on bs1). The
replay-vs-runtime divergence at bs16/msl=897 is registered as an M4 item — it blinds bs16 stage
attribution and nothing else in M3 depended on it.

`opt/m3i10/ferret_targets.json` is regenerated through its own generator with
`--history-key history_m3i10`, so `history_m3i1` survives untouched and the M3-I10 layer it
displaced is preserved beside it. Re-running is byte-identical.

## 4. Two measurement defects found in the inherited basis

Both were load-bearing, both are fixed in-tree.

1. **The recorded "matched 256/1024" numbers were not measured at 256/1024.**
   `mpk_engine_run.py --prompts-file` is not a prompt source — it is read only under
   `--verify-chat-template` (`mpk_engine_run.py:678`). M3-I9 stage 7 passed it alongside
   `--reference`, so the run consumed the AC-3 reference prompts: its committed timings record
   `prompt_ids: ['p06-poem']` and `max_decode_steps: 1255` (= 1280 − 24 − 1) while the analysis
   divided by a hardcoded 1024. Both the prompt length and the token count were wrong, in
   opposite directions. Fixed by `scripts/make_matched_reference.py`, which materialises the
   pinned baseline sampler's own prompts in the shape `--reference` consumes.
2. **The per-stage basis could not express a bs8 or bs16 step** — §3 above.

A third defect was mine, in this gate's first pass: sharing a kernel directory between cap
arms. Recorded here rather than quietly re-run, because the trap generalises to every
compile-time knob and the CPU-side replay in the timings artifact will happily report a
difference that the binary does not have.

## 5. Backlog — every lever terminal

`opt/backlog.json`, closed by `scripts/close_backlog.py` (idempotent, asserts totality):
**6 integrated, 3 rejected-with-evidence, 2 blocked-with-reason**, plus **5 new levers
registered for M4**. One line each:

| # | lever | disposition |
|---|-------|-------------|
| 1 | quantize fuse/widen | **integrated** — I2b row-partition, +18–31 %; now 0.88–0.96× vLLM, nothing left |
| 2 | right-size MoE activation | **integrated** — I8 v1 +9.7–25.1 %; v2a/v2b grid-widen rejected (bs8 regression) |
| 3 | widen narrow task stages | **blocked-with-reason** — largest structural residual; a per-stage programme, not an issue |
| 4 | mbt=16 admission policy | **integrated** — cap 2.10× AC-3 / +86 % e2e at bs16, and now bs4/bs8 too |
| 5 | GDN recurrent throughput | **integrated** — I3 ferret v010, 7.44→1.35× at bs1 |
| 6 | dense fp8 blockscale | **blocked-with-reason** — ferret loop parked at 0.680 by the box quota; nothing landed |
| 7 | attention pass-size | **integrated** — I6a, megakernel-wide spill removed, −48/−51 % wallspan |
| 8 | GDN prefill WY/UT | **rejected-with-evidence** — its precondition (more tokens/iteration) went the other way |
| 9 | prepare_next_batch | **rejected-with-evidence** — ≤0.16 % of the step |
| 10 | MoE dead-task dispatch | **rejected-with-evidence** — 0.3 % of the step |
| 11 | 256/1024 measurement debt | **integrated** — discharged here; two defects found doing it |

## 6. Residual-gap ranking for M4

1. **Prefill throughput** — 20–31 % of the 256/1024 e2e, and at bs8 MPK's prefill alone
   (2.94 s) is 59 % of vLLM's entire end-to-end time. No M3 backlog item covered it; every M3
   measurement was a decode step. Cheapest first move is already measured: extend the
   admission cap to bs4/bs8 (+3.9 % / +14.0 % e2e, AC-3 byte-identical). Then `mbt` for the
   prefill phase — M3-I5b rejected raising it on *decode* evidence, which does not bind here.
2. **MoE routed GEMMs (241/242)** — ranks 1 and 3, together ~3.2 ms of a 9.8 ms bs1 step. The
   bs1-heavy shape is the compile-fixed `BATCH_SIZE=16` → `NUM_M_TILES=1` tensor-core waste the
   ferret MoE task already targets.
3. **Dense fp8 blockscale (279)** — rank 2 and the most batch-independent gap (~1.8 ms at every
   bs). The parked ferret loop resumes on the quota reset.
4. **Graph width** — the reason MPK is 2.2–2.8× off while winning every fused stage. Split-KV
   attention is the first concrete instance.
5. **MoE router top-k/softmax (260)** — worst ratio of any non-trivial stage (4.1–5.7×) and it
   regressed relative to the previous table. Attribute before optimising: I5c knowingly traded
   speed here for correctness and that trade has never been re-costed.

Also registered, not a performance lever: the `schedule_sim`-vs-runtime divergence at
bs16/msl=897 blocks trustworthy bs16 stage attribution.

## Artifacts

- `gates/` — AC-3 run report and every per-case byte diff
- `tables/` — the three geometry tables + the cap A/B + per-rep raw
- `stage/` — window plan, parsed attribution, per-task tables, anchor QC, the comparison join
- `raw_meta/` — per-run logs, per-rep timings/metas, device-state audit, provenance
- profiler buffers (3.0 GB, msl=897 rep0 per bs): `/home/catalyst/mpk-artifacts/m3i7/late_raw/`;
  everything in `stage/` is reproducible from them with `scripts/rederive_stages.sh`
- box-side outputs mirrored at `/home/catalyst/mpk-artifacts/m3i7/box/`
