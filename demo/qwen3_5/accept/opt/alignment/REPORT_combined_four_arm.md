# Combined attention-v024 + MoE-v024 end-to-end report

## Bottom line

The combined runtime result is positive and non-additive: **BOTH beats OFF in all 12 paired
runs**, including all four bs16 rotations.  On the requested M4-I9 geometry, BOTH improves mean
end-to-end throughput by **8.782% / 3.610% / 0.314% at bs1/8/16**.  MoE alone is still negative
at bs16 (-0.855% throughput; 0/4 wins), but a repeatable positive interaction makes BOTH a small
bs16 win (4/4 wins).

The important qualification is a resource-gate failure: the combined production TU compiles at
**255 registers with 96 B spill stores and 96 B spill loads in the deployed fast-math lane**
(255 registers, 232 B stores, 132 B loads without fast math).  The same spill appears in the
ATTN-only TU; MoE alone matches OFF and does not add any further register or spill cost.  Thus the
two features do not create an additional resource collision, but the attention feature rebased on
`ee300d5e` does not meet the required zero-spill gate.

On the pinned 256/1024 gate workload, the combined MPK reaches
**202.5 / 371.4 / 654.5 / 1091.5 / 1786.9 tok/s** versus a protocol-valid two-boot fresh-vLLM
comparator of **288.4 / 537.9 / 946.8 / 1721.0 / 3012.2 tok/s**.  The requested MPK/vLLM ratios
are therefore **0.7021 / 0.6905 / 0.6912 / 0.6342 / 0.5932** at bs1/2/4/8/16.  MPK remains
behind vLLM at every size, although every ratio is higher than the supplied shipped baseline.

## 1. Combined tree and provenance

- Base: `ee300d5e7081253139fb90924ec91645580a6460`.
- Combined tree: `/home/muhengl/mpk-qwen35/mirage-combined-v024`.
- Branch / clean committed HEAD: `combined-attn-moe-v024` / `13e7e8a851312c34ad1260e0a5d31488f640c616`.
- Attention source commit `640f0ae4c74c8f2107698af65303ba85e15d95ca` was cherry-picked as
  `e0c4347e`; MoE source commit `7ba0d62ddff684f89993bb5c31b2087220a571f1` was cherry-picked as
  `13e7e8a8`.
- Combined extension SHA-256:
  `18d62e1afba5b30cec9894887d6f4c2db240e48797202eae032048eeb9546504`.
- Final diff from the shipped base is exactly the six-file union of the two feature changes:
  attention wrapper/header, MoE header plus its bit-exact generator/test, and shared nvcc wiring.

The execution environment mechanically blocked `git worktree add`, including an approved retry.
I therefore used a fresh local shared clone at the requested destination, checked out the exact
base, created the requested branch, initialized submodules at their gitlinks, and performed the
same cherry-picks there.  This is a clean independent Git tree, but technically a shared clone
rather than a registered worktree.

The attention cherry-pick had one conflict in `python/mirage/mpk/persistent_kernel.py`, where the
base's newer M4-I8 nvcc options and the attention flag both touched the option list.  The resolution
retains the base options and adds two independent exact-string controls:

- `MPK_ATTN_SM100_V024_DEVICE_ONLY=1` emits only
  `-DMIRAGE_ENABLE_ATTN_SM100_V024_DEVICE_ONLY=1`.
- `MPK_MOE_FP8_BLOCKSCALE_V024=1` emits only
  `-DMIRAGE_ENABLE_MOE_FP8_BLOCKSCALE_V024=1`.

OFF emits neither macro; ATTN and MOE each emit only their own; BOTH emits both.  The actual sweep
compile transcripts were audited for all four cases.  Both controls remain default OFF.

## 2. Cheap gates on the combined tree

### Correctness and racecheck

| Gate | Default nvcc lane | `-use_fast_math` lane |
|---|---:|---:|
| Attention shipped vs v024, bs1/8/16, `K_SPLITS=1` | PASS; 0 output/K/V-cache mismatches | PASS; 0 output/K/V-cache mismatches |
| MoE v024 bit-exact suite | PASS; 414 arms, 0 failures | PASS; 414 arms, 0 failures |
| Attention racecheck | 0 hazards, 0 errors, 0 warnings | 0 hazards, 0 errors, 0 warnings |
| MoE racecheck | 0 hazards, 0 errors, 0 warnings | 0 hazards, 0 errors, 0 warnings |

Racecheck ran on physical GPU 5 (`GPU-087b0f09-0d22-7908-e4cb-bb49fd81a455`) after three idle
samples and with `CUDA_VISIBLE_DEVICES=5`.

### Full production-TU resources

Every row below compiled the same generated production `test_rank0.cu`
(`sha256 4843e9a459...`) with only the two feature macros varied.

| Arm | nvcc lane | Registers | Stack | Spill stores | Spill loads | Barriers | Zero-spill gate |
|---|---|---:|---:|---:|---:|---:|---:|
| OFF | default | 255 | 112 B | 0 B | 0 B | 16 | PASS |
| OFF | fast math | 254 | 112 B | 0 B | 0 B | 16 | PASS |
| ATTN | default | 255 | 192 B | 232 B | 132 B | 15 | **FAIL** |
| ATTN | fast math | 255 | 160 B | 96 B | 96 B | 15 | **FAIL** |
| MOE | default | 255 | 112 B | 0 B | 0 B | 16 | PASS |
| MOE | fast math | 254 | 112 B | 0 B | 0 B | 16 | PASS |
| **BOTH** | **default** | **255** | **192 B** | **232 B** | **132 B** | **15** | **FAIL** |
| **BOTH** | **fast math** | **255** | **160 B** | **96 B** | **96 B** | **15** | **FAIL** |

**Combined register answer:** 255 in both lanes.  It is legal (`<=255`) but not zero-spill.
BOTH is byte-for-byte equal to ATTN in this resource accounting, while MOE equals OFF, so there is
no incremental MoE-on-attention resource penalty.

## 3. Four-arm paired measurement

### Protocol

- Established `demo/qwen3_5/accept/opt/profile_wave.py` M4-I9 path.
- Synthetic prompt 256, MSL 353, 96 new tokens, `--mbt 16 --page-size 256 --no-profiler`.
- Physical GPU 5, pinned with `CUDA_VISIBLE_DEVICES=5`; three stable-idle samples and no compute PID
  before every process.
- Four balanced orders at every batch size (each arm occupies every position exactly once):

| Rep | Position 1 | Position 2 | Position 3 | Position 4 |
|---:|---|---|---|---|
| 0 | OFF | ATTN | BOTH | MOE |
| 1 | ATTN | MOE | OFF | BOTH |
| 2 | MOE | BOTH | ATTN | OFF |
| 3 | BOTH | OFF | MOE | ATTN |

- Seed was `20260730 + 1000*batch_size + rep`, identical across arms within a paired block.
- Twelve generated-kernel directories were used: one distinct directory per `(arm, batch_size)`;
  no directory crossed a feature-knob value.
- After every four-arm block, the four token JSON files were compared byte-for-byte before its
  latency was retained.

### Per-rep results

Each arm cell is `wall_ms / generated_tok_s (order position)`.  `Tokens` means the four arm files
are byte-identical for that exact seed.

| BS | Rep | OFF | ATTN only | MOE only | BOTH | Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 733.037 / 130.962 (1) | 691.985 / 138.731 (2) | 745.627 / 128.751 (4) | 673.421 / 142.556 (3) | identical |
| 1 | 1 | 733.577 / 130.866 (3) | 693.216 / 138.485 (1) | 719.272 / 133.468 (2) | 674.976 / 142.227 (4) | identical |
| 1 | 2 | 733.095 / 130.952 (4) | 691.819 / 138.765 (3) | 718.368 / 133.636 (1) | 673.692 / 142.498 (2) | identical |
| 1 | 3 | 733.360 / 130.904 (2) | 692.352 / 138.658 (4) | 719.059 / 133.508 (3) | 674.202 / 142.391 (1) | identical |
| 8 | 0 | 2036.205 / 377.172 (1) | 2000.156 / 383.970 (2) | 2010.109 / 382.069 (4) | 1963.929 / 391.053 (3) | identical |
| 8 | 1 | 1959.743 / 391.888 (3) | 1922.193 / 399.544 (1) | 1939.860 / 395.905 (2) | 1893.633 / 405.570 (4) | identical |
| 8 | 2 | 1993.336 / 385.284 (4) | 1955.785 / 392.681 (3) | 1969.601 / 389.927 (1) | 1923.044 / 399.367 (2) | identical |
| 8 | 3 | 2001.564 / 383.700 (2) | 1971.000 / 389.650 (4) | 1978.531 / 388.167 (3) | 1931.821 / 397.552 (1) | identical |
| 16 | 0 | 3070.946 / 500.172 (1) | 3053.732 / 502.991 (2) | 3098.250 / 495.764 (4) | 3062.390 / 501.569 (3) | identical |
| 16 | 1 | 3106.165 / 494.501 (3) | 3089.093 / 497.233 (1) | 3123.631 / 491.735 (2) | 3087.661 / 497.464 (4) | identical |
| 16 | 2 | 3055.929 / 502.629 (4) | 3037.848 / 505.621 (3) | 3087.727 / 497.453 (1) | 3051.013 / 503.439 (2) | identical |
| 16 | 3 | 3061.720 / 501.679 (2) | 3043.573 / 504.670 (4) | 3091.238 / 496.888 (3) | 3055.160 / 502.756 (1) | identical |

### Means and interaction

Positive interaction is
`BOTH saving - (ATTN-only saving + MOE-only saving)`, in paired wall milliseconds.

| BS | Arm | Mean wall ms | Mean tok/s | Wall reduction vs OFF | Throughput gain vs OFF | Wins vs OFF |
|---:|---|---:|---:|---:|---:|---:|
| 1 | OFF | 733.267 | 130.921 | control | control | control |
| 1 | ATTN | 692.343 | 138.660 | 5.581% | 5.911% | 4/4 |
| 1 | MOE | 725.581 | 132.341 | 1.048% | 1.059% | 3/4 |
| 1 | **BOTH** | **674.073** | **142.418** | **8.073%** | **8.782%** | **4/4** |
| 8 | OFF | 1997.712 | 384.511 | control | control | control |
| 8 | ATTN | 1962.284 | 391.461 | 1.773% | 1.806% | 4/4 |
| 8 | MOE | 1974.525 | 389.017 | 1.161% | 1.174% | 4/4 |
| 8 | **BOTH** | **1928.107** | **398.385** | **3.484%** | **3.610%** | **4/4** |
| 16 | OFF | 3073.690 | 499.745 | control | control | control |
| 16 | ATTN | 3056.061 | 502.629 | 0.574% | 0.577% | 4/4 |
| 16 | MOE | 3100.211 | 495.460 | -0.863% | -0.855% | 0/4 |
| 16 | **BOTH** | **3064.056** | **501.307** | **0.313%** | **0.314%** | **4/4** |

| BS | ATTN saving | MOE saving | Sum of parts | BOTH saving | Interaction | Classification |
|---:|---:|---:|---:|---:|---:|---|
| 1 | +40.924 ms | +7.686 ms | +48.610 ms | +59.195 ms | **+10.585 ms** | **beats sum** |
| 8 | +35.429 ms | +23.187 ms | +58.615 ms | +69.606 ms | **+10.990 ms** | **beats sum** |
| 16 | +17.629 ms | -26.521 ms | -8.893 ms | +9.634 ms | **+18.527 ms** | **beats sum** |

Per-rep interaction was positive in all 12 blocks.  At bs1 it was
`+31.155/+3.935/+3.400/+3.849 ms`; at bs8,
`+10.130/+8.677/+9.006/+16.146 ms`; at bs16,
`+18.646/+18.898/+18.633/+17.931 ms`.

### Crossovers and token identity

- BOTH vs OFF: no crossover at any batch size; BOTH won 12/12.
- ATTN vs OFF: no crossover; ATTN won 12/12.
- MOE vs OFF: one bs1 crossover (MOE lost rep0, then won 3/4), no bs8 crossover
  (won 4/4), and no bs16 crossover (lost 0/4).
- ATTN vs BOTH: BOTH won 4/4 at bs1 and bs8.  At bs16 the two crossed: ATTN was faster in
  3/4 reps, BOTH in 1/4.  This does not change BOTH's 4/4 win over OFF.
- All other MOE-vs-BOTH comparisons favored BOTH in all reps.
- Token identity: **12/12 paired blocks passed**, covering 48 arm outputs.  Each arm generated
  exactly 96/768/1536 token IDs at bs1/8/16.  A second independent `sha256sum` audit found one
  unique token-file hash per four-arm block, and an independent `jq`/`awk` mean calculation matched
  the primary analyzer exactly.

Plain answer on interaction: **BOTH beats the sum of its parts at bs1, bs8, and bs16.**  bs16 is
**not negative in combination**: it is a small but unanimous 4/4 win, even though MoE alone is a
unanimous 0/4 loss.

## 4. Fresh same-session vLLM gate

Because BOTH won all four paired reps at every decision batch size, the five-size pinned gate ran
with both flags ON.  The generated nvcc command in the gate's real `full_bs1_rep0` log contains
both macros, and the committed tree remained clean.

The decode rate below is the gate's authoritative slope,
`bs * (D_full - D_pre) / (wall_full - wall_pre)`, not `tokens / full-arm wall`.  This subtracts the
separately measured two-step prefill/control arm as required by the pinned protocol.

| BS | Combined MPK tok/s | Fresh vLLM tok/s | MPK/vLLM | Supplied shipped ratio | Change | MPK e2e s | vLLM e2e s | E2E factor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 202.465 | 288.371 | **0.7021** | 0.656 | +4.61 pp | 5.269 | 3.567 | 1.477x |
| 2 | 371.390 | 537.890 | **0.6905** | 0.650 | +4.05 pp | 5.847 | 3.822 | 1.530x |
| 4 | 654.455 | 946.845 | **0.6912** | 0.640 | +5.12 pp | 6.844 | 4.398 | 1.556x |
| 8 | 1091.543 | 1721.000 | **0.6342** | 0.586 | +4.82 pp | 8.781 | 4.836 | 1.816x |
| 16 | 1786.852 | 3012.196 | **0.5932** | 0.583 | +1.02 pp | 11.380 | 5.539 | 2.054x |

Relative to the supplied shipped MPK numbers, combined decode tok/s changes by
**+8.10% / +7.99% / +9.35% / +8.04% / +2.04%** at bs1/2/4/8/16.  This comparison uses the
user-supplied prior MPK run; the MPK/vLLM ratios in the table use only the fresh same-window vLLM
measurement.

### Comparator dispersion and escalation

The first fresh-vLLM boot narrowly exceeded its single-boot `(max-min)/median <= 5%` requirement at
bs8 (5.43%) and bs16 (5.15%), so the raw `final.sh` report correctly marked those two comparator
rows not evaluable.  One predeclared retry reproduced the medians but also exceeded the bound at
bs8 (5.02%) and bs16 (6.70%).  No boot was selected or discarded.  The pinned protocol's explicit
two-boot escalation merged all six reps and applies robust IQR/median plus boot-median agreement:

| BS | Boot 1 median tok/s | Boot 2 median tok/s | Merged median | Merged IQR/median | Max boot-median deviation | Binding-valid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 288.378 | 288.364 | 288.371 | 0.020% | 0.003% | yes |
| 2 | 537.897 | 537.884 | 537.890 | 0.418% | 0.001% | yes |
| 4 | 946.824 | 946.866 | 946.845 | 2.511% | 0.002% | yes |
| 8 | 1720.935 | 1721.065 | 1721.000 | 3.729% | 0.004% | yes |
| 16 | 3014.997 | 3003.721 | 3012.196 | 4.056% | 0.281% | yes |

All rows satisfy the merge bounds: IQR/median <=5%, each boot median within 3% of the merged
median, and at least six total reps.  Thus the headline five ratios use every fresh vLLM rep and
are protocol-valid.  The unmodified raw `final.sh` report remains an honest non-binding FAIL: MPK
does not exceed vLLM at any size, AC-5's 1.25x e2e bound fails at every size, and its first-boot
bs8/16 rows remain marked not evaluable.  The two-boot merge resolves only the comparator-validity
question; it does not turn any failed performance criterion into a pass.

### Gate correctness result

Before performance collection, AC-3 passed at all five batch sizes: 15/15 independent cold reps
accepted, 0 quarantines, 0 run errors, 0% state-fingerprint divergence, 0% token divergence, and
150/150 scored cases byte-identical to `results/dumps_final`.

The supplied `--agent-root <old-agent-host>/agent` exists only in the intended remote environment;
it is absent on this host, and the configured `catalyst-B200` alias did not resolve.  The first
attempt therefore failed integrity before GPU work.  The retained second attempt uses the gate's
supported local `--non-binding` mode without an agent root: it passed all remaining integrity
checks but records that the external `.pm/accept.sh` contract could not be cross-checked.  GPU 0
was removed from the fallback candidate list.  Actual generated nvcc output confirms that both
candidate macros reached the gate collection.

## 5. Artifact index

- Four-arm raw driver log: `/var/tmp/combined/sweep.driver.log`
- Four-arm analyzer text/JSON: `/var/tmp/combined/e2e/analysis.txt`,
  `/var/tmp/combined/e2e/analysis.json`
- Per-arm logs and GPU audits: `/var/tmp/combined/e2e/logs/`, `/var/tmp/combined/e2e/audit/`
- Cheap-gate master log: `/var/tmp/combined/run_gates.retry1.log`
- Resource logs: `/var/tmp/combined/gates/full_tu/`
- Gate run: `/var/tmp/combined/final_gate_run2/`
- Raw gate score/report: `/var/tmp/combined/final_gate_run2/perf/perf_score.json`,
  `/var/tmp/combined/final_gate_run2/report.json`
- Two fresh vLLM boots and protocol merge: `/var/tmp/combined/final_gate_run2/perf/vllm_fresh/`,
  `/var/tmp/combined/final_gate_run2/perf/vllm_retry1/`,
  `/var/tmp/combined/final_gate_run2/perf/vllm_merged/`
- Preserved, excluded failed pre-GPU attempt: `/var/tmp/combined/final_gate/`
- Execution plan and failure ledger: `/var/tmp/combined/PLAN.md`

Two orchestration-only failed attempts were corrected before retaining measurements: the initial
standalone MoE compile helper omitted nvcc's established `--expt-relaxed-constexpr`, and the first
idle-drain helper accidentally returned the final loop-test status.  Neither attempt produced a
retained GPU result.  The final-gate agent-root failure likewise occurred in integrity before any
GPU measurement.
