# HANDOFF — M4 state as of 2026-08-07 (agent host being replaced)

This directory preserves the standalone-vs-persistent-kernel ALIGNMENT investigation and the M4
operating state so a new agent system can resume without re-deriving anything. The old agent host
is retired; the B200 box (8×B200, user `muhengl`, project tree `/home/muhengl/mpk-qwen35/`)
persists and still holds all raw artifacts under `/var/tmp/`.

## Where M4 stands

- Goal: beat vLLM decode at bs 1/2/4/8/16 with correctness (AC-3 = coherence + ≥90% top-1
  agreement; byte-exactness is a diagnostic, not the gate).
- Shipped tip on `qwen3-5_support`: `93d6d55e` (analyzer generalization) on top of `ee300d5e`
  (136 workers). Authoritative ee300d5e gate: ratios 0.656/0.650/0.640/0.586/0.583.
- **Combined tree** (attention Stage-1 + MoE v024, each default-OFF): box worktree
  `/home/muhengl/mpk-qwen35/mirage-combined-v024`, branch `combined-attn-moe-v024` @ `13e7e8a8`.
  Four-arm counterbalanced e2e: BOTH beats sum-of-parts at every bs (+8.78/+3.61/+0.31% at
  1/8/16, 12/12 paired wins, bs16 MoE sign-reversal RESCUED by +18.5ms interaction).
  Non-binding fresh-vLLM gate on it: **0.702/0.690/0.691/0.634/0.593** (AC-3 150/150 PASS).
  NOT merged: with attention ON the full TU spills (255 regs/160B stack/96B+96B, vs OFF
  255/112/0/0) — zero-spill gate fails; fix contract is ready (ws6 regfit, below).
- Remaining gap to vLLM ≈ 1.4–1.7×; the binding constraints are system-level (packing/critical
  path), not single-kernel speed. See `NOTES_m4_lever_ranking.md`.

## The alignment decomposition (why standalone kernel wins didn't transfer)

Full ledger: `LEDGER_alignment_decomposition.md`. One-line version per mechanism:
spill — ruled out for MoE (0 spill both arms/lanes) · toolchain — FIXED (unqualified nvcc was
CUDA 13.2/C++17 vs production 12.8/C++20) · fast-math — FIXED (scored lane) · analyzer topology —
FIXED (`93d6d55e`) · CTA/SM residency — QUANTIFIED (harness ran 3-4 CTAs/SM vs MPK's 1; closes W2
entirely) · overhead-dilution — REFUTED (1.15µs = 6%) · clocks — REFUTED (1965MHz flat) ·
W13 residual — DECOMPOSED exactly: 1.96pp estimator artifact + 4.14pp from ONE event class
(8.74% of ON instances jump a +7.4–9.9µs quantum; needs co-presence, not density; cause needs E8
captures: %smid stamps / serialized-stage replay) · layer-2 (in-MPK win → step ≈0 at bs16) —
PROVEN packing/straggler (span +5.6% despite tasks −7.4%; self-concurrency 105→92; flips to
CP-bound). Evidence: `REPORT_alignment_part1-4.md`, `REPORT_e1_residency.md`,
`REPORT_e6_phase_decomposition.md`, `REPORT_e7_paired_tail.md`.

## PENDING: the cross-review, then relaunch (user-ordered sequence)

1. **Codex refute-first review** of the whole chain (claims C1–C3, questions Q1–Q8 incl. the
   corrected scoring protocol design): brief is ON THE BOX at `/var/tmp/review_concl/BRIEF.md`
   (copy: `review_BRIEF_pending.md`). The shared codex account quota resets **Aug 7 23:37 EDT**.
   Fire with:
   `cd /var/tmp/review_concl && PATH=$HOME/.nvm/versions/node/v25.9.0/bin:$PATH nohup codex exec -m gpt-5.6-sol -s danger-full-access --skip-git-repo-check -C /var/tmp/review_concl "$(cat BRIEF.md)" > exec.log 2>&1 &`
2. Correct the harness scoring protocol per its Q6/Q8 verdict (expected shape: score at one
   CTA/SM (grid ≤148 or =136), fast-math lane, production toolchain; quote the transfer model:
   typical-instance ≈0.8× the 1-CTA/SM standalone win, mean further −4pp from the excursion
   class, step-level additionally set by packing; mpk-validator WALL-SPAN confirms per cadence).
3. Re-check the two new loop contracts' scoring configs against that protocol, then RELAUNCH
   **1 codex + 2 claude** loops.
4. E8 (excursion root-cause captures) only if the review says it is load-bearing.

## Parked loop contracts on the box (ready, stop-files in place — do NOT start before step 2)

- **ws5 — TOPK round 2** (`ferret/tasks/moe-router-topk-vllm-beat.yaml`, seed
  `.seed_router_r2.txt`): beat the SHIPPED body ≥20% at bs16, zero register growth (round 1's
  +17 regs leaked ~21% of its own gain — cited in-contract), selection bit-exact + weights ≤2 ULP.
- **ws6 — attention regfit** (`ferret/tasks/attention-sm100-regfit.yaml`, seed
  `.seed_next_regfit.txt`): make the v024 body fit the combined TU at 0 spill/≤112B stack, ≤2%
  latency giveback; acceptance instrument = CPU-only TU probe (fixture:
  `/var/tmp/combined/gates/full_tu/test_rank0.cu`, copy into `workspace6/tu_probe/`).
- **ws7 — LINEAR_SM100** (`ferret/tasks/linear-sm100-vllm-beat.yaml`, seed
  `.seed_next_linear.txt`): 24% of bs16 critical path (GDN gate + router logits + lm_head,
  2280 tasks/step); ≥15% weighted bar; starter kernel.cu = shipped body + cuBLAS reference
  (never compiled — first episode's REPRODUCE job).
- Ferret harness fixes already shipped on the box: fast-math = scored `./kernel` lane,
  `MPK_NVCC` pinned to CUDA 12.8, C++20, ALIGNMENT(HARD) blocks in canonical AND workspace
  task.yamls (frozen-snapshot trap: always sync both), mpk-validator periodic cadence
  (first OPTIMIZE tag / every 5th improvement / always before integration-ready),
  kernel-extractor CHECKPOINT/FINAL modes, `mpk_validate.sh --candidate-cuh` + correct
  MIRAGE_ROOT default, codex agent TOMLs in `~/.codex/agents/` (ferret-* prefix) updated to match.
- Driver: `ferret/chain_longhaul.sh <wsN> <task_yaml> <seed> <log> [wall_h] [stall]`;
  runner branch `FERRET_RUNNER=codex|claude` in `scripts/cc-run.sh`. Remove the workspace's
  `.chain_stop` file before launching. Never edit a RUNNING driver script.

## Standing user directives (verbatim intent, still in force)

- Align the test mode with the persistent-kernel regime BEFORE running kernel loops.
- **Substantive conclusions get a codex cross-review before being presented** (refute-first;
  user: "不要自己直接得出来"). Status facts and direct measurements are exempt.
- Loop allocation: 1 codex + 2 claude sessions for the kernel agent.
- Only use GPUs with no compute apps; NEVER GPU 0; the megakernel spin-waits so co-tenants
  deadlock. Foreign workloads appear on this box — always re-check.
- Never modify the user's own `~/mirage` / `~/ferret` checkouts on the box (read-only).
- Numeric-gate law: no precision-format downgrades, no moved cast positions, no data-dependent
  special-casing, nothing that makes a gate corruption-insensitive.
- Mirage work: commit and push to `git@github.com:bill810975/mirage.git` branch `qwen3-5_support`.
- Codex account is shared box+old-host and quota-limited; user has spare accounts — ask when both
  budgets are exhausted.

## Known latent blocker for M4 completion

`verify.py`'s 900s gate timeout cannot fit the multi-hour pinned gate (`final.sh` full run) —
must be raised/parameterized (preserving run-it-fresh semantics) before any M4 completion claim.
M4 acceptance (AC-4) requires mpk > vLLM at ALL five batch sizes — currently 0.59–0.70.

## Evidence index (this directory)

| file | what |
|---|---|
| REPORT_alignment_part1-4.md | in-MPK OFF/ON discriminating measurement + flag audit + shipped harness fixes |
| REPORT_e1_residency.md | 4-regime grid sweep; provenance inversion (claim was already 1 CTA/SM) |
| REPORT_e6_phase_decomposition.md | entry/body/exit split; dilution+clocks refuted; tail structure |
| REPORT_e7_paired_tail.md | exact 7.07pp decomposition; the +7.4–9.9µs excursion class |
| REPORT_combined_four_arm.md | OFF/ATTN/MOE/BOTH e2e + full-TU resource table (the spill) |
| gate_run2_summary.txt | non-binding five-bs gate on the combined tree vs fresh vLLM |
| review_BRIEF_pending.md | the queued codex review (Q1–Q8) — fire it on the box |
| LEDGER_alignment_decomposition.md | mechanism-by-mechanism ledger (the master summary) |
| NOTES_m4_lever_ranking.md | counterfactual lever table + integration history + corrections |
| NOTES_ferret_harness_lessons.md | accumulated harness/test-design lessons incl. provenance rules |
| NOTES_num_workers_136.md | the 136-worker/argmax-divisor saga |
| NOTES_codex_runner_box.md | how the codex runner branch on the box works |
| NOTES_codex_verify_gate_timeout.md | the 900s verify timeout blocker note |
| STATE_relaunch_checklist.md | the operational checklist the watch loop was following |

Raw data stays on the box: `/var/tmp/{alignment,e1_residency,e6_overhead,e7_pairing,combined,review_concl,moe_v024,attn_s1}`.
