# Aligned-test-mode verification checklist → THEN relaunch ferret loops

User directive 2026-08-06: "make sure the remote ferret's test mode's perf aligns with the
persistent kernel design, then continue the ferret kernel agent — otherwise it doesn't produce
anything useful for MPK."

## Gate: do NOT relaunch until every box is checked

### A. Verify what the codex alignment job shipped (after /var/tmp/alignment/REPORT.md lands)
- [ ] Part-1 answer recorded with numbers (already have interim: w13 −7.4%, w2 −12.2%, path −9.3%
      vs −17.55% standalone; cp −2.9%; step −0.2% profiled). Mechanism (contention counters) named.
- [ ] Fast-math is the SCORING lane: check each `workspace{3,4,6}/build.sh` and/or task.yaml wording.
      ws4 build.sh previously had NO fast-math in default lane.
- [ ] ALIGNMENT block present in CANONICAL `tasks/{attention-sm100-vllm-beat,moe-fp8-grouped-vllm-beat,dense-fp8-blockscale}.yaml`
      AND in each `workspace<N>/task.yaml` (frozen-snapshot trap!). Verify with grep for the marker
      in BOTH copies, not by reading only canonical.
- [ ] mpk-validator cadence: `ferret/.claude/agents/mpk-validator.md` AND
      `~/.codex/agents/ferret-mpk-validator.toml` both say periodic (every Nth tag / before
      integration-ready), not convergence-only.
- [ ] The in-MPK per-task profile instrument is a reusable one-command script in `ferret/scripts/`
      (capture profiled run at 136w prod geometry → npz → per-type mean table, like
      /var/tmp/alignment/capture_part1.sh + derive). If codex didn't package it, port it myself.

### B. Residual path fixes (mine, deferred to avoid file collisions with the running job)
- [ ] `ferret/scripts/mpk_validate.sh:64` MIRAGE_ROOT default `$HOME/mirage` → `/home/muhengl/mpk-qwen35/mirage`.
- [ ] `~/.codex/agents/ferret-mpk-validator.toml` + `ferret-codex-dispatcher.toml`: stale `~/mirage` defaults.
- [ ] `ferret/.claude/agents/mpk-validator.md` + `codex-dispatcher.md`: same stale refs.
- [ ] Stale `~/mirage` refs in seeds / task.yamls (grep the tree, excluding workspaces' .git).

### C. Relaunch design (1 codex + 2 claude, per user allocation)
- Gating: standalone KERNEL_RESULT = exploration only; integration-ready requires in-MPK per-task
  confirmation at production geometry; fast-math lane scores.
- Target ranking from fresh bs16 cp composition (ON arm, /var/tmp/alignment/part1/on/cp136.json):
  LINEAR_SM100 1035us (23.9% cp) · MOE w13 695 + w2 440 + topk 573 + mul_sum 185 · LINEAR_FP8 377 ·
  ATTN 349 · GDN 318. Fresh candidates: TOPK_SOFTMAX (never optimized), LINEAR_SM100.
  FINAL target choice waits for the combined REPORT + alignment REPORT (bs1/bs8 views differ).
- Seeds must carry: the two-layer transfer law (×0.53 then ×~0—1 by schedule), REGISTER HEADROOM
  ruling (≤250 standalone, 0 spill), SPLIT-KV numeric gate wording where applicable, GPU etiquette.
- Sync workspace task.yaml after ANY canonical edit (tools/sync_task_yaml.py) + verify marker.

### D. In-flight jobs to keep watching
- /var/tmp/alignment/REPORT.md (codex exec pid ~2237192) — Parts 2–4 ship the fixes above.
- /var/tmp/combined/REPORT.md (codex exec pid ~2397433) — four-arm OFF/OFF|ATTN|MOE|BOTH + maybe
  pinned gate. If BOTH wins → pinned gate on combined tree.

### E. QUEUED (blocked on codex quota — resets Aug 7 23:37 EDT / Aug 8 03:37 UTC, or sooner if user swaps in a spare account in ~/.codex/auth.json on BOTH machines)
- [ ] Codex refute-first review of C1/C2/C3 (contention attribution, flag conclusion, test-design
      question): brief at /var/tmp/review_concl/BRIEF.md on box — relaunch:
      cd /var/tmp/review_concl && PATH=/home/muhengl/.nvm/versions/node/v25.9.0/bin:$PATH \
      nohup codex exec -m gpt-5.6-sol -s danger-full-access --skip-git-repo-check -C /var/tmp/review_concl "$(cat BRIEF.md)" > exec.log 2>&1 &
      Until its verdict: contention attribution is UNDER REVIEW (present the two measured facts only).
- [ ] Combined codex session lost model calls mid-gate: the gate DRIVER (bash) keeps collecting into
      /var/tmp/combined/final_gate_run2/ — when it finishes, compute the five-bs table MYSELF from
      its outputs (do not wait for codex to update REPORT.md §4).
- NEW STANDING RULE in force (user 2026-08-06): substantive conclusions get codex cross-review
      before publishing; queue reviews when quota-blocked and mark conclusions "under review".

### F. RELAUNCH BAR RAISED (user, 2026-08-06 second directive) — ALL of the following before ANY driver starts
- [ ] E1 residency experiment REPORT (/var/tmp/e1_residency/REPORT.md, agent in flight): does the
      stock-grid 19.9% W13 delta collapse toward the in-MPK 7.4% at grid=148/136 (one CTA/SM)?
- [ ] Misalignment DECOMPOSITION written with numbers: spill? (ruled out for MoE: 0 spill both arms
      both lanes, production TU) · worker resources? (residency E1 + scheduler co-tenancy 184 blocks
      /148 SMs) · test protocol? (aggregation granularity E2 if E1 leaves unexplained residue;
      homogeneous-only co-tenancy — codex review Q2b) · already-fixed items (toolchain, fast-math
      lane, 128-worker analyzers).
- [ ] HARNESS PROTOCOL CORRECTED per findings (e.g. scored lane runs at grid<=148 one-CTA/SM or
      grid=136, whichever E1 shows predicts in-MPK) — canonical task.yamls + workspace snapshots
      re-synced + seeds updated for ws3-style MoE tasks AND the two new contracts (ws5/ws7 scoring
      configs re-checked against the corrected protocol before launch).
- [ ] Codex refute-first review verdict on C1/C2/C3 + the corrected protocol (quota back
      Aug 7 23:37 EDT; queued at /var/tmp/review_concl/BRIEF.md — extend it with the E1 result
      before firing).
- ws5 (topk r2), ws6 (regfit), ws7 (linear) contracts are READY but PARKED until every box above
  is checked. The 2h cron must NOT start drivers on A+B alone anymore.

### F-update 2026-08-06 ~10:45 — E1 DONE, finding splits by family
- [x] E1 (/var/tmp/e1_residency/REPORT.md): residency is HUGE for the ws3 harness itself
      (stock 3-4 CTAs/SM inflates cand-vs-golden to −58%/−60%; at 136 CTAs 1/SM it collapses to
      −22.3%/−5.5%) — BUT the −19.88%/−13.55% integration claim did NOT come from that harness:
      it came from bitexact_v024.cu at grid=128 + 205,824B dyn smem = ALREADY 1 CTA/SM. So:
      W2: residency-matched standalone (−11.6% @g148) ≈ in-MPK (−12.18%) → W2 CLOSED (residency).
      W13: 1-CTA/SM standalone −20 to −22% vs in-MPK t_live −7.4% → ~13pt RESIDUAL unexplained.
      W2 claim anchor soft (record log predates final header; fresh rerun −18.6/−19.0%).
      Caveat for review: E1 override pins PATH0 at low grids; production takes bulk path (record
      instrument DID use production path at grid=128 → −20%, so caveat doesn't rescue the residual).
- [ ] E6 OVERHEAD-DILUTION probe (dispatched): decompose in-MPK W13/W2 t_live into entry/body/exit
      via phase timestamps, OFF vs ON. Hypothesis to TEST (not conclude): fixed per-task overhead
      dilutes a ~20% body win to −7.4% t_live; if body improves ~20% in-MPK, standalone(1/SM)
      ALIGNS with in-MPK BODY and the protocol fix = compare body-span + model overhead share.
      Also: SM clock capture (sustained megakernel vs bursty standalone) — clock hypothesis.
- [ ] Review brief EXTENDED with E1 + E6 framing (done below) — fire at codex quota reset.

### F-update ~12:00 — E6 DONE (dilution + clocks REFUTED), E7 dispatched
- [x] E6 (/var/tmp/e6_overhead/REPORT.md): overhead-dilution REFUTED (overhead 1.15us = 6% of
      t_live, needed 63%); clocks REFUTED (1965 MHz flat, both regimes); perturbation gate PASS
      (+0.7-0.8% step, tokens byte-identical). NEW STRUCTURE: W13 body delta = −16.08% MEDIAN vs
      −9.01% MEAN → the loss is a TASK-TAIL effect. W2: −13.27% body, no big tail split.
      Note: part1 provably ran the COMBINED tree @13e7e8a8 (MoE impl byte-identical → numbers
      valid; attention flag OFF both arms).
- [ ] E7 (analysis-only, dispatched): paired per-instance deltas on existing rings — size-decile
      test (A: work-size-dependent win) vs overlap test (B: co-tenancy on the tail) + within-id
      vs across-id variance split. Closes the last fork without GPU time.
- Then: write the decomposition, extend review brief with E6+E7, fire codex at quota reset,
      correct harness protocol per verdict, THEN relaunch loops.

### F-update ~12:15 EDT — E7 agent KILLED by CLAUDE SESSION LIMIT (resets 16:50 UTC = 12:50 EDT)
- E7 died mid-analysis (was extending e7_tail.py; partial scripts likely under /var/tmp/e7_pairing
  on the box). NO data lost — inputs are all persisted (E6 rings, part1 npz).
- NEXT TICK AFTER 12:50 EDT: relaunch E7 with the SAME brief (see conversation / re-derive from
  this checklist + /var/tmp/e6_overhead/REPORT.md) PLUS: "partial work may exist under
  /var/tmp/e7_pairing — inspect and continue it, do not restart from scratch blindly."
- Budget ledger: codex exhausted until Aug 7 23:37 EDT; claude session limit until 12:50 EDT today.
  User has been informed of both (standing instruction).

### F-update ~15:10 EDT — E7 DONE: decomposition COMPLETE (one open cause, captures named)
- [x] E7 (/var/tmp/e7_pairing/REPORT.md, verified byte-exact continuation): 7.07pp split =
      1.96pp estimator artifact + 0.12pp size + 4.99pp within-size tail (4.14pp = the +7.4-9.9us
      QUANTUM excursion class on 8.74% of ON instances; needs co-presence, not density; cause
      needs E8 captures: %smid / serialized replay / routed-token dump). W2: cancellation.
- Decomposition ledger updated (.memory/main/standalone-vs-mpk-alignment.md row 6). Review brief
  now 3 addenda (Q1-Q8). REMAINING before relaunch: codex verdict (fires Aug 7 23:37 EDT) →
  protocol correction per Q6/Q8 → re-check ws5/ws7 scoring configs → THEN launch drivers.
- E8 DEFERRED pending review's Q7 answer (is the excursion cause needed for protocol, or moot?).
  GPUs had 5 foreign compute apps at 14:43 — check freeness before any future GPU work.

### tick 16:43-17:00 EDT — waiting window used for the analyzer port
- [x] In-tree analyzer generalization (last diagnosed-unfixed alignment defect): width.py env
      override + sched_gap.py cyclic-rotation group fit; pushed 93d6d55e; validated identical to
      wrappers on real part1 data (48 sched derived, 133+3, PASS, 0ns).
- Box at 16:43: 33 foreign GPU compute apps (external workload) — parked state remains correct.
- Still waiting on: codex quota (Aug 7 23:37 EDT) → review verdict → protocol fix → relaunch.
