abstraction: how to run completion verdicts cleanly on this box (codex can't read files; commit before verdict).

- [2026-07-25] COMMIT deliverables BEFORE the completion verdict — why: acceptance texts say
  "committed", and the reviewer (correctly) fails uncommitted/untracked deliverables; the
  reviewed unit is branch state, and a post-verdict follow-up commit is normal, not dirty
  (src: M1-I5 r3 FAIL).
- [2026-07-25] Feed --finalize a CLEAN extracted reply, not the raw codex transcript: extract
  everything after the final "tokens used" line (`awk "f{print} /^tokens used$/{getline; f=1}"`);
  the parser now also scans reversed as defense-in-depth — why: transcripts echo the brief,
  which can embed stale verdict text (src: M1-I3 finalize bug).
- [2026-07-25] Inline EVERYTHING the reviewer needs (docs, artifact JSONs, fresh df/nvidia-smi
  captures, log excerpts) — codex on this box cannot shell out; missing inlined evidence = a
  correct fail-closed FAIL that wastes a cycle (src: M1-I5 r1).
- [2026-07-25] Schema-sync reviewers are sharp: a script emitting meta fields the committed
  artifact lacks = out-of-sync deliverable; regenerate with the exact committed script and
  diff token ids (identical → replace; different → numerics alarm, stop) (src: M1-I5 r3).
- [2026-07-25] VERDICT TOKENS DIFFER BY SCRIPT: verify.py parses `VERDICT: PASS|FAIL`;
  milestone_review.py parses `REVIEW: PASS|REVISE` — never override the brief's own format
  line with a generic wrapper instruction; a wrong token fail-closes to REVISE and costs a
  fresh reviewer run (src: M1 milestone finalize).
- [2026-07-26] The milestone reviewer caught the COORDINATOR adjudication error: p08 was
  provisionally waived as a precision tie while its own artifact contained a row-consumption
  contradiction (argmax of dumped logits = ref token, engine emitted another; row 3 computed,
  row 67 consumed) — an implementation-error signature. Rule: before adjudicating any waiver,
  CHECK the artifact for internal contradictions between computed evidence and emitted
  behavior; a waiver is only as good as its consistency (src: M2 milestone review cycle 1).
- [2026-07-27] M1 BitLesson (archived from MAIN): layered independent review earned its
  cost — each altitude caught what the level below missed; 3 of 4 first verdicts failed on
  evidence PACKAGING. Corollary from M3-I8 c1: package the REPORT STRUCTURE too —
  run_report overall_pass=false is the harness's designed unwaived verdict with the M2
  adjudication layered on top; a verdict brief that omits that context earns a fair FAIL.
  Also: verify per-case fields by the M2 report's own schema (the field is `passed`, not
  `status`) before drawing conclusions from an artifact (src: M3-I8 verdict c1).
- [2026-07-30] FINAL-MILESTONE (M4) ISSUES CANNOT BE VERDICTED BEFORE THE GOAL ITSELF PASSES — this
  is BY DESIGN, and my own pending-task list was mis-framed as "8 codex verdicts owed". verify.py
  runs the pinned gate script fresh and treats a nonzero exit as a MECHANICAL FAIL with no reviewer
  discretion; the 2026-07-25 hardening additionally makes exit 3 a hard FAIL for any id starting
  with the final-milestone prefix (`a.id.upper().startswith("M4")`, or a `final-gate: required`
  marker). The gate currently exits 1 (AC-4/AC-5 FAIL: mpk is 0.530–0.608 of fresh vLLM). So
  launching verify.py on M4-I1..I9 today would burn 8 multi-hour gate attempts and write 8 FAIL
  verdicts into the protected proof dir. Those verdicts are BLOCKED ON AC-4, not on paperwork; the
  only way to earn them is to make the gate green. Do not read "no M4 verdicts exist" as neglect.
- [2026-07-30] LATENT FINISH-LINE BLOCKER, found by reading verify.py rather than by running it:
  it invokes the pinned gate with `timeout=900`, but that gate (5 batch sizes × 6 reps, mpk plus a
  fresh vLLM baseline) takes HOURS — the m4status run is the reference. So even at the moment AC-4
  turns green, verify.py would kill the gate at 15 minutes, see rc!=0, and mechanically FAIL every
  M4 issue. The fix must PRESERVE the anti-cheat property ("runs the gate FRESH, never trusts a
  claimed result"): raise or parameterize the timeout, do NOT teach it to accept a cached artifact.
  It is a protected control-plane change — ack'd write + focused test + the selected review policy —
  and it should be reviewed cross-provider precisely because it edits the script that gates my own
  work. Fix it BEFORE the finish line, calmly, not under end-of-milestone pressure.
