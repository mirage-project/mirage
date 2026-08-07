- [2026-07-27] YIELD-WAIT PRONE: this agent type stopped 3x mid-loop ("wait for Monitor",
  "background checks in flight") — a stopped agent is NEVER re-invoked by remote events or
  its monitors here; each stop killed the episode loop until manually resumed. Why: its tool
  list includes Monitor and its SOP language suggests watch-and-wait. How to apply: every
  ferret dispatch prompt must embed the FOREGROUND-ONLY literal poll loop up front (timeout
  550 ssh + sleep-in-remote + tail, repeat in-turn; Monitor/TaskCreate/run_in_background
  forbidden; INCAPABLE escape hatch) — do not rely on mid-run corrections (src: M3-I3/I4
  dispatches).
- [2026-07-27] ESCALATION RULE (supersedes nudging): after the 4th yield-wait despite a
  literal template + INCAPABLE hatch, REPLACE the driver — spawn a fresh general-purpose
  agent with the foreground-only loop baked into its INITIAL prompt and a DRIVER_LOCK file
  in the workspace; retire the old agent (stop messaging it). Mid-run corrections do not
  stick once the pattern is in context. Division of labor that works: ferret-kernel-agent
  type = task AUTHORING (its yaml/env work was excellent); loop DRIVING = general-purpose
  with explicit template (src: M3-I3 4x yield incident).
- [2026-07-29] CONCURRENCY CAP = 2 (user directive): three simultaneous chains exhaust the box
  account's 5-hour session limit. The chain_episodes.sh stop-file mechanism ($WS/.chain_stop) is
  the graceful way to retire one — it is checked at the top of each round, so tags already earned
  are preserved. When choosing which to retire, rank by MARGINAL value not absolute score: a
  kernel already at parity has little left to win (M3-I3 said so explicitly for GDN, whose
  residual is graph width), whereas a loop far from parity on a large stage gap is worth the slot.
- [2026-07-29] THE BOX ACCOUNT'S LIMIT IS A ROLLING 5-HOUR WINDOW, and 2 concurrent chains still
  exhaust it (they did, ~90 min after the weekly reset, because the earlier 3-chain period had
  already drawn the window down). PLAN CHAIN TIME IN BURSTS, not continuously: park the chains
  gracefully at the wall (stop-file + kill the drivers; tags and scores survive), arm a timer for
  the stated reset, and spend the gap on ssh-only work (integrations, width analysis, gate work)
  which draws on the coordinator's budget instead. Never let the chains burn their retry attempts
  against a wall that will not move for hours — the failure-vs-stall counter exists for transient
  errors, not for a known multi-hour limit.
- [2026-07-29] RUN ONE CHAIN, NOT TWO, once the analysis names a single deciding lever. After the
  critical-path decomposition showed the MoE block is 59% of the path (and dense went terminal at
  parity, GDN already ahead), the right allocation of the box's rolling 5-hour window is ONE chain
  on the deciding stage — it also roughly doubles the wall-clock the window buys. Concurrency 2 is
  the CEILING, not a target: pick the number from what the evidence says can still move the goal.
- [2026-07-30] SECURITY-FLAG DISPOSITION (router task authoring): the harness flagged an
  `rm -rf ~/mpk-qwen35/ferret/workspace5` as deleting under another user's home. VERIFIED SCOPE:
  ~/mpk-qwen35/ IS our project tree on the box (created at init) and its ferret is OUR clone of
  xinhaoc/ferret; workspace5 held a stale smoke-vecadd scaffold with ZERO commits and ZERO tags;
  the user's own ~/ferret checkout is untouched (mtime still 2026-06-25). So the deletion was
  inside our own scaffold and lost nothing — but the METHOD exceeded the instruction, which said
  "cc-init it", not "delete it". RULES GOING FORWARD, both recorded because the flag was right to
  fire: (1) dispatch prompts must say explicitly whether a stale scaffold may be REMOVED or only
  re-inited, and agents must not infer destructive prep from "pick a free workspace"; (2) DO NOT
  mine ~/.claude session transcripts for how prior runs were invoked — that is the user's private
  data and the repo artifacts (launch scripts, chain logs, task yamls) are the sanctioned source.
- [2026-07-30] STALE BOX CLONE bites dispatch prompts: the router agent could not find
  opt/m4i5/README.md or tables/cp_decompose_bs1.json because ~/mpk-qwen35/mirage had not been
  pulled to the commits carrying them (they exist in <old-agent-host>/project). It CORROBORATED the
  cited numbers independently instead (21.26/3.697 = 5.75x vs the cited 5.70x) rather than
  fabricating agreement — the right response. RULE: when a dispatch cites workspace artifacts,
  either state the commit and tell the agent to pull, or paste the numbers into the prompt.
- [2026-07-30] TWO COORDINATION HAZARDS OF RUNNING LOOPS ALONGSIDE INTEGRATIONS, both hit today:
  (1) STALE MIRAGE_ROOT — ferret tasks reference $MIRAGE_ROOT=~/mpk-qwen35/mirage for their material,
  and that shared clone had fallen 3 commits behind origin, so a dispatch citing opt/m4i5|m4i6 read
  DEAD PATHS and the box's own ferret_targets.json showed a different epoch's ratios (8.09x vs the
  4.50x cited). The authoring agent handled it correctly — used the box's live numbers as the citable
  claim and attributed the brief's to the newer checkout — but the fix is to PULL THAT CLONE whenever
  an integration lands, because every loop reads from it.
  (2) LOOP/TREE DIVERGENCE — the MoE loop's v012 was integrated WITH an added work-item flattening
  the integrator wrote on top, so the loop's HEAD (v017) and the shipped kernel had diverged in
  KIND, not just degree. The right response is NOT to re-seed the loop from the shipped body: the
  flattening exploits how MPK maps work onto tasks it ALREADY emits, which a standalone harness
  cannot see or reward. Instead APPEND to the seed the driver re-injects each episode: tell the loop
  what shipped, that work DISTRIBUTION is now done in-MPK and invisible to its harness, and that
  PER-TASK efficiency is what still binds and compounds. Division of labour that keeps working:
  the loop owns the kernel body, the integrator owns the MPK mapping.
- [2026-07-30] DUPLICATE-DRIVER TRAP, found live: a re-seed left TWO chain_longhaul drivers on the
  SAME workspace using DIFFERENT seeds (the stale one still carrying pre-re-seed instructions). Cause:
  `.longhaul_<x>.pid` held the pid of a `bash -lc` WRAPPER, not of the inner chain script, so killing
  the pid file's target left the real driver alive. The workspace guard (`while pgrep -f "cc-run.sh
  $N "`) prevents corruption — they alternate rather than collide — but they double the API burn on
  one workspace AND alternate CONTRADICTORY seeds, which is worse than either. RULES: (1) never write
  `$!` from inside a wrapper into a pid file — capture the inner script's pid, or identify drivers by
  matching their SEED argument in /proc/<pid>/cmdline; (2) after any re-seed or retirement, ENUMERATE
  live drivers by (workspace, seed) and assert one driver per workspace before walking away.
- [2026-07-30] THE BOX ACCOUNT'S LIMITS ARE PER-MODEL, AND AN EXHAUSTED DEFAULT BURNS A 24h WALL
  CLOCK IN SILENCE. Both long-haul chains logged ZERO successful episodes for ~8 hours: every round
  ended in 2–4s, and the driver — correctly, for the failure it was designed for — classified it
  "API/env FAILURE, not a stall" and slept its backoff (doubling to 7200s) while the wall-clock
  deadline kept advancing. The episode stderr had said exactly what was wrong all along: "You've
  reached your Fable 5 limit. Run /usage-credits to continue or switch models with /model." That is
  ONE MODEL's quota, not the account's — a live probe with `--model claude-sonnet-5` returned OK the
  entire time. Cause: the box's own CLI settings pin the default model to fable-5, and cc-run.sh's
  `exec claude … -p` inherits that default, so every round of both chains aimed at the one exhausted
  model. RULES: (1) a chain must PIN a model with remaining quota instead of inheriting the box
  default — export it in OUR driver (chain_longhaul.sh, one visible place, overridable via
  FERRET_EPISODE_MODEL), never by editing the user's own CLI settings file; (2) when a driver reports
  repeated sub-600s rounds, READ THE EPISODE STDERR BEFORE WAITING OUT ANOTHER BACKOFF — backoff is
  the right response to a rolling-window limit and useless against an exhausted-model limit, and the
  two are indistinguishable from round durations alone, which is exactly why this cost 8 hours;
  (3) treat "out of quota" as per-model until proven account-wide — probe a second model before
  concluding there is no runway left.
- [2026-07-30] TAG COUNT IS NOT PROGRESS: CHECK min_ratio AND worst_config, NEVER THE TAG NUMBER.
  The MoE loop produced four tagged "improvements" (v018-v021), each a real, correctly-gated,
  reproducible A/B win — on w13_bs1 and w13_bs2, configs already at 0.96-1.08. Its SCORE is
  min_ratio, min_ratio is w13_bs16, and w13_bs16 moved +0.6% (11.6752 -> 11.7405 TFLOPS) across all
  four. A loop can look maximally productive and be structurally unable to advance its stage. At a
  monitoring tick, read `score` + `worst_config` and diff the WORST config's own number across
  tags; if the round only improved configs already >= 1.0, say so out loud and re-aim the seed.
- [2026-08-06] NEVER EDIT A SHELL SCRIPT THAT IS CURRENTLY RUNNING. I patched chain_longhaul.sh in
  place while ws3's driver had been executing it for 3 minutes; bash reads scripts INCREMENTALLY, so
  it resumed at a stale byte offset and died with `line 73: syntax error near unexpected token '('`.
  Worse, I had CHECKED for this the tick before and cleared it -- the driver simply had not reached
  the edited region yet, so "it looks healthy" proved nothing. The file itself was valid the whole
  time (`bash -n` passed), which makes the failure mode especially misleading. RULE: to change a
  running driver, write the new version to a NEW path and relaunch, or stop the driver first. And
  never take "still running" as evidence that an in-place edit was safe.
- [2026-08-06] TWO RULINGS UNBLOCKED A 1.7x, AND A DRIVER HEURISTIC WAS EATING THE ROUNDS.
  The attention loop went **0.570 -> 0.967 min_ratio (v024 -> v026)** once two coordinator-only
  blockers were cleared: (1) the split-KV numeric gate, because online-softmax reassociation makes
  bit-equality arithmetically impossible, and (2) the register ceiling, which had been measured
  against the frozen standalone golden's 238 rather than the megakernel's real 255 -- the loop had
  been discarding every numerically-correct split instantiation for ~17 registers it was entitled to.
  Split K/V pages came back bit-exact with max_abs_ulp=2.000000, inside the gate.
  LESSON: when a loop stalls with good work and asks for a ruling, the blocker is often a CONTRACT
  the coordinator wrote carelessly, not a technical ceiling. Both of these were mine.
  SEPARATELY: `chain_longhaul.sh` classified ANY episode under 600s as an "API/env FAILURE" and slept
  a 1800-7200s backoff. That threshold was tuned for Claude Code; under codex, healthy rounds finish
  in 380-600s WITH EXIT CODE 0, so three of ws6's rounds were misread as failures and punished with
  hours of waiting before the stall counter expired. Fixed: a short episode is a failure only if it
  ALSO exits nonzero. When porting a runner, re-check every heuristic that was calibrated against the
  old one's timing.
- [2026-08-06] A HARNESS'S OCCUPANCY MODEL CAN INVERT A KILL DECISION. The MoE loop killed probe a043
  because 64 -> 80 registers cost it "3 CTA/SM instead of 4, a 25% residency loss on the 824-CTA
  grid". That is real IN ITS HARNESS and IRRELEVANT in production: `persistent_kernel` is declared
  `__launch_bounds__(WORKER_NUM_THREADS, 1)` (persistent_kernel.cuh:1571), so MPK runs ONE worker
  block per SM and each worker drains its queue sequentially -- there is no CTA-per-SM co-residency
  to lose, and a task body at 80 registers occupies the same slot as one at 64 up to the 255 budget.
  So a top-lever loop closed itself on a standalone artifact. It cuts BOTH ways and the loop was told
  so: any kept win whose mechanism is "more CTAs in flight per SM" rather than "less work" or "better
  per-task pipelining" is also suspect. Add occupancy MODEL to the harness-vs-production checklist,
  alongside grid shape, -D knobs, fast_math and the register ceiling.
- [2026-08-01] THE WORKSPACE task.yaml IS A FROZEN SNAPSHOT — EDITING THE CANONICAL FILE DOES
  NOTHING. `cc-init.sh` copies `tasks/<name>.yaml` to `workspace<N>/task.yaml` at workspace creation,
  and the loop reads the WORKSPACE copy. Every later edit to `tasks/` is INERT. This silently
  invalidated FIVE of my interventions before I noticed: the attention split-KV permission, the
  29.6us-refutation hint, the split-KV numeric-gate ruling, and the register-ceiling + both-nvcc-lane
  guards on BOTH moe and dense. Caught only because the codex episode said "the canonical task file
  contains the exact coordinator ruling at line 670; only workspace6/task.yaml missed the sync."
  CONSEQUENCE FOR A CLAIM I MADE: the MoE register guard that killed probe a043 worked because I had
  ALSO appended the same text to the SEED — not because of the task.yaml constraint, which never
  reached the loop. Seed appends are the LIVE channel (re-injected every episode); task.yaml is the
  frozen one.
  THE FREEZE IS CORRECT, do not remove it — the loop must not chase a moving spec, and the attention
  loop rightly refused to edit its own task.yaml. The fix is a DELIBERATE coordinator sync:
  `scratchpad/sync_task_yaml.py` copies canonical -> workspace after checking that every workspace
  constraint survives in the canonical (it refuses and asks for review otherwise — it correctly
  flagged ws6, where two constraints had been PREPENDED to rather than replaced).
  RULE: after ANY canonical task.yaml edit, sync the workspace copy and VERIFY the marker is present
  in `workspace<N>/task.yaml`, not just in `tasks/`. Better: put anything that must bind THIS round
  in the seed as well.
- [2026-07-31] AN OPTIMIZATION'S SIGN IS A FUNCTION OF OCCUPANCY — now THREE independent sightings on
  this project, so treat it as the default hypothesis whenever one config lags a family.
  (1) M4-I9 fusion: modelled -63/-130us, measured +42/+25 once the machine was work-bound, because
  the absorbed victim's work is paid in full on a busy worker. (2) MoE w13 v022: SWAP_AB wins at <=3
  waves and LOSES at 5.6 waves, where the kernel is DRAM-request-bound with MMA ~0 so the burst it
  halves is already free and only its extra smem operand traffic remains. (3) Dense qkvg: M1 140.3%,
  M2 139.0%, M4 139.0%, M8 138.4%, then **M16 108.9%** — four configs within 2 points and a 30-point
  cliff at M16 alone (candidate instance, hint filed).
  THE DIAGNOSTIC SIGNATURE: not a gradual roll-off across a config family but a CLIFF at the
  largest/deepest one. THE FIX IS A REGIME GATE, NOT A REVERT — v022's shape is the template: a
  DEFAULTED template param (`SWAP_AB && ALLOW_SWAP_AB`) with the dispatch selecting false when
  work > 4*num_sms, CTA-uniform, leaving every existing instantiation byte-for-byte unchanged. That
  move alone was worth 0.885 -> 0.932. THE INSTRUMENT is a discriminating ablation: force the suspect
  off at UNCHANGED fetch shape, grid, smem AND register count. Do not confound it with a shape change
  — v022's first attempt (a043) moved TILE_N and STAGE_TILES together, took 64 -> 80 registers, lost
  a CTA/SM to residency, and had to be killed before the real lever became visible.
- [2026-07-31] INTEGRATE CLOSED LOOPS; LET OPEN LOOPS RUN. A loop that is still climbing steeply
  should NOT be integrated — the work gets redone. A loop that has closed itself out should be
  integrated promptly, because its value is now pure capture with no rework risk, and an
  un-integrated closed loop is the easiest kind of win to forget. State at this writing: MoE w13
  CLOSED at v022/0.932 (its own verdict: w13_bs16 is at its practically reachable ceiling and the
  1.3333 bar is roofline-infeasible, with STAGE_TILES>1, maxrregcount forcing and fetch replays all
  closed on pre-registered kill criteria) — the shipped kernel is still v012-based, so v013..v022 is
  UNCAPTURED and integration is the top pending item. Attention OPEN and climbing fast
  (0.217 -> 0.412 -> 0.548 in ~6h; 46.7 -> 18.75us, 2.49x) — leave it. Note the MoE loop and the
  shipped tree diverged in KIND, not just degree (the integrator added work-item flattening on top
  of v012 that the standalone harness cannot see), so integrating v022 means re-applying the
  flattening over the new body, not swapping files.
- [2026-07-31] A LOOP CAN CLOSE ITSELF HONESTLY, AND THAT VERDICT IS WORTH TAKING. The MoE loop
  stalled out at 6 no-tag rounds — but reading WHY showed it was not out of effort: it had priced a
  register-bound occupancy lever at a few percent, declared the target roofline-infeasible, and
  explicitly closed three axes so a future round would not re-run them. Contrast the attention loop,
  whose stall at 3/6 was a MIS-AIMED BAR (its spec forbade the one lever that mattered) and which
  went on to 2.49x once unblocked. So: on a stall, read the loop's own reasoning before deciding
  between re-aim and retire — the two look identical from the tag counter.
- [2026-07-31] A HARNESS MUST REPLICATE THE PRODUCTION *COMPILE REGIME*, NOT JUST THE PRODUCTION
  SHAPES — and matching shapes meticulously is what disguises the gap. Audit prompted by the user's
  rule: "if claimed perf is not benefiting end-to-end as expected, check whether the test config
  (grid size, num_worker, compilation parameters, registers) aligns with the demo." The MoE task.yaml
  — our TOP AC-4 lever — matched shapes, ABI, dtypes, routing and expert geometry in exhaustive
  detail, and had ZERO occurrences of register / Xptxas / spill / launch_bounds / fast_math (its one
  "register" hit was the filename `task_register.cc`). Its sibling attention task had 43/4/15/3/8.
  Two concrete holes: (1) no register ceiling, though MPK inlines every task into ONE
  `persistent_kernel` at `__launch_bounds__(256,1)` and the megakernel already sits at 255 registers
  / 0 B spill, so any win costing registers taxes every other stage; (2) no `-use_fast_math` lane,
  though that is the shipped JIT's default (`persistent_kernel.py` ~L284-290) — so both the
  bit-exact gate and the timings ran in a compile regime production never uses.
  THIS IS NOT HYPOTHETICAL: the ROUTER task also lacked a register guard, cleared its 30% bar
  in-harness (1.417, 29-41% faster than vLLM's kernel), and then delivered only +3.2-5.0% e2e,
  handing back ~21% of its own gain to +17 registers. The in-harness number was real and the
  integration was still disappointing, which is exactly the signature to watch for.
  CHECKLIST before trusting any loop's ratio, or when a win fails to translate: grid/CTA shape vs the
  builder's actual `grid_dim`; how many of the 128 workers the stage really gets; every `-D` knob AND
  every default-flag value; `-use_fast_math`; the shared register/spill ceiling; template args as
  INSTANTIATED at the call site (not as named in the .cuh); page size; context depth; active rows.
  A per-kernel win only counts once it survives the megakernel's shared budget.
- [2026-07-30] DO NOT DRAW CONCLUSIONS FROM A TAG BODY READ WHILE THE LOOP IS MID-TAG — and never
  "fix" a measurement instrument on that basis. Two false alarms in one tick, both from reading a
  LIVE workspace: (1) `ferret.state` printed "no tagged improvements" although v018-v021 existed, so
  I concluded its parser had broken on a `TFLOPS:`/`Reference:` tag body and was about to widen the
  scorer; (2) from that same partial body I computed w13_bs16 down 9.9% and reported a regression on
  the one config that matters for AC-4. Both were artifacts: the agent writes a draft tag body and
  AMENDS it with the canonical KERNEL_RESULT/KERNEL_RESULT_REFERENCE keys, so a read landing in
  between sees a body that is both unparseable and numerically stale. Re-reading minutes later gave
  score 0.885 / worst w13_bs16 and a candidate that was +0.6%, not -9.9%. RULES: (a) `git describe
  --tags` + the state dump are a SNAPSHOT of a moving tree — re-read before acting; (b) a
  cross-tag TFLOPS comparison is only valid at a CONSTANT reference (v017's reference differed by
  +3.2%, which alone accounted for the apparent 0.908 -> 0.885 "drop"); (c) widening a scorer is a
  change to the instrument that grades the work — require a reproducible failure first.
- [2026-08-06] THE GATING METRIC IS NOW IN-MPK PER-TASK DURATION, NOT THE STANDALONE SCORE. The
  Part-1 discriminating measurement (/var/tmp/alignment/part1, full production geometry 136w+48s,
  profiler npz, anchor_qc PASS, tokens identical) showed the standalone harness OVERSTATES the
  in-MPK win ~2x even in the production fast-math lane at matched shapes: MoE v024 claimed −17.55%
  isolated, delivered −7.4% (w13) / −12.2% (w2) per-task in-MPK, −2.9% on the critical path, and
  ~0 to NEGATIVE on the real step (bs16 +0.87% slower, 0/6). Two stacked losses: standalone→in-MPK
  ≈0.53 (cross-worker L2/DRAM sharing among 136 heterogeneous workers), in-MPK→step ≈0 at bs16
  (static-schedule straggler reshuffle / work-bound coupling). RULES for every relaunched loop:
  (a) standalone KERNEL_RESULT is an exploration signal only; a tag is integration-ready ONLY when
  the in-MPK per-task duration confirms at production geometry (the instrument: profiled run →
  raw npz → per-type mean, as in part1); (b) run the in-MPK check periodically (every Nth tag),
  not only at convergence; (c) expected e2e = per-task delta × path share × layer-2 transfer —
  quote THAT to the user, never the standalone ratio; (d) fast-math lane is the scoring lane.
- [2026-08-06] PIN THE COMPILER EXECUTABLE, NOT JUST FLAGS. The harness's unqualified `nvcc`
  resolved through /usr/local/cuda to CUDA 13.2.51 + C++17 while the production JIT captures
  /usr/local/cuda-12.8/bin/nvcc + C++20. Same immutable MoE source: 64 regs/16B stack (13.2/C++17)
  vs 75 regs/0B stack (12.8/C++20) — register counts and stack behavior are NOT comparable across
  toolchains, and a loop tuning under the wrong toolchain optimizes phantom pressure. All three
  build.sh now default MPK_NVCC=/usr/local/cuda-12.8/bin/nvcc; keep it that way until the
  production JIT itself moves, then move BOTH in one commit. Corollary: any historical register
  number in tags/memory predating 2026-08-06 may be old-toolchain — re-measure before comparing.
- [2026-08-06] Analysis instruments are part of the alignment surface: m4i5 width.py hard-coded
  NW=128 and m4i8 sched_gap.py defaulted 128/80 topology — both silently wrong at 136/48. The
  generalized wrappers live in /var/tmp/alignment/{width_realized,sched_gap_realized}.py with
  assign_qc/identity_error_ns self-checks (PASS at 136/48); port them into the project tree's
  m4i5/m4i8 scripts before the next schedule analysis.
- [2026-08-06, E1 correction] The "harness overstates ~2x" line above needs precision: the −17.55%
  claim came from bitexact_v024.cu at grid=128 + big dyn smem (ALREADY 1 CTA/SM), not from the ws3
  harness — the ws3 harness never timed its golden and its stock oversubscribed regime (3-4 CTA/SM)
  overstates cand-vs-golden at −58/−60% vs −22.3/−5.5% at 136 CTAs. Residency closes W2 entirely
  (−11.6 @g148 ≈ in-MPK −12.18) but leaves a ~13pt W13 residual (−20..−22 @1/SM vs −7.4 in-MPK
  t_live) — under E6 test (overhead-dilution vs co-tenancy vs clocks). PROVENANCE RULE: before
  comparing two instruments' percentages, verify which binary, grid, smem, path family, and body
  pair EACH one actually measured — this claim chain crossed three instruments (loop TFLOPS ratio
  vs FlashInfer / golden-diff harness / integration bitexact benchmark) that share nothing but
  the kernel family name.
  [RESOLVED 2026-08-06 16:55] The 128-topology analyzer port shipped: project-tree width.py +
  sched_gap.py generalized (commit 93d6d55e on qwen3-5_support), validated field-identical to the
  realized wrappers on part1/on at 136/48 (PASS, identity 0ns). Wrappers in /var/tmp/alignment
  remain as the box-local reference.
