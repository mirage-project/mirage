# Box orchestration — the multi-GPU session playbook

Every multi-rank verdict run (TP2 micrographs upward, all e2e milestones) happens on a
remote box. These rules are distilled from real leaks/wedges/OOMs during the DSv3-V2
campaign. **All box operations belong to the TOP orchestrator only** — never inside a
phase-lead or worker subagent.

> PORTABILITY: §1-2 (machine inventory, Nebius CLI/paths/keys) are SITE-SPECIFIC to the
> original environment — on a different site, replace the inventory and keep §3-8
> verbatim (the sync/build rules, session command discipline, launch discipline, safety
> invariants, and testing tiers are box-agnostic and each encodes a real failure).
> Single-GPU campaigns (dense models at TP1) need none of this file.

## 1. Machine inventory

> Site-specific values (IPs, instance IDs, users, SSH keys, work dirs) live in the
> operator's local memory/notes — never commit them; the structural rules below transfer.

- **Local dev box** (several GPUs): micro-benches, harness runs, single-GPU gates.
  No reporting/etiquette; find a FREE card and use it — BUT some cards are broken
  while looking free in nvidia-smi. Torch-probe per use, never hardcode a broken list:
  `CUDA_VISIBLE_DEVICES=$g python -c "import torch; torch.zeros(8,device='cuda'); torch.cuda.synchronize(); print('OK')"`
- **Nebius 8×B200** (`~/nebius_box.sh`, alias `box`): the TP8 verdict machine.
  Exclusive when started; costs money while RUNNING → verify-STOPPED after use.
- **Nebius single-B200**: exclusive 1-GPU gates. Same start/stop discipline.

## 2. Nebius box facts (verify per session — these drift)

- CLI: `~/nebius_box.sh {start|wait|stop|status|ip|ssh <cmd>}`.
- **IP CHANGES every restart.** `nebius_box.sh ip` can time out on the API and fall
  back to a STALE baked IP → always re-query the live IP per session
  (`$(~/nebius_box.sh ip)`) and pin it. Direct `ssh -i <SSH_KEY_PATH> <BOX_USER>@<BOX_IP>`
  is the reliable path — the box accepts exactly ONE specific key; record WHICH in
  operator-local notes (guessing the wrong key burns session time).
- Box paths (layout convention): repo `<BOX_WORK_DIR>/mirage`, model
  `<BOX_WORK_DIR>/models/<Model>`, weight cache `.../cache`, outputs
  `.../outputs/`, venv `.../venv/mpk` (Python 3.12 — see rsync rule 4).
- **Start is flaky**: retry loop 6-8 attempts, 30 s apart; after "start done" wait the
  FULL boot (~5-10 min; `sleep 600` then check) — an early probe looks like failure.
  Never auto-stop on an early failed probe.
- Persistence: the OS disk survives stop/start — built `core.cpython-*.so` and the
  weight cache persist (verify `core.so` mtime vs your last rsync). A REBOOT can
  revert recently-rsync'd source files to a snapshot → re-rsync after `box start`.

## 3. Sync + build rules

1. **NEVER `rsync --delete` onto `python/mirage/`** — it wipes the box's
   `core.cpython-*.so` (a build artifact, not in git). Recovery:
   `MIRAGE_SKIP_NATIVE_BUILD=1 pip install -e . --no-build-isolation --no-deps`.
2. Sync ONLY source: `src/ include/ python/mirage/ demo/<model>/ tests/...` +the run
   scripts. Exclude `scratch/` (tens of GB of trace junk), `.git/ outputs/ .pk_*/
   *.csv deps/`.
3. **Never rsync your LOCAL `core.*.so`** — local venv may be a different Python ABI
   than the box (3.11 vs 3.12 bit the campaign). Exclude `*.so`.
4. Rebuild on box after any C++/Cython-visible change:
   `pip install -e . --no-build-isolation --no-deps` (~7 min). `.cuh`-only edits need
   NO rebuild (JIT'd at launch) — but every launch pays ~10-15 min of nvcc for the
   megakernel `.so` (no content-hash cache exists yet); budget windows accordingly.

## 4. Session command discipline (the three structural rules)

**Rule A — split setup(A) / poll(B), CONDITIONAL traps.** Never one combined bg
command with an unconditional `box stop` EXIT trap (a killed poller then stops the box
under a live run). Phase A launches the detached run and its trap stops the box ONLY
if the launch failed (`ssh box "pgrep mpirun" || box stop`); Phase B is a SEPARATE
poll command that stops the box only after detecting completion/timeout. A killed B is
harmless — relaunch it.

**Rule B — one box-touching command at a time.** A foreground `nebius_box.sh`/`ssh`
while a background box command is live races shared state and has KILLED the bg
command. Wait for the bg command before issuing anything new.

**Rule C — detach long remote commands; poll a completion file.** Idle SSH gets
dropped mid-command ("Connection closed by remote host"):
```bash
ssh -o ServerAliveInterval=30 box \
  "setsid nohup bash -c 'long_cmd >log 2>&1; echo \$? >/tmp/done.rc' &"
until ssh box "[ -f /tmp/done.rc ]"; do sleep 30; done   # keepalive on each poll
```
Subagent corollary: a bg command + "park waiting for its notification" DEADLOCKS a
subagent — drive box waits with FOREGROUND timeout'd polling. (This is one reason box
ops live at the top orchestrator.)

## 5. Launch discipline (every GPU/host job)

- **Memory-cap every launch**: `systemd-run --user --scope --expand-environment=false
  -p MemoryMax=<N>G -p MemorySwapMax=0 <cmd>` (needs `XDG_RUNTIME_DIR=/run/user/$(id -u)`
  when detached). OOM then kills only your scope, not the box.
- Wrap runs with timeout + trap-cleanup + a D-state-zombie guard (the
  `scratch/run_dsv3_mpk_remote.sh` pattern — that script is git-ignored/machine-local;
  the pattern = systemd-run memory cap + `timeout` + a trap that kills the job tree +
  a post-run `ps -eo stat | grep '^D'` check).
- **TP8 cold weight-convert OOM**: export `MPK_CONVERT_SEMAPHORE=K`,
  K = floor(MemoryMax_GB/350) clamped [2,4] (1500G cap ⇒ K=4). The warm cache skips
  the ~40-min convert on follow-up runs; the cache key is env-var-agnostic.
- **The mpirun `-x` gap**: generic launchers forward only core NVSHMEM/CUDA vars. Any
  `MPK_*` env-gated lever/probe MUST be forwarded with an explicit `-x MPK_<VAR>` per
  var, and its per-rank effective value confirmed in the log — otherwise the lever
  silently runs at default and your A/B measures noise (this trap produced two
  confabulated results).
- mpirun env also needs `nvcc` on PATH (`-x CUDA_HOME`, `PATH=$CUDA_HOME/bin:...`) or
  `mpk.compile()` dies with "nvcc not found"; `NVSHMEM_MAX_TEAMS=128`;
  `LD_PRELOAD` the nvshmem host lib per the demo readme.

## 6. Safety invariants (non-negotiable)

- **Never crash-loop the megakernel.** A wedged persistent kernel leaves unkillable
  D-state zombies that pin GPU memory; only a reboot clears them. One hung/failed
  run ⇒ collect logs, clean up, verify no D-state (`ps -eo stat,pid,cmd | grep '^D'`),
  STOP — do not retry on a faulted node.
- **Verify-STOPPED after EVERY session** (`box status` + a util check). A finished
  campaign that leaves the box RUNNING is a money leak (happened: ~55 min idle).
  Corollary: `status=RUNNING` ≠ "a live run" — RUNNING + all GPUs 0% + no
  demo/mpirun procs = leaked-idle, free to reuse or stop.
- Put the box-stop in a trap/finally of the top-level session script so it fires on
  early exit — but per Rule A, conditional on "no live run I still own".
- Exclusive GPUs: the persistent kernel co-schedules workers across all visible GPUs;
  any foreign process on a target GPU deadlocks the launch. Check
  `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
  before every run.

## 7. Multi-rank testing tiers (cheapest sufficient tier wins)

1. Single-GPU harness (local, probed card): per-op correctness, leaf/mega bodies.
2. **TP2 micrograph** (2 GPUs, local if 2 clean cards probe OK, else box): the FIRST
   gate for every collective port; known vectors, bit-exact on both ranks.
3. TP8 on the 8-GPU box: collective re-validation, all e2e milestones, every
   verdict-grade number.
Do NOT fake TP-N by packing multiple PEs per GPU: NVSHMEM >1 PE/GPU is "limited MPG"
(`team_split_strided` fails); the MPS workaround exists
(`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE<=49`, private pipe dir) but inflates kernel bodies
~3× — stress-testing only, never absolute numbers. Also note: fused paths gated on
exact geometry (e.g. DSv3 FFN mega needs routed_tp==4 ⇒ world==8 at EP2) simply do
not exist at smaller TP — measuring "it" there measures a different code path.

## 8. Canonical session skeleton (top orchestrator)

```
probe/START box (retry ≤8, sleep 600, live IP pinned)
  → rsync source (rules §3) → rebuild if C++/pyx changed → verify core.so mtime
  → [cold cache? export MPK_CONVERT_SEMAPHORE=K]
  → Phase A: launch detached run (systemd-run cap + timeout + completion file);
             trap: stop box ONLY if launch failed
  → Phase B: poll completion file (keepalive); collect logs/tokens/traces to
             durable outputs dir
  → post-run: D-state check → next arm or DONE
  → STOP box → verify STOPPED
  → record results in experiment_history/ (journal + INDEX row)
```
