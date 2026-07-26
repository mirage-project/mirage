# ferret operational bringup — mpk-qwen3.5 (M3 prep)

Status as of this bringup: **ferret is runnable on catalyst-B200 for this project.** Proven
by a live smoke run (not just docs review) — see §4. This doc is the operating reference for
whoever dispatches the first real M3 kernel-optimization issue through `ferret-kernel-agent`.

## 0. TL;DR for the M3 dispatcher

- Our install: `~/mpk-qwen35/ferret` (branch `cc`, HEAD `463d4f3f5de83ef8a66222c5e0fd5298ce088d99`).
  **Not** `~/ferret` (that is the user's own reference install — read-only, never touched).
- Bridge: always invoke ferret via **our clone's own path**
  (`bash ~/mpk-qwen35/ferret/scripts/cc-run.sh ...`, or `cd` into it first). No env-var override
  is needed or exists — see §2.
- `ferret-kernel-agent.md`'s contract text hardcodes `~/ferret/...` throughout. Until that file
  is edited (out of this bringup's write scope), **every dispatch prompt must explicitly tell
  the dispatcher to substitute `~/mpk-qwen35/ferret` for every `~/ferret` path it reads there.**
- claude CLI on B200: installed, authenticated, confirmed working headlessly (§3). Not a blocker.
- codex CLI on B200: not on `PATH` as `codex` (binary + auth exist under a different name) — a
  soft gap, not a blocker (§3.4). Fix is a global PATH change outside this bringup's scope.
- Smoke run: full pass, one bounded episode, 350s. `v001` tagged, reviewer PASS, goal-reached
  (§4).

---

## 1. Repo state, deps, workspace layout

### 1.1 Bring-current

```
git -C ~/mpk-qwen35/ferret fetch origin && git -C ~/mpk-qwen35/ferret checkout cc && git -C ~/mpk-qwen35/ferret pull
```

Result: already on `cc`, already up to date. **HEAD `463d4f3f5de83ef8a66222c5e0fd5298ce088d99`**
— `feat(remote): scripts/remote_run.sh — CC-mode mainthread self-submits compile+benchmark to a
remote GPU` (2026-06-09). This matches `design/ferret/` (the local read-only orientation clone)
byte-for-byte on every top-level file checked, confirming the design clone was accurate orientation
material.

### 1.2 Dependencies

ferret's own runtime dependency footprint is intentionally tiny (`pyproject.toml`:
`pyyaml>=6.0`, nothing else; "ferret runs in place — no pip install of the package itself").

- **pyyaml: already satisfied.** System `python3` (3.12.3, `/usr/bin/python3`) has PyYAML 6.0.1
  installed via the apt package `python3-yaml` (`/usr/lib/python3/dist-packages`) — pre-existing
  on the box, not something this bringup installed. `pip show` confirms the package; the smoke
  run confirms it functionally — `task_spec.py`'s CLI dry-scored our task cleanly, and every
  `ferret.state`/`ferret.cc_goal`/`ferret.task_spec` module import used during the run worked.
- **No venv used, matching ferret's own convention.** `cc-run.sh` never activates a venv — it
  just calls plain `python3`. The user's own `~/ferret` has no venv either (checked, read-only).
  Do **not** introduce a dedicated ferret venv; a mainthread's Bash calls plain `python3`, so
  anything installed only inside an unactivated venv would silently not exist to it.
- **If pyyaml were ever missing** on a fresh box: `UV_CACHE_DIR=~/mpk-qwen35/.uv-cache python3 -m
  pip install --user pyyaml` (user-site install, so plain `python3` picks it up without a venv
  activation step). Not exercised this run — not needed.
- **Submodules** (`resources/<lib>/`, 11 total per `.gitmodules`): all were uninitialized at
  clone time. Initialized **only** `resources/kernelwiki` (shallow, `--depth 1`), per the
  README's explicit install step (`git submodule update --init resources/kernelwiki && bash
  scripts/update_kernelwiki.sh`) — 27 MB, negligible. **Deliberately left the other 10
  uninitialized** (`cutlass-4.4.2`, `flashinfer-0.6.7.post3`, `flash-attention-fa4`,
  `triton-3.6.0`, `tensorrt-llm-1.2.0`, `nccl-2.29.7-1`, `deepgemm-2.1.1.post3`, `flashmla-main`,
  `thunderkittens-main`, `documentsass`) — see §1.3 for why. These are lazy reference material
  (`resources/<lib>/` paths an individual task's `references[]` points at); a real M3 dispatch
  should init only the specific submodule(s) its chosen references actually need, checking `df`
  before/after each.
  - Minor friction found: the `--depth 1` shallow clone breaks `update_kernelwiki.sh`'s channel
    (A) upstream fast-forward sync (`fatal: refusing to merge unrelated histories` — shallow
    history has no common ancestor with `origin/master` for a clean FF). The script's own
    WARN-and-continue design absorbed this without failing the run. If disk allows, prefer a
    **full** `git submodule update --init resources/kernelwiki` (no `--depth`) next time — the
    corpus is small enough (27 MB shallow) that full history is unlikely to be a real cost.
  - `update_kernelwiki.sh` also runs an optional channel (B) — a `gh`-search-based corpus
    *enrichment* pipeline (new merged-PR pages across vllm/flashinfer/pytorch/deepgemm/sglang
    etc.). `gh` is installed and authenticated on this box, so channel (B) ran for real — it is
    network-bound and open-ended (no timeout in the script), explicitly documented as
    "safe to run from cron." Left running in the background; it does not gate anything — the
    **offline read path** (`resources/kernelwiki/scripts/query.py`, what `planner`/`iterator`
    actually query) was verified independently functional regardless (a live query for
    "vector add" returned 3 relevant hits from the pre-enrichment corpus).

### 1.3 Disk headroom (shared filesystem — real constraint, not hypothetical)

`/dev/md1` (mounted at both `~` and `/raid`) is a **28 TB, ~100%-utilized, shared** filesystem.
Observed **34G → 28G avail over roughly 20 minutes purely from other users' concurrent activity**
(confirmed via `nvidia-smi --query-compute-apps`: the GPUs busy during this bringup were other
accounts' jobs — a `ziweizho` process and two `VLLM::EngineCore` instances — none of it ours).
Still comfortably above the required ≥10G floor throughout (spot-checked stable at 28G over a
20s re-check, i.e. not a fast continuous drain), but the trend is real and external. This is why
§1.2 intentionally skipped the 10 heavy vendor submodules (cutlass/flash-attention/triton/
tensorrt-llm alone can plausibly be multi-GB each) rather than initializing them speculatively.
**Recommendation for M3:** check `df -h ~` immediately before any submodule init or large
workspace op, not just once at session start.

### 1.4 Workspace provisioning

Provisioned **all 8** workspaces via ferret's own mechanism (`scripts/cc-init.sh <N>
tasks/smoke-vecadd-b200.yaml`, run once per N=1..8) — each got its own independent `.git` (no
shared history with the parent repo or siblings), a copy of `task.yaml`, and a `progress.md`
skeleton. This mirrors the layout observed in the user's real `~/ferret/workspace1` (own `.git`
with real tags `v101`-`v105`, `kernel.cu`/`kernel.cuh`/`kernel`/`progress.md`/`task.yaml`, plus an
optional task-specific `gate/` dir — see §5.3). Only `workspace1` was actually **run** (the smoke
episode, §4); workspaces 2-8 are initialized-but-idle, ready for the next 7 parallel dispatches.

`cc-init.sh` refuses to create a workspace without a `task.yaml` argument and refuses to clobber
a non-empty existing one — so "provisioning" a workspace is inherently paired with assigning it a
task, matching the real system's actual first-come-first-served convention (`ferret-kernel-agent.md`
§Step 2), not a separate empty-mkdir step.

---

## 2. The `~/ferret` vs our clone bridging decision

**Question:** the `ferret-kernel-agent` contract (`workspace/.claude/agents/ferret-kernel-agent.md`)
hardcodes `~/ferret/...` everywhere. Our operational install is `~/mpk-qwen35/ferret`. Does ferret
support an override (`FERRET_HOME`-style env var, a `cc-run.sh` flag), or does it only work at
`~/ferret`?

**Answer, read directly from source (`scripts/cc-run.sh`, `scripts/cc-init.sh`):**

```bash
FERRET_DIR="$(cd "$(dirname "$0")/.." && pwd)"     # cc-run.sh line 26, cc-init.sh line 20
...
export FERRET_ROOT="$FERRET_DIR"                    # cc-run.sh line 89 — OUTPUT, not input
```

`FERRET_DIR` is derived **entirely from the invoked script's own path** (`dirname "$0"`, one
level up). There is **no env var ferret reads as an input override** — `FERRET_ROOT` is only ever
*written* by `cc-run.sh` (exported for downstream subagents to read), never *read* as a way to
redirect where `cc-run.sh` itself operates. Grepped the whole repo for `FERRET_HOME`/`FERRET_ROOT`
/`FERRET_DIR` usage (`.claude/agents/*.md`, `docs/dev-memory-seed/*.md`, all scripts) — every
consumer downstream (`iterator.md`, `planner.md`, `kernel-extractor.md`, `reviewer.md`, the
`kernelwiki` skill) reads `${FERRET_ROOT:-$HOME/ferret}`, i.e. **falls back to `$HOME/ferret` only
when `FERRET_ROOT` isn't set** — and `cc-run.sh` always sets it correctly from its own location
before `exec`-ing `claude`. `docs/dev-memory-seed/machine.md` confirms in prose: *"`FERRET_ROOT`
is the ferret checkout (**defaults to** `$HOME/ferret`)"* — a default, not a requirement.

**Conclusion:** the bridge is simply **always invoke the launcher by our clone's own path.** No
env var, no flag, no ferret-source edit needed:

```bash
bash ~/mpk-qwen35/ferret/scripts/cc-run.sh <N> ~/mpk-qwen35/ferret/tasks/<name>.yaml --prompt "..."
# or: cd ~/mpk-qwen35/ferret && bash scripts/cc-run.sh <N> tasks/<name>.yaml --prompt "..."
```

That reading of the source and the smoke run agree — its startup banner:

```
ferret root         : /home/muhengl/mpk-qwen35/ferret
FERRET_WORKSPACE    : workspace1
FERRET_ROOT         : /home/muhengl/mpk-qwen35/ferret
PYTHONPATH          : /home/muhengl/mpk-qwen35:
```

**What this means for the dispatcher contract:** `ferret-kernel-agent.md` was never edited (out
of scope for this bringup — it's a shared control file, not something a bringup task should
modify unreviewed). Until it's parameterized, **every M3 dispatch prompt must explicitly instruct
the agent to read `~/ferret/...` in that file's text as `~/mpk-qwen35/ferret/...`** — i.e. carry
the substitution by hand at dispatch time, the same way this bringup did it. Recommended follow-up
(not performed here): add one line near the top of `ferret-kernel-agent.md` naming
`~/mpk-qwen35/ferret` as `FERRET_INSTALL_ROOT` and replace the hardcoded paths with that variable,
so future dispatches don't need a per-invocation reminder. This is a small, mechanical,
low-risk edit — a native-subagent-lane task in its own right, not a workflow.

**Never touch `~/ferret` itself.** It is the user's own reference install (their real, live
kernel-optimization history — workspace1-4 have genuine tagged work from May-July). This bringup
only ever `ls`'d and `cat`'d files there (read-only) to learn the real layout/conventions; every
write happened under `~/mpk-qwen35/ferret`.

---

## 3. cc-run.sh environment, GPU integration, and the claude/codex CLI status

### 3.1 What `cc-run.sh` sets up (all automatic — a dispatcher never sets these by hand)

| Var | Source | Observed value (smoke run) |
|---|---|---|
| `FERRET_WORKSPACE` | `workspace$N` literal | `workspace1` |
| `FERRET_ROOT` | `dirname($0)/..`, resolved | `/home/muhengl/mpk-qwen35/ferret` |
| `PYTHONPATH` | `dirname($FERRET_ROOT)` prepended | `/home/muhengl/mpk-qwen35:` |
| `TMPDIR` | `/tmp/$USER` (ncu workaround) | `/tmp/muhengl` |
| `CUDA_VISIBLE_DEVICES` | `pick_gpu.sh` output, `eval`'d | `0` (see §3.2) |

A dispatcher's job is only: pick `N`, write `task.yaml`, pass `--prompt`/`--goal`. Everything
above is `cc-run.sh`'s responsibility.

### 3.2 GPU selection — two independent mechanisms, worth knowing about

This project's own convention (`probes/gpu_guard_v2.sh`, referenced from `MAIN.md`) is a
**3-sample, 3s-apart, ≤500 MiB/≤5% util stability check**, writing a claimed GPU id to
`.gpu-locks/<name>.lock`. ferret has its **own**, separate `pick_gpu.sh`: a single-sample,
memory-only heuristic (picks the lowest-`memory.used` GPU under 50%, self-excluding GPUs already
pinned by another *live ferret* session — but not other jobs generally).

**`cc-run.sh` unconditionally calls its own `pick_gpu.sh` right before exec-ing claude — it
cannot be pre-empted or skipped**, and there's no flag to disable it. Concretely, in this smoke
run: our project's 3-sample guard validated **GPU 4** as stable-idle and wrote the lock file;
seconds later `cc-run.sh`'s own `pick_gpu.sh` independently picked **GPU 0** (also genuinely
idle — just a different idle candidate, tie-broken by `nvidia-smi` index order). Both were
correct; they just don't have to agree.

**Practical guidance for M3 dispatchers:** run the project's 3-sample guard as **pre-flight
due-diligence** (it's real validation, worth keeping for the audit trail), but treat it as
advisory, not a hard pin — **the GPU actually in use is whatever `cc-run.sh`'s own startup
banner reports** (`CUDA_VISIBLE_DEVICES: N` line, printed before it execs claude). Read that
back and reconcile the lock file to match reality, as this bringup did
(`.gpu-locks/ferret-smoke.lock` → corrected from `4` to `0` post-launch).

`pick_gpu.sh` also does not itself enforce the "MPK megakernel needs an EXCLUSIVE GPU, can
deadlock if shared" rule (`resources.md`) — that risk is specific to the real persistent
megakernel, not a plain standalone `nvcc`-compiled `kernel.cu` benchmark. For a **real** M3
dispatch whose kernel eventually runs through `mpk-validator` (§5.3), GPU exclusivity is enforced
there instead (it torch-probes and picks an idle GPU itself, separately, for the in-MPK
validation step).

### 3.3 claude CLI on B200 — installed, authenticated, confirmed working (not a blocker)

This was flagged as the critical thing to check, since ferret's loop runs a real headless Claude
Code mainthread **on the B200 host itself**. Findings:

- **Installed:** `/home/muhengl/.local/bin/claude` → `/home/muhengl/.local/share/claude/versions/2.1.218`.
- **Only resolves via a login shell.** Plain non-interactive `ssh catalyst-B200 'which claude'`
  fails (`command not found`) — `~/.local/bin` isn't on the non-interactive PATH. `ssh
  catalyst-B200 'bash -lc "which claude"'` resolves it. Same class of issue as the already-known
  nvcc-PATH lesson (`MAIN.md`) — anyone scripting a remote dispatch must use a login shell or an
  explicit `PATH=` prefix, not a bare non-interactive `ssh cmd`.
- **Authenticated and functional — verified two ways, not assumed:**
  1. A direct, minimal `claude -p "reply with the single word OK"` (no extra flags) returned
     `OK`, exit 0.
  2. The full smoke run (§4) dispatched real subagents (`planner`, `reviewer`) headlessly and
     produced correct, reviewed, tagged work — the more direct proof, since it exercises exactly
     the path a real M3 dispatch will use.
- **Why headless autonomy works without `--dangerously-skip-permissions`:** `cc-run.sh` never
  passes that flag (confirmed by reading it — it only passes `--append-system-prompt`). Autonomy
  instead comes from this account's **global** `~/.claude/settings.json`:
  `"permissions": {"defaultMode": "auto"}` plus `"skipDangerousModePermissionPrompt":
  true`/`"skipAutoPermissionPrompt": true`. This is an **operator-level** precondition, not
  anything ferret's scripts configure — whoever's account runs `cc-run.sh` needs an equivalent
  global config, or the headless mainthread will stall on permission prompts it has no way to
  answer. Worth stating plainly for any future host bringup.
  - Aside, purely procedural: an early test of mine that added
    `--dangerously-skip-permissions` myself (not part of ferret's actual invocation — my own
    extra flag, to probe auth) was blocked by **my own calling harness's** safety classifier
    (nothing on the B200 side). Root-caused by testing the bare form next: `claude -p` alone
    (ferret's real invocation shape) is not blocked. Net effect: zero — ferret never needed that
    flag, the actual dispatch proceeded cleanly. Noted only so a future dispatcher doesn't
    reach for `--dangerously-skip-permissions` "to be safe" — it is unnecessary and will trip
    that classifier for no benefit.

### 3.4 codex CLI on B200 — present but not on `PATH` as `codex` (soft gap, not a blocker)

- `codex` (bare command) is **not found**, even via login shell.
- The actual binary exists at `~/bin/codex-x86_64-unknown-linux-gnu` (v0.124.0, on `PATH` via
  `~/bin`, but under a machine-specific name, not `codex`), **with a populated, working
  `~/.codex/auth.json`** — i.e. codex itself is installed and authenticated, just not exposed
  under the name ferret's scripts expect.
- **Does not block anything.** Read `reviewer.md`'s actual current logic (not the possibly-stale
  `docs/dev-memory-seed/machine.md`, which describes an older MCP-based path): the reviewer's
  primary ABI check is a **manual `Read`-based comparison**; the optional codex cross-check
  explicitly degrades to *"you just record `API: PASS (manual)`... Never let a Codex call block
  the review"* if unavailable. Our smoke run's reviewer completed with a full PASS without codex
  (moot for this particular task anyway — see §4, it had no Mirage ABI target to check against).
- **Recommendation (not performed — outside this bringup's write scope, a global PATH change):**
  a symlink such as `~/bin/codex -> ~/bin/codex-x86_64-unknown-linux-gnu` (or under
  `~/.local/bin/`) would let a **real** M3 dispatch's reviewer exercise the optional codex
  cross-check. Flagging for the coordinator/user rather than doing it, since it changes the
  account's global environment, not anything scoped to `~/mpk-qwen35/`.

---

## 4. Smoke run — evidence

**Task:** deliberately trivial — a plain FP32 elementwise vector-add (`tasks/smoke-vecadd-b200.yaml`,
`~/mpk-qwen35/ferret/tasks/smoke-vecadd-b200.yaml`), explicitly labeled in its own
`problem.description` as a bringup/loop-mechanics validation task, not a real MPK requirement.
Single config `N16M` (16.7M elements), `target_ratio: 0.85`, `stage_gate.ratio: 0.50`,
`budget: {max_iterations: 6, max_wall_minutes: 30}`. Chosen over reusing an existing real
task (MLA/GEMM decode kernels) specifically to avoid pulling in the heavy vendor submodules
(§1.2/§1.3) and to keep the smoke run's own risk surface minimal — the goal was proving the
harness, not kernel-design skill.

**Launch:** one bounded **episode** (per `CLAUDE.md` §6.6 — "do a small chunk, at most ~4
iterations, then stop and print `EPISODE_STATUS`"), `nohup`+`setsid`-detached on the remote host
so it survives the ssh round-trip, monitored via repeated **foreground, bounded (≤550s)** polling
calls against the workspace log + `git tag` + the state CLI — never backgrounded from the caller's
side, never a blind wait. Hard cap: 40 minutes. **Actual wall time: 350 seconds** (under 6
minutes) — the loop terminated on its own via the episode contract, well inside budget.

**Result.** The mainthread's own `EPISODE_STATUS` line is below, but every number in it was
re-derived independently afterward from the raw git/state-CLI sources, not taken on trust:

```
$ git -C workspace1 tag
v001

$ git -C workspace1 log -1 --format=%B v001
v001: naive coalesced grid-stride FP32 vector-add + CLAUDE.md §6 harness [memory-access]

KERNEL_RESULT {"N16M": 4465.19}
KERNEL_RESULT_REFERENCE {"N16M": 4516.48}
Latency_ms: 0.0451 (median of 100, cudaEvent, L2 flushed)
Max_error: 0
Status: improvement

$ PYTHONPATH=/home/muhengl/mpk-qwen35 python3 -m ferret.state workspace1 workspace1/task.yaml
RunState vs smoke-vecadd-b200:
  stage           : OPTIMIZE
  score           : 0.989 (via min_ratio)
  per-config ratios (kernel / reference):
    N16M: 4465.2 / 4516.5 =  98.9% (target 85%) ✓
```

`EPISODE_STATUS stage=OPTIMIZE score=0.989 best_tag=v001 advance=true
note=smoke-test-passed-loop-machinery-verified-first-tag-clears-target`

**What this proves, concretely:** cold-start detection → `planner` subagent dispatch → correct
from-scratch kernel authored to spec (proper `cudaEvent` timing, ≥20 warmup/median-of-100, 130 MB
L2 flush, host-side correctness check) → `nvcc -gencode arch=compute_100a,code=sm_100a` build →
execution → `KERNEL_RESULT`/`KERNEL_RESULT_REFERENCE` parsing → `git commit`+`tag v001` → `reviewer`
subagent dispatch (verified output keys, all 4 constraints via `grep` evidence, correctness,
correctly marked the Mirage-API check **not applicable** rather than faking a pass) → state CLI
agreement → `EPISODE_STATUS` in the exact requested format. `progress.md` (8 KB, `workspace1/`) is
a genuinely well-reasoned artifact, not filler — it correctly identified up front that this
task has no Mirage ABI target and that `kernel-extractor`/`kernel.cuh` delivery is therefore
N/A-by-design for this particular task, deferring that call to the dispatcher rather than
guessing. That deferral was honored: the delivery/`kernel.cuh`-extraction path (§5.3's
`mpk-validator` included) was deliberately **not** exercised here, since forcing it on a task with
no real Mirage sibling `.cuh` to mirror would test an unrepresentative edge case, not the real
pipeline — it remains to be exercised on the first genuine M3 dispatch, which will have a real
`references[]` target.

No stray processes or GPU memory left behind post-run (`nvidia-smi --query-compute-apps` clean of
our PIDs immediately after).

---

## 5. What an M3 dispatch will look like — worked example: `gdn_recurrent_sm100`

This maps the **real** M2-built kernel `gdn_recurrent_sm100` (SM100 TaskType id 237) to a
hypothetical M3 ferret dispatch, grounded in the pinned architecture
(`docs/qwen35/v1-architecture.md` §2.2/§3.2) and the M2-I5 implementation's own measured numbers
(`.memory/inbox/promoted/20260726-M2I5-gdn-recurrent-numerics.md`). **This is illustrative, not a
committed target** — a real M3 issue would re-derive/confirm these numbers against the actual
M2-I5 build before dispatch, per `ferret-kernel-agent.md`'s own Step-0 quality bar ("name the
precise MPK op... the baseline MUST be the kernel being replaced").

### 5.1 The op

Fused delta-rule recurrence + gated RMSNorm, decode path. One task per (v-head, request slot):
grid `(32, mbr, 1)` — 32 v-heads × `mbr` concurrent request slots. Per M2-I5's own numerics
findings, this is a numerically delicate kernel (bf16-native q/k L2-norm, bf16-rounding-before-
`*norm_w` gated RMSNorm, FMA/association-sensitive fp32 state) — exactly the kind of task where
the frozen-gate pattern in §5.3 earns its cost.

### 5.2 Hypothetical `task.yaml` (illustrative shapes/targets — NOT pinned)

```yaml
name: gdn-recurrent-decode-sm100
gpu: B200
arch: sm_100a
precision: BF16          # io bf16; internal S state fp32 (mandatory, mamba_ssm_dtype)

problem:
  description: |
    Qwen3.5 GDN (gated DeltaNet) linear-attention decode: fused delta-rule
    recurrence + gated RMSNorm, one task per (v-head, request slot).
    S state [32 heads, 128, 128] fp32, persists across decode steps
    (kernel-side step==0 lifecycle, indexed by request slot per the MPK
    porting rule — never by blockIdx). Must reproduce HF's exact bf16
    CAST POSITIONS (q/k L2-norm is bf16-native, NOT fp32; gated RMSNorm
    rounds to bf16 BEFORE *norm_w; o rounds to bf16 before the epilogue) —
    read pytorch_reference.py for cast positions, not this description.
  shapes:
    NUM_V_HEADS: 32
    HEAD_DIM: 128
    HIDDEN_Z: 4096       # z gate width == out_proj input width
    BATCH: 1             # then 8, 16 — see configs

  io:
    inputs:
      - { name: qkv_c, shape: [BATCH, 8192], dtype: bf16 }
      - { name: ba, shape: [64, 2048], dtype: bf16 }
      - { name: alog_dtbias, shape: [2, 32], dtype: fp32 }
      - { name: S, shape: [BATCH, 32, 128, 128], dtype: fp32, prezeroed: false }
      - { name: z, shape: [BATCH, 4096], dtype: bf16 }
      - { name: norm_w, shape: [128], dtype: fp32 }
    outputs:
      - { name: g_out, shape: [BATCH, 4096], dtype: bf16 }

baseline:
  source: "MPK gdn_recurrent_sm100 (M2-I5 correctness-first build, id 237) — the kernel THIS task replaces, benched via its own test-mode driver, NOT an external library"

references:
  - $MIRAGE_ROOT/include/mirage/persistent_kernel/tasks/blackwell/gdn_recurrent_sm100.cuh
  - tests/runtime_python/blackwell/sm100_gdn_recurrent/  # M2-I5's own oracle + testmode driver

configs:
  - name: bs1
    args: { BATCH: 1 }
    target_ratio: 1.30    # M2-I5 measured 13.6us/layer @ bs1; ~27% of 8TB/s HBM roofline
    weight: 1.0
  - name: bs8
    args: { BATCH: 8 }
    target_ratio: 1.15    # state bandwidth already ~2.15TB/s (27% peak) at bs8-16 — less headroom
    weight: 1.0
  - name: bs16
    args: { BATCH: 16 }
    target_ratio: 1.15
    weight: 1.0

scoring: min_ratio
stage_gate: { ratio: 0.95, strict: false }   # M2-I5 baseline is ALREADY correct+working — REPRODUCE should be near-instant

constraints:
  - "Single CUDA stream only. No CUDA graphs."
  - "blockIdx is the WORKER id, not the data row — index all per-request state (S) by request slot, never by blockIdx (MPK porting rule, bit commit 8b19538)."
  - "Bit-exact cast positions per pytorch_reference.py: bf16-native q/k L2-norm; gated RMSNorm rounds to bf16 BEFORE *norm_w; o rounds to bf16 before epilogue. Do not 'fix' these to fp32 — that changes the answer."
  - "S state stays fp32 (mamba_ssm_dtype) — never downcast for a speed win."
  - "Real megakernel compiles with -use_fast_math; validate any candidate under both fast-math and plain builds (M2-I5 measured a ~21 flips/M delta)."
  - "Must pass mpk-validator (cos>0.99, sentinel_rows==0, no crash) at the REAL in-MPK shared-worker WALL-SPAN, not just standalone — a standalone win that doesn't transfer in-MPK is not a win (see gate/ pattern, §5.3)."

budget: { max_iterations: 25, max_wall_minutes: 120 }
output: { result_format: kernel_result_json, result_keys: [bs1, bs8, bs16] }
```

### 5.3 Beyond the basic loop: the `gate/` pattern for high-stakes tasks

The basic reviewer (manual Read + optional codex ABI check, §3.4) is necessarily shallow — it
judges the standalone benchmark, not the real MPK integration. Two mechanisms observed in the
real system close that gap, and a GDN-class task (numerically subtle, already bitten by
cast-position bugs per M2-I5) should use both:

1. **`mpk-validator` subagent** (`.claude/agents/mpk-validator.md`, invoked at FINALIZE alongside
   `kernel-extractor`) — runs the candidate through the **real MPK compile pipeline**
   (`scripts/mpk_validate.sh`) on an exclusive GPU: correctness gate (no crash, `cos>0.99`,
   `sentinel_rows==0`, driver reports PASS) **plus a mandatory in-MPK WALL-SPAN latency**
   (`max(end_ts)-min(begin_ts)` from the test-mode profiler — explicitly NOT the median, which is
   bimodal-idle-CTA-skewed at decode and can misrank a 1.32x-faster kernel as "16x slower"). This
   is a built-in, reusable ferret mechanism — any M3 dispatch can invoke it.
2. **A frozen, hash-locked `gate/` directory** — observed firsthand in the user's own real
   `~/ferret/workspace1/gate/` (read-only reference; a `fused_moe_routed` task), **not** something
   ferret auto-generates. Pattern: a `gate.md` contract + `gate.sha256` (re-verified before every
   round — a hash mismatch aborts the run) + a canonical `host_reference()` copied **verbatim**
   from an already-trusted source (not re-derived) + stage-by-stage intermediate-tensor checks
   (not just a final cosine) + a self-check that the oracle agrees with a trusted GPU reference
   before it judges anything. This is a hand-authored, project-specific defense against "ferret
   marks its own homework" — worth building for `gdn_recurrent_sm100` given its known
   numerically-subtle cast positions, but it is prep work the M3 dispatcher (or a human) does
   *before* launch, not something the task.yaml alone triggers.

---

## 6. Missing prerequisites / open items for the coordinator

1. **`ferret-kernel-agent.md` hardcodes `~/ferret`.** Works today only because every dispatch
   prompt manually substitutes `~/mpk-qwen35/ferret`. Recommend parameterizing it
   (`FERRET_INSTALL_ROOT`) — small, mechanical, native-subagent-lane edit. Not done here
   (shared control file, out of this bringup's write scope).
2. **`codex` not on `PATH` as `codex`** (binary + auth present under a different name, §3.4).
   Soft gap — doesn't block anything today (reviewer degrades gracefully), but a real M3 dispatch
   loses the optional cross-check until a `~/bin/codex` (or `~/.local/bin/codex`) symlink is
   added. Global PATH change — flagged for the user/coordinator, not performed here.
3. **9 of 11 vendor submodules remain uninitialized** (§1.2/§1.3), deliberately, to protect a
   shrinking shared disk (28G avail, draining from other users, not us). Init the specific one(s)
   a real task's `references[]` needs, checking `df` immediately before/after.
4. **`update_kernelwiki.sh` channel (B)** (corpus enrichment) was still running in the background
   at the time of this report — open-ended, network-bound, but self-describes as cron-safe and
   does not block the offline read path planner/iterator actually use (verified independently).
5. **`kernel-extractor`/`mpk-validator` delivery path not yet exercised end-to-end** — the smoke
   task correctly had no Mirage ABI target to extract against. First real M3 dispatch will be the
   first live test of that path; budget extra episodes for it.
6. **claude CLI: no gap.** Included here only to close the loop on the task's explicit ask —
   installed, authenticated, confirmed working both by direct probe and by the full smoke run.
