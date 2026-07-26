# GPU etiquette evidence — M1-I5

Verbatim excerpts from the actual B200 run logs (`~/mpk-qwen35/logs/*.log`, all produced by
`bash -x` scripts, so every `export`/command is echoed with a `+` prefix by bash's own
xtrace before it runs). No text below is edited beyond trimming surrounding lines for
length; provenance (file, line range, timestamp) is given for each block. Etiquette rule in
force throughout: only use a GPU at `~0% util` and `<500 MiB` used, pinned via
`CUDA_VISIBLE_DEVICES`, rechecked immediately before every run.

## Step 7 — HF reference generation (`generate_reference.py`), `CUDA_VISIBLE_DEVICES=1`

Source: `~/mpk-qwen35/logs/generate_reference.log`, lines 1–2 and 44–56 (final successful
run; the file is truncated per-launch by `>`, so it holds only this run — the one that
produced the committed `reference_outputs.json`, exit 0 at `03:17:24`).

```
1:+ export CUDA_VISIBLE_DEVICES=1
2:+ CUDA_VISIBLE_DEVICES=1
...
+ echo '=== generate_reference start: Sat Jul 25 03:14:03 EDT 2026 ==='
=== generate_reference start: Sat Jul 25 03:14:03 EDT 2026 ===
+ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
index, memory.used [MiB], utilization.gpu [%]
0, 166318 MiB, 0 %
1, 4 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 0 %
4, 45253 MiB, 0 %
5, 56097 MiB, 98 %
6, 44651 MiB, 98 %
7, 56095 MiB, 98 %
+ python3 /home/muhengl/mpk-qwen35/generate_reference.py --model-id Qwen/Qwen3.5-35B-A3B-FP8 ...
```

GPU 1 (the pinned device): **4 MiB, 0%** — well inside the `<500 MiB / ~0%` etiquette bound.
GPUs 5/6/7 were at 98% util at this instant (other users' active jobs) and were correctly
not selected.

## Step 8 — vLLM smoke (`vllm_smoke.py`), `CUDA_VISIBLE_DEVICES=3`

Source: `~/mpk-qwen35/logs/vllm_smoke.log`, lines 1–2 and 42–52 (final successful run; same
per-launch truncation — this is the run that produced `matches_hf_reference: true`, exit 0
at `03:42:46`).

```
1:+ export CUDA_VISIBLE_DEVICES=3
2:+ CUDA_VISIBLE_DEVICES=3
...
+ echo '=== vllm_smoke start: Sat Jul 25 03:26:17 EDT 2026 ==='
=== vllm_smoke start: Sat Jul 25 03:26:17 EDT 2026 ===
+ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
index, memory.used [MiB], utilization.gpu [%]
0, 166318 MiB, 0 %
1, 1266 MiB, 0 %
2, 31830 MiB, 0 %
3, 162 MiB, 0 %
4, 45253 MiB, 0 %
5, 56097 MiB, 98 %
6, 44651 MiB, 0 %
7, 56095 MiB, 0 %
```

GPU 3 (the pinned device): **162 MiB, 0%** — under the 500 MiB bound (not literally 0 MiB;
recorded as-is, not rounded down). GPUs 1 and 2 were already in use by other processes
(1266 MiB and 31830 MiB) and were correctly not selected; GPU 5 was at 98% util.

Note: this script was launched once before this run with `CUDA_VISIBLE_DEVICES=3` and
failed at model-registry-inspection time on an unrelated `vllm.vllm_flash_attn` ABI error
(before any GPU memory was touched — the failure was a Python import error, not a CUDA
allocation) — see `generation_run.log`/agent report for that root cause. No GPU
memory/compute was used by that failed attempt.

## Step 4 (optional) — qwen3-8B mirage demo smoke, `CUDA_VISIBLE_DEVICES=2`

Source: `~/mpk-qwen35/logs/qwen3_8b_demo.log`, lines 1–5 (complete run, exit 0 at `03:45:20`).

```
1:+ export CUDA_VISIBLE_DEVICES=2
2:+ CUDA_VISIBLE_DEVICES=2
3:+ export PATH=/home/muhengl/.local/bin:/usr/local/cuda-12.8/bin:...
4:+ PATH=/home/muhengl/.local/bin:/usr/local/cuda-12.8/bin:...
5:+ cd /home/muhengl/mpk-qwen35/mirage/demo/qwen3
```

**Honest gap: this log does not contain an `nvidia-smi` call** — `grep -c nvidia-smi
qwen3_8b_demo.log` returns 0. Unlike `run_reference.sh` and `run_vllm_smoke.sh`, the demo
runner script (`run_qwen3_8b_demo.sh`, written ad hoc for this optional/best-effort check)
did not embed a pre-run `nvidia-smi` call, so there is no B200-log-file evidence of GPU 2's
state at the moment this specific script started. What actually happened per the agent's
own tool-call transcript (not a B200 log file, so not verbatim-quotable here the same way):
a `nvidia-smi --query-gpu=...` was run as a separate command immediately before launch and
showed GPU 2 at `4 MiB, 0%`, and a second check right after showed GPU 2 at `4 MiB, 0%`
again — but since the coordinator asked specifically for B200-log evidence, this step is
flagged as a genuine documentation gap rather than papered over. The `CUDA_VISIBLE_DEVICES=2`
pinning export itself IS in the log (above), so the run was still pinned, just not
log-evidenced for pre-run idleness the way steps 7 and 8 are.

## Step 7 regeneration (reproducibility check), `CUDA_VISIBLE_DEVICES=2`

Source: `~/mpk-qwen35/logs/generate_reference_v2.log`, lines 1–2 and 46–59 (the schema-sync
reproducibility re-run requested in review; exit 0 at `04:09:48`, output token ids verified
identical to the original run — see README "Regeneration / reproducibility check" for the
diff result).

```
1:+ export CUDA_VISIBLE_DEVICES=2
2:+ CUDA_VISIBLE_DEVICES=2
...
+ echo '=== generate_reference (regen v2) start: Sat Jul 25 04:06:43 EDT 2026 ==='
=== generate_reference (regen v2) start: Sat Jul 25 04:06:43 EDT 2026 ===
+ sha256sum /home/muhengl/mpk-qwen35/generate_reference.py
852d74ccc6a294dd08d65bf4e60d95adc642ad18f7c8c2c20e2a609ca817f063  /home/muhengl/mpk-qwen35/generate_reference.py
+ nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
index, memory.used [MiB], utilization.gpu [%]
0, 166318 MiB, 0 %
1, 171386 MiB, 0 %
2, 4 MiB, 0 %
3, 4 MiB, 0 %
4, 171262 MiB, 0 %
5, 4 MiB, 0 %
6, 4 MiB, 0 %
7, 130 MiB, 0 %
```

GPU 2 (the pinned device): **4 MiB, 0%**. Script sha256 on the box at run time
(`852d74cc...`) matches the committed `generate_reference.py`'s sha256 exactly — confirmed
byte-identical before this run (see README provenance entry).

## Step 7, M2-I3 addendum regeneration (top-k persistence), `CUDA_VISIBLE_DEVICES=5`

Two attempts; both pinned the same GPU via the same etiquette check, re-verified immediately
before each launch (agent tool-call transcript, this run was launched via `ssh`, not a persisted
B200 log file the way steps 7/8's `run_*.sh` wrappers are — flagged for the same honesty reason
as the qwen3-8B demo smoke gap above).

Attempt 1 (`CUDA_VISIBLE_DEVICES=5`, pre-run check):
```
index, memory.used [MiB], utilization.gpu [%]
0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 75220 MiB, 100 %
3, 111570 MiB, 100 %
4, 0 MiB, 0 %
5, 0 MiB, 0 %
6, 0 MiB, 0 %
7, 27958 MiB, 0 %
```
GPU 5: **0 MiB, 0%** — well inside etiquette. GPUs 2/3 at 100% util (other users) correctly
skipped; GPU 7 had 27958 MiB resident but 0% util (an idle-but-loaded process) — also skipped
per the `<500 MiB` bound rather than treated as free just because it was idle. This attempt
crashed with a script-level `AssertionError` on `p06-poem` (torch.max vs torch.topk tie-break
disagreement — see README provenance table) before touching the KV-cache-heavy tail of
generation; script was fixed and re-run as attempt 2 below, same GPU, same lock file held
throughout (`~/mpk-qwen35/.gpu-locks/M2-I3.lock`, never released between attempts since this
was the same logical job, not a new one).

Attempt 2 (`CUDA_VISIBLE_DEVICES=5`, immediately before re-launch):
```
index, memory.used [MiB], utilization.gpu [%]
5, 4 MiB, 0 %
```
GPU 5: **4 MiB, 0%** — still free (this box's own process from attempt 1 had fully exited,
freeing the 0 MiB reading back up to a residual 4 MiB baseline typical of this box when idle).
Completed clean, exit 0, all 10 prompts. Lock file removed after this run completed and the
identity diff (README provenance table) passed.
