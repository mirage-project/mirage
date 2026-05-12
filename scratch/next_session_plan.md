# MPK perf — handoff (afternoon session 2026-05-12)

Earlier session morning notes at `scratch/next_session_plan_morning.md`
(Q/KV phase analysis from before this session — still relevant background).

## What landed in this afternoon's session

- `18dc6a4b` — Revert `3a9588cf` AR per-task barrier. Premise was wrong; defaults already had per-team private psync. Saved only 2.1 ms (1.25%), below the 3 ms keep threshold. See `project_ar_per_task_barrier` memory.
- `0246a671` — **topk_sigmoid CORRECTNESS FIX**. Kernel was processing only 8/128 rows for prefill mbt=128; routing_indices stayed 0 for rows 8..127, group GEMM silently skipped them. Wrapped Phase 1-6 in a `for (row_base ...)` outer loop over `ROWS_PER_CTA=8` chunks. DSv3 prefill MoE now processes all 128 tokens correctly. See `project_topk_sigmoid_prefill_bug` memory.
- `41d8e042` — `MPK_MOE_W13_M_SPLIT` / `MPK_MOE_W2_M_SPLIT` env knobs (defaults unchanged at 16 / 14).
- Journal entries: `669e7b40` (A3 revert), `8f0e80d2` (Y-sweep), `1ed325bb` (A2 bisect).

## Outstanding pending tasks

### A series (from start.md original order)

- **A6 router GEMV (`splitk_linear_bf16`)**: 92 μs/call vs vLLM ~3 μs. 1.5 ms/token total. Kernel-level rewrite — hand off to kernel owner.
- **Standalone benches** (LM head + MLA decode TP4): diagnostic. Quick to run (use `tests/runtime_python/blackwell/sm100_linear/`, `sm100_mla_mtp_decode/`).

### B series (system-level)

- **B5 look-ahead pre-fetch** (task #16): builder restructure to schedule next layer's RMSnorm before current AR2 completes. Limited gain (~2-3 ms) per analyzer — downstream is dependency-strict.
- **B3 DTensor view/slice API** (task #15): API extension to support memory-aliased tensor views. Unlocks task #26 (Identity → Dummy) and Path 1 (fuse kv_a+kv_rope).
- **B4 worker queue skip-blocked** (task #17): runtime change to let workers pull next ready task instead of FIFO blocking. Large.
- **B1 Q/KV phase 27% util** (task #14): root cause is fp8_gemm_dense_smallm being persistent (`grid_dim=(num_workers,1,1)`). Theoretical ~4.4 ms recovery if FP8 dense becomes non-persistent or dep-graph refactored. See morning notes for the kv_a+kv_rope fusion idea.
- **B2 EVENT_LAUNCH_TASKS** (task #18): BLOCKED. Never autonomously touch `src/kernel/runtime.cc:1011-1028`. Needs interactive cuda-gdb to fix scheduler race.

### Meeting tasks (2026-05-12, "low priority")

- **#24 PyTorch ref via HF official DSv3**: clone `github.com/deepseek-ai/DeepSeek-V3` to user dir; ditch own impl.
- **#25 Land PR #674** (blocked by #27 fusion): kernel reportedly fine, tests/invoke broken. Current Perfetto shows >1 ms which is unreasonable.
- **#26 Identity → Dummy** (blocked by #15 DTensor view): without alias support, removing the copy would silently break chunked_prefill k_rope correctness.
- **#27 Tensor Fusion (BIG)**: fuse QKV three Linears into one (offset-read consumers); fuse Q rope/nope. Stabilize current branch first.
- **#28 BMM on Q NOPE** (blocked by #27): decode-side KV matmul moves to Q NOPE; needs existing BMM kernel + fused-Q-output kernel from the repo.
- **#29 MLA Decode pointer reads**: switch decode to direct-pointer KV. `_direct_paged_decode_kv` already does this for TP=2/4 page_size=128; meeting task is probably about TP=8 (currently hangs) or removing the buffer fallback.

## Known limitations (do NOT chase)

- **AR2 straggler** ~5-8 ms/token. Structural to EP=2 unbalanced token routing. See `project_ar2_straggler` memory.
- **MoE W13/W2 ~1.4× over vLLM-ideal**: kernel-level. File with MoE-kernel owner.
- **fp8_dense_smallm at N<72** crashes for prefill (was N<56 for decode per af38cf42). Hidden kernel constraint, parked until kernel owner / cuda-gdb session.

## Recommended next steps (data-driven)

1. (Optional) Push the current commits to `origin/dev-v8-rope-prefill-main` after a final Qwen3 + DSv3 layers 0-19 smoke test.
2. **#27 Tensor Fusion** if stable enough — biggest structural improvement available.
3. Otherwise: **#15 DTensor view API**, which unblocks #26 + #27 + Path 1.
4. Standalone benches (#12) are quick and give a sharper kernel-vs-system attribution; do them if a kernel owner asks for "how much of this is calling overhead vs kernel time".
5. Morning notes' **Path 1 (kv_a + kv_rope fusion)** is still viable once #15 lands — see `scratch/next_session_plan_morning.md`.

## Files to read first next session

- `scratch/perf_optimization_journal.md` — running tracker (now contains everything done this session).
- `scratch/mpk_vs_vllm_perf_comparison.md` — per-kernel gap vs vLLM.
- `scratch/next_session_plan_morning.md` — earlier today's analysis (Q/KV root cause + Path 1/2/3 ROI ranking).
- `project_active_workplan` memory — top-level plan with the meeting TODOs slotted in.
- `outputs/dpskv3_perf_post_fixes_140550/` — freshest perfetto trace (post-revert + post-topk fix).
