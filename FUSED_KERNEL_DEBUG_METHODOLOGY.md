# Fused-kernel debugging methodology (hard-won, DSv3 decode fusion 2026-06)

When a fused decode mega-kernel (`attn_block_megakernel` / `ffn_full_megakernel` / any
single-task fusion of a multi-task chain) produces wrong output, debug in **THIS ORDER**.
These steps are written from a real failure where ~9 debug rounds were burned chasing
red herrings before the actual root cause (a missing TP collective) was found. Do NOT
repeat these traps.

## The order of operations (do them in this sequence)

1. **FIRST: full-layer `gate0==gate1` TOKEN-MATCH.** Run the chain (env OFF) vs the fused
   kernel (env ON) at the **full layer count** (`--layers 0-60`) and compare the generated
   tokens. A fused/math-changing kernel is correct iff it is **token-identical to the chain
   for ~tens of tokens** (then FP non-associativity diverges — that's expected, not a bug).
   - **DO NOT** judge correctness from `--layers 0-3` output coherence. Low-layer output is
     GARBAGE regardless of correctness (a known artifact). Judging "it looks degenerate at
     4 layers" as a bug cost 8 rounds chasing a non-bug. Coherence ≠ correctness; token-match
     is the only correctness signal.

2. **If broken: diff against the CHAIN (ground truth), stage-by-stage.** The chain is the
   trusted reference. Instrument BOTH the fused kernel and the chain to dump the SAME tensors
   at the SAME position; the **first stage that diverges from the chain** is the bug.
   - Compare at a **CLEAN, distinct-token position**, not the prompt start — a chat template
     can put a duplicate token at positions 0,1 (e.g. a double-BOS / `tokens[0]==tokens[1]==0`),
     which makes pos-0,1 intermediates *legitimately identical*. Chasing "pos 0,1 produce the
     same c_latent" was a red herring — it's the same token.
   - Measure the **FULL output vector**, never a worker-subset. A tap that summed
     `out[i]; i += gthreads` with `gthreads (34816) > N (7168)` summed only `out[0]` — making
     the genuinely-unverified final stage (o_proj+residual) look "tiny and fine" for rounds.

3. **For TP kernels: the single-rank / cooperative-launch gate CANNOT catch a missing
   cross-rank COLLECTIVE.** A standalone faithful gate is ONE process = TP=1, where every
   AllReduce / reduce-scatter / EP-dispatch is a no-op. So it validates per-rank local math
   but is structurally blind to a missing/wrong collective.
   - **"Garbage at TP>1 but the gate and TP=1 are fine" ⇒ suspect a missing/wrong cross-rank
     reduction FIRST**, before any local-math lead. It is the one hypothesis that explains the
     TP>1-vs-gate discrepancy. (Real root cause here: the fused attention dropped the
     RowParallel o_proj AllReduce — `o_proj.weight` is sharded on dim=1, so each TP8 rank holds
     a [7168] PARTIAL from only its 16/64 heads; the partials must be AllReduce-summed with the
     residual added EXACTLY ONCE. The chain does it in `_fp8_linear` world_size>1; the fused
     kernel fused `out=dot+residual` = correct only at TP=1.)
   - Before fusing/replacing a chain op, check the builder SHARD_RULES + the op's `world_size>1`
     branch: does it do a collective? If so the fused kernel must reproduce it, and add any
     residual/bias ONCE **post-AllReduce** (`Σ_r(partial_r + residual) = Σpartial + N·residual`
     is wrong — bind a zero-residual buffer to the kernel and fold the real residual in the AR).
   - **Disambiguate ranks vs layers** when reading multi-pointer tap output: under
     `mpirun -np N`, **N distinct `out` pointers at ONE step = the N TP RANKS** (separate
     processes each building all layers), NOT N layers. Identical replicated tensors
     (residual/self.x) across those pointers is CORRECT TP behavior. Print
     `runtime_config.my_gpu_id` to disambiguate.

4. **No blind fixes.** Run your OWN in-kernel probe to confirm the mechanism before shipping a
   fix. Two separate "fixes" this campaign were proven data no-ops by review (e.g. an
   input/output-binding "fix" — MPK input_ptrs and output_ptrs for a root cuda_tensor resolve
   to the SAME physical address, so the write already persisted). A fix that isn't confirmed by
   your own probe is a guess.

5. **Intermediate "anomalies" are usually red herrings — verify with the right metric.**
   - Equal `sum|.|` (L1) across tokens does NOT mean equal vectors — RMSNorm equalizes
     magnitudes; two distinct vectors can share a magnitude. Use a **vector diff (cosine + sum
     of abs diff)**, not `sum|.|`, to claim "identical".
   - Tiny leading dims are often just a tiny layernorm WEIGHT on those dims, identical for every
     row — not degenerate data.

## Gate-fidelity rules (the gate must match the deployed regime — 6 classes)

A faithful gate must reproduce ALL of these or it will pass a kernel that fails in MPK:
1. **COLD-L2** — flush ≥256MB before every timed iter; target the COLD number (MPK streams
   weights cold each layer; a warm gate over-states the win ~2.5×).
2. **FULL-GRID geometry** — 136 worker-CTAs via cooperative launch + the right thread count
   (256 for attn, 512 for FFN), NOT `<<<1,256>>>`.
3. **PRODUCTION scale/weight FORMAT** — per-128-block fp32 `weight_scale_inv` read as plain
   fp32 `[n>>7][g]`, not a self-invented per-row UE8M0.
4. **RECURRENCE** — for a stateful kernel (KV cache), exercise real multi-step decode (write@N
   read-as-history@N+1), not a single pre-filled call.
5. **PER-RANK EP/TP SHARDING + typical active count** — at TP8 EP2 the routed FFN sees 128
   local experts and ~4 of top-8 (not all-256-local=8); measure at the deployed per-rank size.
6. **CROSS-RANK COLLECTIVES** — see step 3 above. A single-process gate cannot test them;
   the fused kernel must reproduce any AllReduce/reduce-scatter/EP-dispatch the chain op does.

## Status of the fused decode (2026-06-25)
Both `MPK_DSV3_ATTN_MEGAKERNEL=1` and `MPK_DSV3_FFN_FULL_MEGAKERNEL=1` are committed
(`52ed6e64`), env-gated default-OFF, in-MPK token-identical to the chain at TP8 EP2.
FFN-full is ~−0.8ms/tok (faster); attn-mega is +0.69ms/tok (correct but SLOWER — fusion
loses the chain's multi-task overlap). The fusion is correctness-parity, not yet a perf win;
beating the chain needs FASTER kernels (the attn/FFN/AllReduce optimization agents) — see the
3-module split: Attention, FFN, AllReduce.
