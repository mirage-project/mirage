# Qwen3.5-35B-A3B-FP8 on MPK — v1 architecture decision (M1 exit)

**Decides:** the complete v1 design for serving `Qwen/Qwen3.5-35B-A3B-FP8` (text decode, greedy,
single B200, batch 1..16) on MPK, and the M2 issue decomposition that implements it.

**Source shorthand** (all claims trace to one of these or to a `file:line` verified on branch
`qwen3-5_support`, HEAD `2c87a75`):

| tag | document |
|---|---|
| [VG] | `docs/qwen35/vllm-graph.md` (M1-I1, vLLM decode graph + FP8, primary reference) |
| [SG] | `docs/qwen35/sglang-notes.md` (M1-I2, secondary) |
| [MG] | `docs/qwen35/mpk-gaps.md` (M1-I3, MPK capability map) |
| [REF] | `demo/qwen3_5/accept/reference/README.md` (M1-I5, pinned HF reference + vLLM smoke) |
| [GOAL] / [CONSTR] | `.pm/goal.md` / `constraint.md` (agent repo) |

---

## 0. Decision summary

1. **Alignment: vLLM's decode-graph structure, HF's numerics.** Op decomposition, state layouts,
   and the packed GDN decode path follow vLLM [VG §2]; the bit-level target for every kernel is
   the HF `transformers` reference (AC-3 pins HF, not vLLM) [GOAL AC-3, REF].
2. **GDN = 2 new task types** (conv1d state update; fused delta-rule recurrence + gated norm),
   per-request state pools indexed by request slot, fp32 `S` state (mandatory, from
   `mamba_ssm_dtype`), lifecycle handled kernel-side via `step == 0` predication — no
   `prepare_next_batch` changes (§3).
3. **Prefill = chunked in-kernel**, the same task family as decode, runtime `Q_LEN` per the
   proven MLA pattern [MG §3.2 Option 1]. Corrected per-batch-size cost model (§8): workload
   constraint `I ≤ O/4` (binding at B=16, mbt=16); the pinned (256, 1024) sits exactly on that
   boundary and holds conditional on `t_pf(16-tok iter) ≤ t_dec(16-req step)` — a falsifiable
   prediction probe P8 tests early-M2 on the shipped Qwen3-8B path.
4. **FP8 ruling:** MPK's dense UE8M0 requantization path is **rejected for this model**
   (two-engine evidence). v1/M2 runs **dense projections in bf16** (dequantized at load,
   probe-gated); routed experts stay **fp8 with preserved block scales**. M3 restores fp8 dense
   via a fp32-scale block GEMM — the semantics class both reference engines themselves run;
   its full-set token-exactness is a hypothesis the M2 harness and probe P10 test (§6).
5. **Attention:** cherry-pick `5715c6f`; `[q|gate]` sliced in-kernel + σ-gate epilogue (zero new
   inputs, 7-input limit respected); `MAX_TOKENS` decoupled from mbt at the one sm100
   registration site with an in-kernel Q-loop (4 queries/pass) so one build serves decode bs 16
   and prefill chunks; partial RoPE via load-time column permutation (probe-gated, kernel
   untouched) (§4).
6. **MoE:** DSV3 FP8 chain at our shapes (probe-gated for the untested 512-intermediate regime),
   existing softmax-top-k router (semantics probe-gated), shared expert as bf16 dense MLP + one
   new sigmoid-gate-mul-add task (§5).
7. **Budget:** 3 firm + 2 reserved TaskType ids of 14 free; per-step HBM traffic 5.0 GB (B=1) to
   22.1 GB (B=16) in v1, floors 0.63–2.76 ms at nominal B200 bandwidth — ≥50× headroom vs the
   vLLM bs=1 smoke datapoint; the real AC-4 fight is B∈{8,16} where both engines converge on the
   same compulsory bytes (§9).

---

## 1. Chosen alignment and numeric target

**Structural alignment: vLLM.** vLLM's decode graph is fully mapped with `file:line` evidence
[VG §2] and its op decomposition matches MPK's task granularity almost one-to-one (per-GEMM
tasks, separate routing, fused attention with in-task KV append). We adopt:

- the packed GDN decode path (`conv1d_update` → fused recurrence with in-kernel q/k L2 norm,
  g/β computation, and state update) [VG §2.1.3–2.1.4] — sglang's default lands on the same
  algorithm family (Triton `packed_decode`) [SG §1 row 1], so this is the convergent shape;
- the deferred-residual structure (add folded into the next norm / GEMM-with-residual)
  [VG §1.3], realized with MPK's `linear_with_residual` + `mul_sum_add` residual inputs;
- the KV-append-before-read invariant with `seq_len` counting the current token [VG §2.2.5].
  MPK's `paged_attention_sm100` already appends KV in-task and derives
  `seq_len = history + chunk` at runtime [MG §1, §3.1], so no separate KV-write op is needed —
  the invariant, not vLLM's two-op split, is what must hold;
- the router semantics `RenormalizeNaive` = full 256-wide fp32 softmax → top-8 (lower index wins
  ties) → renormalize [VG §2.3.2];
- the always-on shared expert scaled by `sigmoid(x·W_sg)` applied after `down_proj`
  [VG §2.3.3].

**Numeric target: HF `transformers`.** AC-3 compares MPK tokens against the pinned
`reference_outputs.json` produced by HF running the FP8 checkpoint on the Triton
`finegrained-fp8` backend [REF]. Where vLLM and HF differ in low-order bits (e.g. vLLM's packed
GDN kernel round-trips `β = σ(b)` through bf16 while its generic kernel does not
[VG §2.1.4, §6 g.10]; the FlashInfer MoE cubin's internal accumulate points are unobservable
[VG §2.3.4]), **HF's behavior is the one to match**; probe P6 (§14) extracts HF's exact op
order and dtypes as the oracle. vLLM's 64/64 exact-token agreement with HF on p01 [REF] is used
as evidence about *semantics classes*, not as a reference to bit-match.

Nothing structural is adopted from sglang; it corroborates the vLLM shape and contributes
M3-era ideas (warp-specialized overlap of small/large GEMMs, [SG §4.4]) plus one baseline
caution: sglang's real GDN/attention decode kernels under default launch args are Triton, not
the "fast" SM100 paths [SG §1 rows 1, 3] — relevant only if an sglang comparison is ever run.

---

## 2. Decode-step MPK task graph (op-by-op, with shapes)

Conventions: `B` = tokens this iteration (= active requests in pure decode, ≤ mbt);
`mbr` = max batched requests (16 for the full-batch build); weights `[N,K]`, GEMMs compute
`x @ W.T`. "NEW" marks the three new task types (§10). Dtypes reflect the v1 ruling (§6):
dense GEMMs bf16, routed experts fp8. Existing task/layer names from the inventory [MG §1].

### 2.0 Weight-loading transforms (builder, load time)

| transform | what | why |
|---|---|---|
| Gemma fold | `w_eff = 1 + w` for `input_layernorm`, `post_attention_layernorm`, final `norm`, `q_norm`, `k_norm` | exact, zero-cost; MPK kernels compute `x·rsqrt(·)·w` [MG Gap 5]; the checkpoint stores zero-centered weights [VG §1.2] |
| **No** fold on GDN `linear_attn.norm` | used as-is (ones-init, F32 in ckpt, no `+1`) | the one non-Gemma norm [VG §1.2, §6 g.1] |
| QKVG concat | stack `q_proj [8192,2048]` ‖ `k_proj [512,2048]` ‖ `v_proj [512,2048]` → `[9216,2048]`, preserving the per-head `[q(256)|gate(256)]` interleave already inside `q_proj` | [VG §2.2.1–2.2.2, §5.2]; MPK attention reads one packed QKV tensor [MG Gap 2] |
| RoPE permutation | permute head_dim columns of `q_proj`(q half only)/`k_proj`/`q_norm`/`k_norm` so Qwen's rotated pairs `(j, j+32)`, j<32, land on MPK's NeoX pairs `(j, j+128)`; cos/sin table = real values at rotated slots, `cos=1, sin=0` elsewhere, θ=1e7 | zero-kernel-change partial RoPE [MG Gap 4 route 2]; exactness gated by probe P4 (§14) |
| GDN projections kept separate | load `in_proj_qkv [8192,2048]`, `in_proj_z [4096,2048]` as two GEMMs; concat `in_proj_b`+`in_proj_a` → `ba [64,2048]` | the checkpoint ships them separately in plain `[q|k|v]`/`[b|a]` order [VG §5.1–5.2, §6 g.5]; two GEMM outputs avoid all strided-view plumbing (deviation from vLLM's load-time fusion, §13.5) |
| dense dequant (v1 only) | `W_bf16 = (W_fp8.float() * scale_inv_128x128_expanded).bfloat16()` for all dense fp8 projections | §6 ruling; exact per-block scaling, bf16 rounding once |
| expert weights | keep checkpoint `[gate; up]` packing and per-expert 2-D → stack to `w13 [256,1024,2048]`, `w2 [256,2048,512]` fp8 + `[256,8,16]`/`[256,16,4]` fp32 scales | [VG §2.3.4, §5.3]; do **not** copy vLLM's post-load W31/BlockMajorK shuffle [VG §7 row 11] |
| `A_log`+`dt_bias` pack | one `[2,32]` fp32 tensor per layer | saves an input slot on the recurrence task |
| vocab | `padded_vocab = 248320` (no padding; 248320 = 970·256) | §7 |

### 2.1 Prologue / epilogue (once per step)

| # | op | task | shapes | dtype |
|---|---|---|---|---|
| P1 | embed gather | `embed_layer` (ampere kernel, works on sm100 [MG §1, Gap 9]) | `tok[B] → [B,2048]`, table `[248320,2048]` | bf16, no scale [VG §2.4] |
| E1 | final norm | `rmsnorm_layer` (Gemma-folded) | `[B,2048]` | bf16, eps 1e-6 |
| E2 | lm_head | `splitk_linear_layer` | `[B,2048]×[248320,2048] → [B,248320]` | bf16 (never quantized [VG §2.4]) |
| E3 | greedy argmax | `argmax_partial_layer` + `argmax_reduce_layer` | `[B,248320] → [B]` | tie behavior must be lowest-index (verify vs HF in M2-I1) |

### 2.2 GDN linear-attention layer (×30; i where (i+1) % 4 ≠ 0 [VG §1.1])

| # | op | task | shapes (per iteration) | notes |
|---|---|---|---|---|
| 1 | pre-norm | `rmsnorm_layer` | `h[B,2048] → x[B,2048]` | Gemma-folded, eps 1e-6 |
| 2 | qkv proj | `linear_layer` bf16 | `x × [8192,2048] → qkv[B,8192]` | v1 bf16 (§6); layout `[q(2048)|k(2048)|v(4096)]` matches the recurrence offsets [VG §5.2] |
| 3 | z proj | `linear_layer` bf16 | `x × [4096,2048] → z[B,4096]` | |
| 4 | ba proj | `linear_layer` bf16 | `x × [64,2048] → ba[B,64]` | always bf16 — both shards in `modules_to_not_convert`, N=32 < block_n [VG §6 g.7] |
| 5 | conv update | **NEW `gdn_conv1d_sm100`** (id 234) | in: `qkv[B,8192]`, `W[8192,4]`, state pool `[mbr,3,8192]` bf16; out: `qkv_c[B,8192]` | grid `(mbr,1,1)`; per channel: window `[S₀,S₁,S₂,x]` (or `[state‖chunk]` FIR for Q_LEN>1), `y = silu(Σ Wⱼ·winⱼ)` (no bias), state ← last 3 inputs; fp32 accumulator [VG §2.1.4] |
| 6 | recurrence + gated norm | **NEW `gdn_recurrent_sm100`** (id 237) | in: `qkv_c`, `ba`, `alog_dtbias[2,32]` fp32, S pool `[mbr,32,128,128]` **fp32**, `z[B,4096]`, `norm_w[128]` f32; out: `g_out[B,4096]` (6 in / 1 out ≤ 7/3) | grid `(32, mbr, 1)` — one task per (v-head, request slot); math in §3.2 |
| 7 | out proj | `linear_with_residual_layer` bf16 | `g_out × [2048,4096] + h → h'[B,2048]` | |

### 2.3 Full-attention layer (×10; i ∈ {3,7,…,39})

| # | op | task | shapes | notes |
|---|---|---|---|---|
| 1 | pre-norm | `rmsnorm_layer` | `[B,2048]` | |
| 2 | qkvg proj | `linear_layer` bf16 | `x × [9216,2048] → qkvg[B,9216]` | `[16×[q(256)|gate(256)] ‖ k(512) ‖ v(512)]` [VG §2.2.1] |
| 3 | attention | `paged_attention_sm100` **modified** (same TaskType, params-gated variants) | `qkvg`, paged KV `(pages, 2, page, 512)` bf16, permuted cos/sin, folded q/k norm weights → `attn[B,4096]` | in-task, in order: per-head q slice at stride 512 / gate at +256; Gemma q/k norm (folded weights, variance over full 256); NeoX RoPE on permuted layout (= partial RoPE [0:64], θ=1e7); **KV append for the current token, then read with seq_len incl. t** [VG §2.2.3–2.2.5]; softmax scale 1/16; epilogue `out·σ(gate)`; Q-loop ≤4 queries/pass (§4.3) |
| 4 | o proj | `linear_with_residual_layer` bf16 | `attn × [2048,4096] + h → h'[B,2048]` | |

Zero new inputs on the attention task: the gate rides input 0 (the 7-input limit is already
saturated at `num_inputs = 7`, `task_register.cc:2039` [MG Gap 2]).

### 2.4 MoE block (×40 — every layer [VG §2.3])

| # | op | task | shapes | notes |
|---|---|---|---|---|
| 1 | post-attn norm | `rmsnorm_layer` | `h'[B,2048] → x[B,2048]` | |
| 2 | router GEMM | `linear_layer` bf16 | `x × [256,2048] → logits[B,256]` | never quantized (`quant_config=None`) [VG §2.3.1] |
| 3 | routing | `moe_topk_softmax_routing_layer` | `logits → w[B,8]` fp32, `ids` | must equal RenormalizeNaive: fp32 softmax over all 256 → top-8, lower-index ties → renorm [VG §2.3.2]; kernel takes 256 threads / 8 warps, `NUM_EXPERTS=256` passes the power-of-2 assert [MG Gap 7]; **probe P5** |
| 4 | act quant | `quantize_fp8_layer` (fp32-scale variant) | `x → x_q[B,2048]` e4m3 + scales `[B,16]` fp32 | group 128, eps 1e-10, absmax/448, `x/scale` then clamp ±448, RN-even — the exact contract [VG §3.4]; MPK's kernel implements this scheme [MG §2.2] |
| 5 | experts w13 | `moe_w13_fp8_layer` | `[≤8B,2048] × w13[256,1024,2048]` fp8 → `[B,8,1024]` bf16 | preserved block scales via `repeat_interleave` expansion [MG §2.2.1]; grid `y ≤ 8` (1024/128); **probe P2** for the untested inter-512 regime [MG Gap 7] |
| 6 | silu·mul | `moe_silu_mul_layer` | `[B,8,1024] → [B,8,512]` | `silu(gate)·up`, `[gate;up]` row order preserved from checkpoint |
| 7 | act quant | `quantize_fp8_layer` | `[B,8,512] → fp8 + scales` | |
| 8 | experts w2 | `moe_w2_fp8_layer` | `× w2[256,2048,512]` fp8 → `[B,8,2048]` bf16 | K=512 → 4 k-tiles vs `num_ab_stages=8` pipeline — the hang-risk regime; **probe P2** [MG §2.3, Gap 7] |
| 9 | shared gate_up | `linear_layer` bf16 | `x × [1024,2048] → [B,1024]` | shared expert is a *dense* projection pair → bf16 in v1 (§6) |
| 10 | shared silu·mul | `silu_mul_layer` | `[B,1024] → [B,512]` | |
| 11 | shared down | `linear_layer` bf16 | `[B,512] × [2048,512] → s[B,2048]` | |
| 12 | shared gate + residual | **NEW `sigmoid_gate_mul_add_sm100`** (id 238) | in: `x[B,2048]`, `W_sg[1,2048]` bf16, `s[B,2048]`, `h'[B,2048]`; out: `r'[B,2048]` (4/1) | `r' = h' + σ(x·W_sgᵀ) ⊙ s`; gate scalar from the *pre-MLP* hidden state, applied after down_proj [VG §2.3.3]; computes the N=1 GEMV inline (a `linear_layer` at N=1 is degenerate [MG Gap 8]) |
| 13 | combine | `moe_mul_sum_add_layer` | `Σⱼ wⱼ·yⱼ + r' → h''[B,2048]` | router weights fp32 into the reduce (HF ordering pinned by oracle P6) |

Steps 9–12 have no dependency on 3–8 and interleave freely in the event graph — the megakernel
gets sglang/vLLM's dual-stream shared-expert overlap [VG §2.3.3, SG §1 rows 5–6] for free.

---

## 3. GDN per-request state design

### 3.1 State tensors and dtype ruling

Per GDN layer, two pools attached at build time, **indexed by request slot** — never by
`blockIdx` (worker id ≠ data row; the `8b19538` porting rule [CONSTR §2a]) and never by page:

| pool | shape | dtype | bytes/slot | total (mbr=16, ×30 layers) |
|---|---|---|---|---|
| conv state | `[mbr, 3, 8192]` | bf16 | 48 KiB | 22.5 MiB |
| recurrent `S` | `[mbr, 32, 128, 128]` | **fp32** | 2 MiB | 0.94 GiB |

**fp32 for `S` is mandatory, not a tuning choice**: the checkpoint sets
`mamba_ssm_dtype: "float32"` and both HF and vLLM honor it [VG §1.1, §2.1.5, §6 g.6] — the
AC-3 reference's recurrence is fp32-state math; a bf16 state would diverge from the reference
by construction. Cost: 4 MiB read+write per request per layer per step ⇒ **128.8 MB per token
per step across 30 layers** [VG §4.7] — carried in the §9 budget. Conv state stays bf16 (model
dtype, matching vLLM's resolution [VG §2.1.5]; P6 confirms HF).

MPK sizes the two caches independently and skips vLLM's 1056-token hybrid page alignment
entirely — that machinery exists only to unify two allocators [VG §4.3, §4.7]. One slot per
request slot; no paging, no `preprocess_mamba` analog (align-mode snapshotting is a
prefix-caching feature we don't have [VG §4.7]).

### 3.2 Task-level math (bit-parity targets)

`gdn_recurrent_sm100`, one task per (v-head hv, request slot), Q_LEN tokens sequential, S tile
(64 KiB fp32) resident in smem for the whole task:

```
i_h = hv // 2                                   # GVA: 2 v-heads per k/q head [VG §2.1.4]
q = qkv_c[t, i_h·128  : +128];  k = qkv_c[t, 2048 + i_h·128 : +128]
v = qkv_c[t, 4096 + hv·128 : +128]
q ← q/√(Σq²+1e-6);  k ← k/√(Σk²+1e-6);  q ← q·128^-0.5          # in-kernel L2 norm
g = −exp(A_log[hv]) · softplus(a[t,hv] + dt_bias[hv])            # softplus threshold 20
β = σ(b[t,hv])                                                    # dtype path per HF oracle P6
S ← S·e^g;  S ← S + β·(v − S·k) ⊗ k;  o = S·q                    # all fp32
# fused epilogue (replaces a separate RMSNormGated task):
out[t, hv·128 : +128] = (o · rsqrt(mean(o²)+1e-6) · norm_w) · silu(z[t, hv·128 : +128])
```

Everything above is the vLLM packed-decode kernel's math [VG §2.1.4] with the gated norm
(`norm_before_gate=True`, silu, weight shared across heads, **no** Gemma `+1`
[VG §2.1.6]) folded into the epilogue — each task owns its head's full 128-vector, so the norm
is free in registers. The `β` bf16 round-trip question and the exact softplus/decay dtypes are
pinned by the HF oracle (P6), since vLLM's own two kernels disagree [VG §6 g.10].

`gdn_conv1d_sm100`: depthwise FIR over `[state(3) ‖ chunk(Q_LEN)]` per channel — token-parallel
even in prefill chunks (no sequential dependency within a chunk); silu applied; final state =
last 3 inputs of the chunk. Output goes to a **separate** buffer `qkv_c` rather than vLLM's
in-place overwrite [VG §2.1.4] — cleaner event-graph semantics, +128 KB/step, nothing else.

### 3.3 Lifecycle: kernel-side, zero runtime changes

`reset on admission / persist across steps / release on completion` maps to **one predicate**:
both GDN tasks treat their state as zero when `runtime_config.step[slot] == 0` (first prefill
chunk of whichever request occupies the slot) instead of loading it, and write the updated
state unconditionally. Slot reuse by a later request re-zeros implicitly because its `step`
restarts at 0. `step[]` is already injectable into generated task code
(`task_register.cc` precedents [MG §3.1]); per-slot addressing uses the proven
`base + request_slot × stride` MLA pattern (`task_register.cc:3768` [MG §3.2]).

This is the [MG §3.2] recommendation, chosen over a `prepare_next_batch` extension for two
reasons: `prepare_next_batch` is a serial, single-thread section that is already the batch-8
latency knee on Qwen3-8B [MG §8 risk 4], and it is CI-protected shared runtime [MG §6.2]. The
issue contract's "reset/preserve in prepare_next_batch" is therefore implemented *against* that
literal placement — deviation recorded in §13.1.

Inactive slots (`Q_LEN == 0` from `qo_indptr`) early-return — the standard dead-task masking
every per-request task uses [MG §3.1].

---

## 4. Attention plan

### 4.1 Cherry-pick `5715c6f` (required, not sufficient)

Picks cleanly onto `qwen3-5_support` (verified by worktree cherry-pick, zero conflicts
[MG §5]). It makes `S_O_BUFFER` smem independent of `MAX_TOKENS×NUM_QO_PER_KV`; at our shape
(`HEAD_DIM=256`, `NUM_QO_PER_KV=8`) the post-pick arena is 170/178/196/232/304 KiB at
MAX_TOKENS 1/2/4/8/16 against a 201 KiB budget ⇒ **mbt ≤ 4 per task** [MG §5]. Batch 16 needs
§4.3.

### 4.2 Output gate + fused QKVG (7-input limit)

The gate is not a new input: the checkpoint packs `[q(256)|gate(256)]` per head inside
`q_proj` [VG §2.2.2, §6 g.3], so the fused QKVG tensor (§2.0) carries it in input 0. Kernel
changes, all gated behind a new `params[]` flag so the default-path generated code string —
and therefore `register_task_variant()`'s dedup and the Qwen3-8B CI output — stays
byte-identical [MG Gap 2 blast radius, §6.2]:

- q addressing: per-head stride `2·head_dim`, gate at `+head_dim` (the fused vLLM kernel's
  exact addressing [VG §2.2.2]);
- epilogue: `out ← out · σ(gate)` before the store — full sigmoid, applied outside the softmax,
  not a sink [VG §2.2.4].

Fallback if the fused route destabilizes CI: a standalone elementwise gate task (reserved id
240), which is what vLLM itself does structurally [MG Gap 2 (ii)].

### 4.3 MAX_TOKENS / mbt decoupling + in-kernel Q-loop

Today `max_tokens = input_ops[0]->dtensor.dim[0]` = the activation's leading dim = mbt, even
though a decode task handles exactly one token of one request [MG Gap 3]. All six derivation
sites were re-verified this session and **all six are paged-attention-family registrations**:
`register_paged_attention{,_hopper,_sm100,_split_kv_sm100,_split_kv_merge_sm100,_split_kv_hopper}_task`
(`task_register.cc:384, 1097, 2052, 3468, 3543, 4000`). Only the sm100 one (`:2052`, plus
`:3468/:3543` iff split-KV is wired) is on the B200 path — narrower blast radius than the
"6 CI-protected sites" framing suggests [CONSTR §2a; deviation §13.3].

v1 design: a new optional param `max_tokens_per_pass` (default: old derivation ⇒ byte-identical
Qwen3-8B codegen; precedent: the `Q_LEN_OVERRIDE`/`TAIL_OFFSET` optional params
[MG Gap 3]). The Qwen3.5 builder sets it to **4**; the kernel loops
`ceil(Q_LEN / 4)` passes over the request's queries, reusing the same smem arena. Effects:

- decode at any mbt: Q_LEN = 1, one pass — smem is the MAX_TOKENS=4 instantiation (196 KiB,
  fits) regardless of mbt ⇒ **bs 16 works with one build**;
- prefill chunks up to mbt tokens: ≤ 4 passes at mbt=16; KV tiles re-stream per pass — ≈ 20 MB
  at the pinned workload's deepest prefill chunk, irrelevant (§8.2);
- no second task type, no dual-dispatch [MG §3.2 Option 2 held in reserve].

Probe P3 re-validates the smem table by standalone `nvcc -arch=sm_100a` instantiation before
the builder hard-codes 4 — the same method `5715c6f` used [MG §5].

### 4.4 Partial RoPE (64 of 256) — load-time permutation first

MPK's rotation pairs column `i` with `i ± 128`; Qwen needs `(i, i+32)` on the first 64 columns
only, so identity-padding cos/sin alone is wrong [MG Gap 4]. The v1 route is the **load-time
column permutation** (§2.0): `q·k` is invariant under a permutation applied identically to q
and k rows, RMSNorm over head_dim is permutation-equivariant with a permuted weight, and
`cos=1/sin=0` makes the un-rotated pairs an identity rotation. Zero kernel change, zero CI
exposure. It is *unverified* [MG Gap 4] — probe P4 proves it numerically (expected exact in
fp32) **before** the builder relies on it; the fallback is the `ROTARY_DIM` template-parameter
kernel route (CI-protected edit, params-gated).

The mRoPE wrinkle is a no-op for us: text-only positions reduce `MRotaryEmbedding` to plain
partial NeoX RoPE with θ=1e7 [VG §2.2.3] — MPK implements exactly that, on the permuted layout.

### 4.5 KV cache

`(pages, 2 kv-heads, page_size, 512)` bf16 with K‖V packed in the last dim, matching the
FlashInfer HND view MPK's kernel already uses; head_size 256 is supported [VG §4.2]. KV is
**bf16 on both sides** — the checkpoint ships no k/v scales and vLLM's cache resolves to bf16
[VG §3.2, §3.7.5 item 7]; an fp8 KV cache would be a different precision contract (and MPK has
none anyway [MG §2.4]). Sizing: 2 KiB per token per layer [VG §2.2.5]; at the pinned
workload's max_seq (256+1024+template ≈ 1536), bs 16: 480 MiB resident; late-decode KV read
≈ 26 MB/request/step (< 2 % of B=16 step bytes — §9.2 note).

---

## 5. MoE plan

**Router.** `moe_topk_softmax_routing_layer` structurally fits (256 experts = power of two,
`renormalize=true` unconditional [MG Gap 7]). What is *not* yet established is exact semantic
equality with RenormalizeNaive — order (softmax-then-top-k, not the reverse [VG §6 g.4]), fp32
softmax, lower-index tie-breaking. Probe P5 settles it with crafted tie rows; a mismatch costs
one new router task (reserved id 239), not a redesign.

**Routed experts.** DSV3's FP8 chain at our shapes: `[256, 1024, 2048]` w13 / `[256, 2048,
512]` w2, preserved 128×128 block scales expanded per-row [MG §2.2.1]. Two shape risks, both
probe P2: (i) w2's K=512 gives `fp8_k_tile_count = 4` against a pipeline whose depth was raised
to 8 stages precisely because 4 stages hung at count 8 — our regime is *below* anything ever
run [MG §2.3, Gap 7]; (ii) the kernel's *internal UE8M0 conversion* of the expanded scales
(`builder.py:945` [MG §2.2]) must be shown numerically harmless for **this** checkpoint's
scales — the same E8M0 family flagged on the dense path (§6), so P2 compares against an
fp32-block-dequant reference and looks for systematic per-row bias beyond activation-quant
noise, not just "runs". If P2 fails numerically, the fix is scale handling inside the grouped
GEMM (fp32-scale application), not bf16 experts — experts dominate step bytes at B≥2 (§9) and
doubling them would genuinely threaten AC-4.

**Shared expert.** Dense MLP (bf16 in v1 per §6) + the one new `sigmoid_gate_mul_add_sm100`
task (§2.4 #12). DSV3's builder pattern feeds the shared output through `mul_sum_add`'s
residual argument [MG Gap 8]; we insert the gate multiply one hop earlier.

**Tuning prior.** vLLM ships an exact-shape Triton config (`E=256, N=512, B200`:
`BLOCK_M=16, BLOCK_N=128, BLOCK_K=128`, `GROUP_M=64` at M=16) — dead weight on vLLM's default
path but the best available tile prior for M3 tuning of our grouped GEMM [VG §3.6.5].

---

## 6. FP8 mapping and the dense-path ruling

### 6.1 Per-GEMM dtype map

| GEMM (per layer type) | `[N,K]` | ckpt dtype | MPK path class | **v1** | **M3 target** |
|---|---|---|---|---|---|
| GDN `in_proj_qkv` / `in_proj_z` | `[8192,2048]` / `[4096,2048]` | fp8 block | dense | **bf16 (dequant)** | fp8, fp32-scale |
| GDN `in_proj_ba` | `[64,2048]` | bf16 | — | bf16 | bf16 (pinned: `modules_to_not_convert`, N=32<128 [VG §6 g.7]) |
| GDN `conv1d` | `[8192,4]` depthwise | bf16 | — | bf16 | bf16 |
| GDN `out_proj` | `[2048,4096]` | fp8 block | dense | **bf16** | fp8, fp32-scale |
| attn `qkv_proj` (q‖gate‖k‖v) | `[9216,2048]` | fp8 block | dense | **bf16** | fp8, fp32-scale |
| attn `o_proj` | `[2048,4096]` | fp8 block | dense | **bf16** | fp8, fp32-scale |
| router `gate` / `shared_expert_gate` | `[256,2048]` / `[1,2048]` | bf16 | — | bf16 | bf16 (pinned [VG §2.3.1]) |
| shared expert `gate_up` / `down` | `[1024,2048]` / `[2048,512]` | fp8 block | dense | **bf16** | fp8, fp32-scale |
| routed `w13` / `w2` | `[256,1024,2048]` / `[256,2048,512]` | fp8 block | MoE grouped | **fp8, preserved scales** | same |
| `lm_head` / `embed_tokens` | `[248320,2048]` | bf16 | — | bf16 | bf16 (pinned) |
| attention QKᵀ/PV, GDN recurrence | — | — | — | bf16 io / fp32 state+accum | same |

Activation quantization (MoE path, and dense in M3) must reproduce vLLM/HF's primitive exactly:
group 128, `absmax = max(|x|, 1e-10)`, `scale = absmax/448`, `x/scale` (division, not
reciprocal), clamp ±448 *before* the e4m3 cast, RN-even [VG §3.4 items 1–4]. MPK's fp32-scale
quantize variant implements this scheme [MG §2.2]; the packed-UE8M0 variant is not used in v1.

### 6.2 The ruling on the dense path

**MPK's existing dense FP8 path is rejected for this model, in any milestone.** It discards the
checkpoint's 128×128 fp32 block scales and re-quantizes weights under per-row power-of-two
UE8M0 scales — "a real numeric delta, both directions" [MG §2.2.1]. Two independent engines
mark exactly this scale-format family unsafe on B200 for this architecture: HF transformers'
DeepGEMM backend **crashes** on our GDN `in_proj_qkv` and ships
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` with a documented token-drift caveat [REF], and vLLM
**auto-disables** DeepGEMM for `qwen3_5_moe_text` on sm100 citing "accuracy degradation with
DeepGemm's E8M0 scale format" [VG §3.5, §6 g.9]. Against a zero-tolerance exact-token gate
[GOAL AC-3, MG §6.5], adopting the one scale scheme both reference engines refuse is
indefensible. (sglang, which lacks the guard, is a cautionary tale, not a counter-example
[SG §3.2–3.3].)

**v1/M2 runs dense projections in bf16** (dequantized at load): zero kernel work (the bf16
`linear_sm100` family is CI-exercised [MG §6.2]), and it removes an entire error class from the
GDN bring-up — when tokens mismatch during integration, the dense GEMMs are known-exact and
suspicion concentrates on the new kernels. Cost, from the per-step budget (§9): **+1.41 GB/step
at every batch size** (dense 2.47 → 3.88 GB) — +39 % of total step bytes at B=1 but only
**+6.8 % at B=16**, because experts + GDN state dominate at large B. Feasibility survives at
every batch size (§9), so this is a correctness-first staging decision, not a performance
gamble. It is gated by **probe P1** (HF with dense-dequantized-to-bf16 vs the pinned reference,
all 640 positions): pass ⇒ GO; fail at wide margins ⇒ v1 jumps directly to the M3 design below.

**M3 restores fp8 dense with fp32 block scales** (CUTLASS/DeepGEMM-promotion style: fp8 MMA,
fp32 accumulation, per-128-k-tile `a_scale·b_scale` application — the semantics class of
vLLM's `CutlassFp8BlockScaledMMKernel` [VG §3.5] and HF's Triton `finegrained-fp8` [REF]).
Evidence for that class, stated at its actual strength: the one direct datapoint — vLLM
(CUTLASS fp32-scale dense + FlashInfer fp8 MoE) matching HF **64/64 on p01 only** [REF] —
supports that specific configuration on that prompt. "The fp32-scale class is token-exact over
the full 640-position set" is a **hypothesis, not a result**; its tests are the M2 harness's
margin data accumulating on every run (§12) and probe P10's CUTLASS-vs-HF-Triton comparison on
real checkpoint weights (§14), both of which land before M3 commits kernel work. What the
datapoint does license: an M3 fp32-scale implementation that diverges *grossly* from the
reference indicates an MPK bug rather than an intrinsically dead numeric class — which is what
makes deferral safe. The end state satisfies the goal's fp8-both-sides framing [GOAL]; the M2
bf16 staging is surfaced for user batch review (§13.4).

Precision boundary stays MPK's existing contract: fp8 inside GEMMs, bf16 between ops, GDN and
attention on the bf16 side [MG §2.4].

---

## 7. Vocab 248320

`padded_vocab_size` is hardcoded to 153600 in the Qwen3 builder (`builder.py:39`) and must not
be "fixed" in place — it sizes Qwen3-8B's argmax tensors [MG Gap 6]. The Qwen3.5 builder
computes its own: **248320 needs no padding** (= 3880·64 = 970·256, satisfying the 256-divisor
grid constraint [MG Gap 6; VG §1.1]). `lm_head` and `embed_tokens` stay bf16 (not `LinearBase`
in vLLM's terms; `modules_to_not_convert` besides [VG §2.4, §3.2]) — 1.02 GB of per-step
lm_head traffic is compulsory on both engines.

---

## 8. Prefill strategy and the AC-5 feasibility model

> **REVISION (review cycle 1).** The previous version's "AC-4 ⟹ AC-5" derivation was invalid:
> it priced every prefill iteration at one B=16-decode-equivalent step and then substituted the
> *same-batch* decode time `t_mpk(B)` — unsupported at B < 16, where a 16-token prefill chunk
> and a B-token decode step have different byte/compute profiles. Replaced by the
> per-batch-size accounting in §8.2 plus a falsifiable prediction tested by probe P8 (§14).
> **Coordinator outcome: the workload boundary is UNCHANGED — `I ≤ O/4` at mbt=16, binding at
> B=16 — so the provisionally pinned (input 256 / output 1024) remains admissible; note it sits
> exactly ON the boundary.** Its AC-5 margin then rests on AC-4's strict decode win plus vLLM's
> nonzero prefill time (tolerance quantified in §8.2). Re-coordinate the pin only if P8
> falsifies the iteration-cost prediction or the B=16 decode win comes in under ~25 %.

### 8.1 Mechanism: chunked in-kernel, one task family

Prefill runs through the **same static task graph and the same GDN/attention tasks as decode**,
chunked by `prepare_next_batch`'s existing admission logic (`num_new_tokens =
min(prompt_len − step, mbt − used)` [MG §3.1]). Tasks read their chunk length from
`qo_indptr` at runtime — the exact codegen pattern MLA prefill ships today
(`Q_LEN = qo_indptr[bi+1] − qo_indptr[bi]`, per-request state slicing [MG §3.2 Option 1]).
Outside-kernel prefill (host runs prompt through torch, seeds the state) is **not expressible
under MODE_OFFLINE** — `init_kernel` zeroes `step[]` unconditionally and admission always
starts at token 0 [MG §3.2 Option 4] — and is not pursued.

Per chunk of Q_LEN tokens: conv = token-parallel FIR (§3.2); recurrence = sequential loop over
Q_LEN inside the task, S resident in smem ⇒ **state traffic amortizes to one read+write per
chunk** instead of per token; attention = Q-loop in ≤4-query passes (§4.3). Option 2
(dual-dispatched separate prefill kernel, the MLA precedent) stays in reserve if the unified
kernel's tiling turns out badly for Q_LEN ≫ 1 [MG §3.2].

### 8.2 Per-batch-size cost model and the CORRECTED CONSTRAINT

AC-5: `N_pf·t_pf + O·t_dec(B) ≤ 1.25·(P_v(B) + O·t_v(B))` at every B, same workload [GOAL].
Two distinct iteration types must be priced separately (this is what the previous version got
wrong):

- **decode step at batch B** — `bytes_dec(B) = 3.88 (dense, a per-iteration constant) +
  0.126·min(256, 8B) (worst-case distinct experts) + 0.129·B (GDN state)` GB — the §9.2 table;
- **prefill iteration** (chunk of C ≤ mbt = 16 prompt tokens across ≥ 1 requests) —
  `bytes_pf ≤ 3.88 + 0.126·min(256, 8C) + 0.129·(requests advancing) ≤ bytes_dec(16) =
  22.05 GB`. Non-byte adders at the pinned I=256: attention Q-loop KV re-streams ≤ 4× ≈ 20 MB
  at the deepest chunk (noise); GDN sequential chunk compute ≲ 0.5 ms worst case across 30
  layers (§3.2 sizing) — real, and folded into probe P8's threshold below.

Iterations to serve (I, O) at batch B: `N_pf = ⌈B·I/16⌉` prefill + `O` decode [MG §3.1];
request stagger (mixed prefill+decode iterations) only lowers the total — dense streams once
per iteration either way.

**Sufficient condition.** Dropping `P_v ≥ 0` and using AC-4's strict `t_dec(B) < t_v(B)`,
AC-5 follows from

```
N_pf · t_pf  ≤  0.25 · O · t_dec(B)                      (*)
```

Modeling `t_pf / t_dec(B)` by the byte ratio (equal achieved bandwidth on both iteration
types — conservative for prefill, whose 16-token GEMM tiles run no worse than 1-token decode):

| B | bytes_dec(B) | t_pf/t_dec (byte ratio) | N_pf | (*) as an I/O bound |
|---|---|---|---|---|
| 1 | 5.02 GB | 4.39 | I/16 | I ≤ 0.91·O |
| 2 | 6.15 | 3.59 | I/8 | I ≤ 0.56·O |
| 4 | 8.43 | 2.62 | I/4 | I ≤ 0.38·O |
| 8 | 12.96 | 1.70 | I/2 | I ≤ 0.29·O |
| **16** | **22.05** | **1.00** | **I** | **I ≤ 0.25·O — binding** |

In the opposite (fixed-overhead-dominated) limit `t_pf ≈ t_dec(B)` for every B and (*) reduces
to `B·I/16 ≤ O/4` — binding again only at B=16. The general claim covering both limits:
per-token iteration cost is monotone non-increasing in tokens/iteration (dense bytes and
per-iteration overhead amortize; worst-case expert and state bytes per token never grow), so
`t_pf(16)/16 ≤ t_dec(B)/B` and the B=16 bound dominates.

**CORRECTED CONSTRAINT: `I ≤ O/4` at mbt=16, binding at B=16 — the same boundary as the
invalid derivation, now conditional on one explicit, unmeasured assumption:
`t_pf(16-token prefill iteration) ≤ t_dec(16-request decode step)`.** This cannot be proven
pre-measurement; it is a falsifiable prediction — justified by the byte-subset + amortization
+ equal-overhead arguments above — and probe P8 (§14) tests it on the shipped Qwen3-8B
MODE_OFFLINE path early in M2 (re-run on the Qwen3.5 graph once it stands). Prediction:
prefill-iteration wall time within **1.5×** of the same-config decode iteration (the headroom
covers the GDN-chunk compute adder Qwen3-8B cannot exercise).

**The pinned workload (I=256, O=1024)** [coordinator, with I6]: I/O = 0.25 ⇒ meets (*) with
equality at B=16 (N_pf = 256 vs O = 1024) and strict slack at B ≤ 8 (table) — **the corrected
model does not move the boundary; the pin stands, at the boundary.** Equality at (*) still
yields strict AC-5 margin: `MPK_e2e = 1.25·O·t_dec(16) < 1.25·O·t_v(16) ≤ 1.25·vLLM_e2e` by
AC-4. Failure tolerance at this pin: AC-5 survives `t_pf ≤ k·t_dec(16)` for
`k ≤ 5·(t_v/t_dec) − 4` (before crediting vLLM's own prefill time) — a 25 % decode win
tolerates k ≤ 2.25, a 2× win tolerates k ≤ 6. The pin is threatened only if BOTH the B=16
decode win is thin (< ~25 %) AND P8 lands near its 1.5–2× falsification band; either signal
re-opens the workload choice — larger O, or an mbt=64 build (attention Q-loop scales to 16
passes; GDN chunk loop and GEMM row masking scale trivially), which cuts N_pf 4× and relaxes
the bound to I ≤ O at B=16, at the cost of decode dead-row overhead.

If a future workload pushes I ≫ 256, the chunk-parallel matmul formulation of the delta rule
is the M3+ lever — not v1 scope.

---

## 9. Batch 1..16 plan, per-step byte budget, AC-4 feasibility

### 9.1 Build and schedule

One compiled graph per benchmarked batch size (`mbr = B`, `mbt = 16` uniformly — mbt=16 keeps
16-token prefill chunks even at B=1). Per-B builds are the analog of vLLM's per-size CUDA-graph
capture [SG §1 row 7] and eliminate dead-slot task overhead at small B; MPK compiles per config
regardless. Task counts **per layer** per decode step at B=16: full-attn attention 16
(1/request), GDN conv 16, GDN recurrence 512 (32 v-heads × 16), MoE grouped-GEMM tasks
expert-major over ≤128 distinct experts/layer — the schedule shape [VG §5.4] recommends (visit each resident expert once,
gather its tokens), which the existing mask/indices-driven grouped GEMM already implements
[MG Gap 7].

### 9.2 Per-step HBM byte budget (compulsory traffic, worst-case distinct experts)

Dense (non-expert) weights: **2.47 GB/step fp8-target / 3.88 GB v1-bf16** (§6.2). Experts fp8:
1.01→16.11 GB for B=1→16. GDN state: B × 128.8 MB (fp32 S read+write + conv) [VG §5.4, §4.7].
KV read grows with position: ≈ 26 MB/request/step at the pinned workload's final positions
(~1300) → ≤ 0.42 GB/step at B=16 (< 2 %, not tabulated) [VG §2.2.5].

| B | experts | GDN state | **v1 total** | M3 total (fp8 dense) | v1 floor @ 8 TB/s | v1 floor agg tok/s |
|---|---|---|---|---|---|---|
| 1 | 1.01 | 0.13 | **5.02 GB** | 3.61 | 0.63 ms | 1 590 |
| 2 | 2.01 | 0.26 | **6.15** | 4.74 | 0.77 | 2 600 |
| 4 | 4.03 | 0.52 | **8.43** | 7.02 | 1.05 | 3 800 |
| 8 | 8.05 | 1.03 | **12.96** | 11.55 | 1.62 | 4 940 |
| 16 | 16.11 | 2.06 | **22.05** | 20.64 | 2.76 | 5 800 |

(8 TB/s = nominal B200 HBM3e; floors scale linearly with the true sustained figure.) Resident:
33.26 GiB fp8-target weights [VG §5.4] + 1.41 GB v1 bf16-dense delta + 0.96 GiB GDN pools +
0.47 GiB KV ≈ **36 GiB** — no memory pressure on a B200.

### 9.3 Feasibility vs the vLLM datapoint — stated honestly

The only measured baseline number is the **bs=1 smoke**: 29.88 tok/s = 33.5 ms/step, warm, CUDA
graphs captured, default config [REF]. That is ~1.3 % of the B=1 bandwidth roofline — vLLM's
small-batch step is dominated by fixed overhead and small-kernel inefficiency across ~600 op
instances/step, not by compulsory bytes. This is precisely the regime the megakernel thesis
attacks: MPK needs only ≥ 2 % sustained bandwidth utilization at B=1 to beat 30 tok/s **with
bf16 dense**; at a conservative 10–25 % it lands at 160–400 tok/s. The smoke number is not the
AC-4 protocol (single request, no fixed workload) and the M4 baseline will be tuned
(`--language-model-only` etc. — M1-I6 owns that ruling), but the headroom multiple (≥ 50×)
absorbs any plausible tuning delta at B ∈ {1, 2, 4}.

**B ∈ {8, 16} is the real fight.** There both engines converge toward the same compulsory
~13–22 GB/step — vLLM's per-step overhead amortizes over batched work and its floor at B=16 is
2.6 ms. MPK does not get to win on fewer bytes (the GDN kernel on both sides already does one
S read + one write per step [VG §2.1.4, §4.7]); it must win on *achieved fraction of the same
roofline* plus removed inter-kernel gaps. Known MPK-side hazards at B=16 and their plan:

1. **`prepare_next_batch` serial knee** — +69 % step latency at batch 8 on Qwen3-8B
   [MG §8 risk 4]. Mitigation: GDN lifecycle stays out of it (§3.3); M3 profiles the serial
   section with the MPK profiler CSV (after fixing the stale `profiler_persistent.py` map
   [MG §4]) and parallelizes or trims it.
2. **Small-shape MoE efficiency** — inter-512 tiles were never tuned [MG Gap 7]; the vLLM
   Triton config table is the starting prior (§5).
3. **Per-token rmsnorm task storm** — 16 tasks/norm/layer at B=16 with no batching
   [MG Gap 9]; an M3 batched-rmsnorm task is additive.

Assertions to record on the vLLM side of every benchmark (fairness list): FlashInfer TRT-LLM
MoE backend confirmed from the log line, CUTLASS dense kernel + the DeepGEMM auto-disable
warning present, all env/CLI at defaults, bf16 KV, `mamba_ssm_cache_dtype=float32`
[VG §3.7.5].

---

## 10. TaskType id budget

14 free ids below `TASK_SM100_TASK_END` — not a constraint [MG §4]. Assignments:

| id | task | TMA note |
|---|---|---|
| 234 | `gdn_conv1d_sm100` | in-window |
| 237 | `gdn_recurrent_sm100` | in-window; M3 may add real TMA loads for the 64 KiB S tile |
| 238 | `sigmoid_gate_mul_add_sm100` | in-window |
| 239 | *reserved*: router variant (if P5 fails) | |
| 240 | *reserved*: standalone attn gate (fused-route fallback, §4.2) / GDN prefill split (if Option 1 splits) | |

**Refinement to [MG §4], verified this session:** ids in the open interval (231, 256) get
`create_tma_desc_by_task()` called unconditionally at init (`runtime_header.h:128,140`;
codegen at `runtime.cc:1170-1172`), and that function's switch ends in `default:
assert(false);` (`tma.cuh:1465, 1629-1631`). **Every new in-window TaskType must add a case to
that switch** — a bare `break;` for non-TMA kernels — or init aborts. Only one free id (279)
sits outside the window, so in-window + explicit case is the standard recipe. This makes the
`add-mpk-task` recipe **9 files** for in-window ids: the skill's 7 + `runtime.cc`'s
`task_type_to_name` [MG §4] + `tma.cuh`'s switch (plus `profiler_persistent.py` for usable
profiles). Feeds the dev-skill maintenance issue (M2-I10).

---

## 11. Explicitly out of scope (v1)

- **Vision tower**: never loaded (HF text path ignores it; AC-2). The 112 `model.visual.*`
  skip-list entries and all vision tensors are dropped at load [VG §3.1, §5.4].
- **MTP / speculative decoding**: `mtp.*` weights skipped (as vLLM does,
  `skip_prefixes=["mtp."]` [VG §5.4]); no spec-decode on either side of any benchmark
  [GOAL AC-2]. The GDN conv-state widening for `num_spec > 0` [VG §4.1] is not provisioned.
- Multi-GPU, prefix caching, fp8 KV, serving features beyond the benchmark protocol
  [GOAL non-goals].

---

## 12. AC-3 harness specification (build in M2 from day one)

The gate is `mpk_token_ids == reference_token_ids`, no tolerance anywhere; the tie-flip clause
is a documented human-adjudicated exception to an already-failed gate, never a soft compare
[GOAL AC-3; MG §6.5]. The M2 harness (fixed path under `demo/qwen3_5/accept/`) must:

1. Reuse `reference_outputs.json`'s `input_ids` verbatim (no re-tokenization) and its EOS
   semantics (`eos_token_id ∈ {248046, 248044}`, compare up to `num_generated`) [REF].
2. Emit, for **every** position on **every** run (passing included): reference top-2 ids +
   margin `logit[top1] − logit[top2]`, MPK's argmax, MPK's logits for the reference top-2 ids,
   the equality boolean, and the **first divergent position per prompt** (later positions of a
   diverged prompt are a different conditioning sequence, not independent evidence) [MG §6.5].
3. Run MPK at bs=1 per prompt (matching the reference's conditions) with a per-step top-16
   logit dump (the logits tensor is already a graph output; ~8 MB/step host copy in harness
   mode is acceptable).
4. Archive the margin distribution with the run report — a tie-flip claim is only credible
   against margins measured *while passing* [MG §6.5].

The per-op bring-up ladder below the end-to-end gate follows the DFlash methodology: spec →
`ref_dump.py` oracle (probe P6) → standalone kernel vs oracle → test-mode in the megakernel →
end-to-end [MG §6.5]; per-folder `pytorch_reference.py` per the test-mode skill contract
(convention to be established — no such file exists in-tree yet [MG §6.4]).

---

## 13. Deviations and overrules (explicit)

1. **GDN lifecycle not in `prepare_next_batch`** (issue contract wording): implemented as
   kernel-side `step == 0` predication instead (§3.3), per [MG §3.2]'s own recommendation —
   the serial scheduler section is the known batch-8 knee and CI-protected shared runtime.
   Same semantics (reset on admission / persist / implicit release), different owner.
2. **[MG §4] refined**: "the only mechanical range check in the SM100 window is the TMA one"
   understates it — the TMA hook *asserts* on unknown in-window ids (`tma.cuh:1629`), so
   in-window ids are usable but each requires a switch case (§10). Conclusion unchanged
   (14 free ids, no budget constraint), recipe grows by one file.
3. **"6 registration sites" for the MAX_TOKENS decouple** [CONSTR §2a]: all six verified to be
   paged-attention-family registrations; only the sm100 site(s) are touched, params-gated,
   default byte-identical (§4.3). Narrower than the constraint's framing; the Qwen3-8B CI run
   remains the acceptance check.
4. **v1 bf16 dense projections** (§6.2) go beyond the checkpoint's `modules_to_not_convert`
   list for M2 only. The M4 configuration that AC-4/AC-6 measure is fp8-dense (fp32-scale).
   Staging decision, surfaced for user batch review per the autonomy grant — not a goal re-pin.
5. **GDN in_proj not fused at load** (§2.0): vLLM fuses `qkv`+`z` and `b`+`a` into single
   GEMMs [VG §5.1]; MPK keeps `qkv` and `z` as separate GEMM tasks (and concatenates only
   `b`+`a`), because separate outputs feed the conv and recurrence tasks without strided views.
   Same math, one extra GEMM task per GDN layer, zero extra bytes.
6. **Conv output not in-place** (§3.2): vLLM's conv kernel overwrites its input
   [VG §2.1.4]; MPK writes a separate buffer for clean event semantics. +128 KB/step.
7. The scouting-report claims already corrected by the input docs (2-free-ids, 3-D expert
   tensors, bf16 state sizing, etc.) are inherited as corrected [VG §7; MG §9] — designs above
   use only the corrected values.

---

## 14. M2 seeded issue list + probes  **(mandatory section)**

Probes are the first acceptance step of their owning issue. Each probe is an executable
command with its expected *discriminating* outcome — the outcome decides a design branch, not
just "works". Scripts referenced under `~/mpk-qwen35/probes/` are ≤ ~100-line artifacts the
owning issue writes first; all GPU probes run on `catalyst-B200` under the GPU-etiquette rules
[CONSTR §1].

### Issues

| id | scope (one line) | SOP (dev skill) | depends on |
|---|---|---|---|
| **M2-I1** | AC-3 harness per §12: gate + per-position margin instrumentation + MPK top-16 logit dump + argmax-tie unit check | `test-mode` (reference discipline; §12) | — |
| **M2-I2** | FP8 execution validation: probes P1, P2, P7, P10 (M3 fp8-dense numerics + perf bar); final dense GO/NO-GO per §6.2 decision tree; wire fp32-scale `quantize_fp8` for MoE activations | `test-mode` + `add-mpk-task` Step 9 (benches) | — |
| **M2-I3** | HF oracle `ref_dump.py`: per-op tensor dumps for one GDN layer, one full-attn layer, one MoE block (probe P6) | `test-mode` (`pytorch_reference.py` convention) | — |
| **M2-I4** | `gdn_conv1d_sm100` (id 234): kernel + conv-state pool + `step==0` init + unit/test-mode tests vs oracle | `add-mpk-task` (9-file recipe, §10) | M2-I3 |
| **M2-I5** | `gdn_recurrent_sm100` (id 237): fused delta rule + gated norm per §3.2; per-(head,slot) grid; chunked Q_LEN loop; unit/test-mode tests vs oracle | `add-mpk-task` (9-file) | M2-I3 (∥ M2-I4) |
| **M2-I6** | Attention adaptation: cherry-pick `5715c6f`; QKVG slice + σ-gate epilogue (params-gated); `max_tokens_per_pass` + Q-loop at `task_register.cc:2052`; RoPE permutation loader; probes P3, P4; local Qwen3-8B CI run green | `add-mpk-task` steps 2/5/7 + `mpk-internals`; CI-byte-identical discipline [MG §6.2] | — (P3/P4 first) |
| **M2-I7** | MoE block at our shapes: probe P5; w13/w2 wiring at `[256,1024,2048]`/`[256,2048,512]`; shared expert + `sigmoid_gate_mul_add_sm100` (id 238); test-mode pipeline test | `add-mpk-task` (one new task) + `test-mode` | M2-I2 (P2) |
| **M2-I8** | Qwen3.5 registry builder + weight loader: §2.0 transforms, config plumbing, `mbr ≤ mbt` + page-capacity asserts [MG §8 risk 5], vocab §7 | `add-mpk-model` (registry path) + `mpk-internals` | I4–I7 interfaces |
| **M2-I9** | End-to-end bring-up: full 40-layer graph, per-layer test-mode vs oracle, AC-3 run via M2-I1 to 640/640 (or adjudicated tie-flips) | `test-mode` + systematic-debugging | all above |
| **M2-I10** | Dev-skill maintenance commits: `add-mpk-task` 7→9-file recipe (+`tma.cuh` case, `runtime.cc`), `pytorch_reference.py` convention, `add-mpk-model` registry-path staleness [MG §9] | constraint.md §2b (skill-maintenance rule) | — |
| **M2-I11** | Early runtime measurements on the shipped Qwen3-8B path (no Qwen3.5 code needed): prefill-iteration cost (P8, tests §8.2's load-bearing assumption) + scheduler-knee attribution (P9) | MPK profiler + `test-mode` | — (P9's labeled trace wants M2-I10's profiler-map fix; raw task-type ids work meanwhile) |

### Probes

**P1 — dense-bf16 token equivalence (gates the §6.2 v1 ruling). Owner M2-I2.**
```bash
ssh catalyst-B200 'source ~/mpk-qwen35/venv-vllm/bin/activate && \
  TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 python ~/mpk-qwen35/probes/p1_dense_bf16.py \
    --model Qwen/Qwen3.5-35B-A3B-FP8 --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
    --reference-json ~/mpk-qwen35/reference_run/reference_outputs.json'
# script: load HF model; replace every FP8 linear EXCEPT the routed experts (mlp.experts.*)
# — i.e. all dense-path modules incl. mlp.shared_expert.* — with an nn.Linear carrying the
# exact block-dequantized bf16 weight; greedy 64 tok x 10 prompts from reference input_ids;
# diff token ids and report per-position reference margins at any mismatch.
```
Expected: **640/640 match ⇒ bf16-dense v1 GO.** Any mismatch at a reference margin above the
run's noise floor ⇒ NO-GO ⇒ M2 pulls the fp32-scale fp8 dense GEMM forward (§6.2), before
integration starts.

**P2 — MoE FP8 kernel at our shapes + 4-k-tile pipeline + internal-UE8M0 numerics. Owner M2-I2.**
```bash
ssh catalyst-B200 'cd ~/mpk-qwen35/mirage && \
  python tests/runtime_python/blackwell/sm100_fp8_moe/test_fp8_moe_gemm.py \
    --num-experts 256 --n 1024 --k 2048 && \
  python tests/runtime_python/blackwell/sm100_fp8_moe/test_fp8_moe_gemm.py \
    --w2 --num-experts 256 --n 2048 --k 512'
# in-tree test is hardcoded to DSV3 shapes [MG §2.3] — the issue parameterizes it and adds a
# comparison vs an fp32 block-dequant torch reference using REAL layer-0 checkpoint scales.
```
Expected: completes with **no hang at `fp8_k_tile_count = 4`** and error consistent with
activation-quant noise (no systematic per-row bias). Hang ⇒ `num_ab_stages` becomes a
DSV3-affecting shared change — measure before touching [MG Gap 7]. Systematic bias ≫ act-quant
noise ⇒ the kernel's internal UE8M0 scale handling is lossy for our checkpoint ⇒ escalate: the
grouped GEMM needs fp32-scale application (same E8M0 class as the dense rejection, §5).

**P3 — attention smem instantiation sweep post-pick (validates §4.3's MAX_TOKENS=4). Owner M2-I6.**
```bash
ssh catalyst-B200 'cd ~/mpk-qwen35/mirage && git worktree add /tmp/p3-pick qwen3-5_support && \
  cd /tmp/p3-pick && git cherry-pick 5715c6f2a6cce5d0d18da4e6776332b6ad04d7e4 && \
  for MT in 1 2 4 8; do \
    PATH=/usr/local/cuda-12.8/bin:$PATH nvcc -arch=sm_100a -DP3_MAX_TOKENS=$MT \
      -I include -c ~/mpk-qwen35/probes/p3_attn_smem.cu -o /dev/null \
      && echo "MT=$MT COMPILES" || echo "MT=$MT STATIC_ASSERT"; done'
# p3_attn_smem.cu: bare TU instantiating the attention task impl at
# NUM_QO_PER_KV=8, HEAD_DIM=256, MAX_TOKENS=P3_MAX_TOKENS — the 5715c6f validation method.
```
Expected: **COMPILES at 1/2/4, STATIC_ASSERT at 8** ⇒ confirms the 196-KiB-at-4 model [MG §5]
and the Q-loop pass size of 4. COMPILES at 8 ⇒ the paper model was pessimistic; Q-loop pass
size becomes 8 (fewer passes, same design).

**P4 — partial-RoPE permutation exactness (gates §4.4's zero-kernel route). Owner M2-I6.**
```bash
python ~/mpk-qwen35/probes/p4_rope_perm.py   # torch-only, CPU ok, no checkpoint needed
# path A: Gemma-norm -> HF partial NeoX RoPE (rotary_dim 64, theta 1e7, pairs (i, i+32));
# path B: permuted norm-weights/q/k columns -> full-256 NeoX (pairs (i, i+128)) with
#         cos=1/sin=0 padding; compare q,k element-wise and q·k logits, fp32 and bf16.
```
Expected: **max abs diff = 0.0 in fp32** (permutation is algebraically exact), ≤ 1 ulp in
bf16 ⇒ permutation route GO. Any structural mismatch ⇒ fall back to the `ROTARY_DIM` template
parameter kernel route (params-gated, CI-protected edit) [MG Gap 4].

**P5 — router semantic equality with RenormalizeNaive. Owner M2-I7.**
```bash
ssh catalyst-B200 'cd ~/mpk-qwen35/mirage && \
  python ~/mpk-qwen35/probes/p5_router.py --num-experts 256 --topk 8'
# drives the existing topk_softmax_sm100 kernel (via its test wrapper) on random rows PLUS
# crafted exact-tie rows; reference: fp32 full-256 softmax -> top-8 with lower-index
# tie-break -> renormalize [VG §2.3.2].
```
Expected: **exact id sets incl. tie rows, weights within 1e-6 rel** ⇒ reuse as-is. Divergent
tie-breaking or a topk-then-softmax ordering ⇒ new router variant task (reserved id 239);
weights-only drift ⇒ check the kernel's softmax precision before writing any new kernel.

**P6 — HF GDN/attention/MoE oracle (`ref_dump.py`). Owner M2-I3.**
```bash
ssh catalyst-B200 'source ~/mpk-qwen35/venv-vllm/bin/activate && \
  TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 python ~/mpk-qwen35/probes/p6_ref_dump.py \
    --layers 0,3 --prompt-id p01-history --tokens 8 --out ~/mpk-qwen35/oracle/'
# forward hooks on the HF modeling code: dump post-conv qkv, g, beta, S, o, gated out (GDN
# layer 0); post-norm/rope q,k, gate, attn out (layer 3); router probs/ids/weights, per-expert
# partials, shared-expert path (both layers' MoE). Records op order and every dtype.
```
Expected outcome (this probe *produces* the ground truth rather than branching): the numeric
oracle files every M2 kernel test compares against, plus the answers to the open bit-parity
questions — HF's actual `β = σ(b)` dtype path, decay/softplus precision, conv accumulator
dtype, and the MoE combine order — where vLLM's own kernels disagree with each other
[VG §2.1.4, §6 g.10] and the HF answer is binding (§1).

**P7 — UE8M0 dense requant divergence, quantified (supporting evidence, optional). Owner M2-I2.**
```bash
ssh catalyst-B200 'source ~/mpk-qwen35/venv-mpk/bin/activate && \
  python ~/mpk-qwen35/probes/p7_ue8m0_delta.py --tensor "layers.0.self_attn.q_proj"'
# replicates _requantize_fp8_for_ue8m0 [MG §2.2] in torch on real checkpoint tensors;
# reports element-wise delta stats between block-dequantized weights before vs after.
```
Expected: **nonzero deltas in both directions** (checkpoint scales are not powers of two) —
quantifies the §6.2 rejection with data on our own weights. Exactly-zero deltas would
re-admit the existing dense path (not expected; would mean the checkpoint's scales were all
powers of two, contradicting [MG §2.2.1]).

**P8 — prefill-iteration cost vs decode-iteration cost (tests §8.2's one load-bearing
assumption; runnable today, no Qwen3.5 code). Owner M2-I11.**
```bash
ssh catalyst-B200 'cd ~/mpk-qwen35/mirage && \
  python ~/mpk-qwen35/probes/p8_prefill_iter_cost.py --model Qwen/Qwen3-8B --mbt 8 \
    --input-len 32 --input-len 512 --output-len 128 --ignore-eos'
# ≤60-line adaptation of tests/ci-tests/run_batch_perf.py (already measures ms/token with
# --ignore-eos on the Qwen3-8B MODE_OFFLINE path [MG §6.1]): pad the prompt to each
# --input-len; [T(512,128) − T(32,128)] / ((512−32)/mbt) = wall time per extra prefill
# iteration at chunk = mbt; decode ms/token from the same run is t_dec at that config.
```
Expected: prefill-iteration time within **1.5×** of the same-config decode iteration ⇒ the
§8.2 assumption `t_pf ≤ t_dec` holds on the real runtime (this validates the *iteration
mechanics* — chunked prefill through the static graph has no hidden per-iteration penalty;
the Qwen3.5-specific bytes are §8.2's model, re-tested by re-running P8 on the Qwen3.5 graph
once M2-I9 stands). **> 2×** ⇒ §8.2's model is falsified for this runtime: escalate — the
Option-2 dual-dispatch prefill kernel and/or an mbt=64 build enter M2 scope, and the
(256, 1024) workload pin is re-coordinated (§8.2).

**P9 — batch-8/16 scheduler-knee attribution (runnable today). Owner M2-I11.**
```bash
ssh catalyst-B200 'cd ~/mpk-qwen35/mirage && for r in 1 2 4 8; do \
  python tests/ci-tests/run_batch_perf.py --max-num-batched-requests $r --ignore-eos; done'
# step 1 reproduces the recorded knee (4.40/4.41/4.44/7.49 ms/token at r=1/2/4/8, commit
# 92603ca [MG §8 risk 4]); step 2 re-runs r ∈ {4, 8} with params["profiler_tensor"] set and
# reads, from the CSV trace, the per-iteration gap between the last task of iteration N and
# the first task of N+1 (= prepare_next_batch + dispatch wall time) vs summed task execution
# time. (Read raw task-type ids until M2-I10 fixes the stale profiler_persistent.py map
# [MG §4].)
```
Expected: knee reproduces AND the inter-iteration gap grows by ≈ the ms/token jump (~+3.0 ms)
from r=4 → r=8 while summed task time stays ≈ flat ⇒ **knee = serial `prepare_next_batch`** —
confirms §3.3's keep-lifecycle-out ruling and makes scheduler-section work the top M3 item.
Gap flat + task time grows ⇒ knee is task-side (occupancy/attention tiling) ⇒ M3 effort
redirects to kernels; §3.3 still stands on blast-radius grounds alone.

**P10 — fp32-scale dense fp8 GEMM: numerics + perf bar that greenlights the M3 restoration.
Owner M2-I2.**
```bash
ssh catalyst-B200 'source ~/mpk-qwen35/venv-vllm/bin/activate && \
  python ~/mpk-qwen35/probes/p10_fp8_dense_bar.py \
    --shapes 12288x2048,9216x2048,2048x4096 --batch 1,4,16'
# vLLM 0.25.1's own CUTLASS block-scaled kernel (ops.cutlass_scaled_mm — the exact kernel the
# AC-4 baseline runs [VG §3.5]) on REAL layer-0 checkpoint weights + scales:
# (i) numerics vs the HF kernels-hub Triton finegrained-fp8 linear on identical inputs;
# (ii) latency vs a bf16 torch.matmul at the same shapes (proxy for MPK's bf16 linear task).
```
Expected: **(i)** CUTLASS-vs-HF-Triton max rel diff at fp32-accumulation-reorder scale
(~1e-3, no systematic bias) ⇒ two independent implementations of the fp32-scale semantics
agree on our weights — the M3 target semantics is well-defined and MPK's implementation has
two oracles; **(ii)** fp8 ≥ **1.5×** faster than bf16 at B ≤ 16 ⇒ the +1.41 GB/step reclaim
is real ⇒ **M3 GO** for adapting `linear_fp8_sm100.cuh` to promotion-style fp32-scale
accumulation. (i) fails ⇒ the §6.2 M3 plan needs a numerics redesign review before any kernel
work; (ii) < 1.2× ⇒ the restoration is perf-marginal — re-rank it among M3 levers (the goal's
fp8-execution framing still forces it eventually; its priority, not its existence, changes).

### Dependency order

`I1, I2, I3, I10, I11` start immediately in parallel → `I4, I5` (need the I3 oracle), `I6`
(starts with its own P3/P4), `I7` (needs I2's P2; starts with its own P5) → `I8` (interfaces)
→ `I9` (integration + the AC-3 gate). The critical path is `I3 → I5 → I8 → I9` — the
recurrence task is the project's critical path overall [MG §8 risk 1]; everything else is
parallel to it.
