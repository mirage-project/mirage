# SGLang v0.5.21 (ROCm) target-verify contract for our KDA and MLA kernels

Source: the `sglang` package of lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261009 (paths below relative to that `sglang/` dir).
v0.5.19 reference: the v0.5.19 `sglang` package (prefix `v519:`).
Our B300 side: `mirage_kda/mpk-kernels/mpk_kernels/integrations/sglang_k3_{kda,mla}.py`, `mpk_kernels/kimi_k3/{kda,mla}.py`,
`tests/{kda,mla}_reference.py` (prefix `B300:`).
Config: `--tp 8 --speculative-algorithm DSPARK --enable-linear-replayssm-spec --attention-backend triton --dtype bfloat16
--max-running-requests 8 --cuda-graph-max-bs 8`, `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1`, HIP gfx950. One request, 8 verify
tokens. **(inferred)** marks anything not read directly in code. Read on 2026-10-13, no GPU run behind any of this.

---------------------------------------------------------------------------------------------------------------------------

## A. KDA verify with `--enable-linear-replayssm-spec` on HIP

### A.1 Which kernel runs

* DSPARK forces `speculative_eagle_topk=1` (`srt/arg_groups/speculative_hook.py:691-705`), gamma default 7 -> 8 verify tokens
  (`speculative_hook.py:739-765`, `speculative/dspark_components/dspark_config.py:21`). `SGLANG_RAGGED_VERIFY_MODE` default
  `static` (`srt/environ.py:1420`) -> `spec_info.ragged_verify_layout is None` (dense layout).
* `--linear-attn-verify-backend` is pinned to `nv_cutedsl` for K3 whenever a spec algorithm is set, no platform check
  (`srt/arg_groups/kimi_k3_hook.py:28-38`, called from `srt/arg_groups/model_hook.py:241-252`).
* `KDAKernelDispatcher` maps `nv_cutedsl` to the **Triton** kernel object (`srt/layers/attention/linear/kda_backend.py:174-175`).
* `_forward_target_verify` (`kda_backend.py:1186`):
  1. CuTe DSPARK path `_can_run_dspark_cutedsl_mtp` returns False on HIP: `... or not is_cuda()` (`kda_backend.py:1702`);
     `is_cuda()` = `torch.version.cuda is not None` (`srt/utils/common.py:157-158`), False on ROCm.
  2. Fused chain-verify `fused_kda_conv_gating_verify` is off: `_fused_chain_verify_fn` is only set when
     `SGLANG_OPT_FUSED_KDA_VERIFY` and verify backend `triton` (`kda_backend.py:509-519`), and with ReplaySSM it also needs
     SM90/SM100 (`kda_backend.py:1493-1501`).
  3. So the unfused chain runs:
     * `causal_conv1d_update(...)` (`kda_backend.py:1369-1381`) -> Triton `_causal_conv1d_update_kernel`
       (`kernels/ops/mamba/causal_conv1d_triton.py:607`), no bias, `activation="silu"`, `num_accept_tokens=None`;
     * split q|k|v -> [1, 8, 12, 128] each (`kda_backend.py:1395-1398`);
     * `TritonKDAKernel.target_verify` (`kda_backend.py:1400`, `srt/layers/attention/linear/kernels/kda_triton.py:159-224`)
       -> `fused_sigmoid_gating_delta_rule_update(is_kda=True, disable_state_update=True, use_qk_l2norm_in_kernel=True,
       lower_bound=-5.0, cache_ring=True, replayssm_*=rings, intermediate_states_buffer=None)`
       (`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:531-750`).
  `match_cutedsl_decode` is not passed (default False, `fused_sigmoid_gating_recurrent.py:567`), so the plain fp32 formulas run.

### A.2 Inputs (per layer, TP8 rank: 12 heads x 128)

| tensor | shape / dtype | source |
|---|---|---|
| `mixed_qkv` (pre-conv q,k,v) | [8, 4608] bf16 (q 0:1536, k 1536:3072, v 3072:4608) | `fused_qkvg_proj(x)` split (`srt/models/kimi_k3.py:2038-2040`) |
| `a` = forget gate f_b | [1, 8, 12, 128] bf16 | `tiny_gemm(bfa[:, :128], _bfa_f_b_w)`, `kimi_k3.py:2041-2046, 2079-2081` |
| `b` = raw beta | [1, 8, 12] bf16 (no sigmoid yet) | `bfa[:, 128:140]`, `kimi_k3.py:2046, 2082` |
| `conv_states` = `cache.conv[0]` | per-layer [size+1, 3, 4608] bf16, step 0 = oldest | `kda_backend.py:1209`; alloc `srt/mem_cache/memory_pool.py:709-716` |
| `ssm_states` = `cache.temporal` | per-layer [size+1, 12, 128(v), 128(k)] fp32, k contiguous | `fused_sigmoid_gating_recurrent.py:264-273`; fp32 forced by `srt/arg_groups/attention_hook.py:436-446` |
| `conv_weights` | [4608, 4] fp32 (view of `qkv_conv1d.weight.squeeze(1)`), tap 3 multiplies the newest input | `kimi_k3.py:1755`; `causal_conv1d_triton.py:819-829, 935-955` |
| `A_log` [1,1,12,1] fp32, `dt_bias` [1536] fp32 | | `kimi_k3.py:1692-1733` |
| `cache_indices` = `forward_metadata.mamba_cache_indices` | int32 [bs] | see B.1 |
| `intermediate_state_indices` | int32 | `select_verify_intermediate_state_indices(self.verify_intermediate_state_indices, ...)` (`kda_backend.py:1245-1250`) -> returns the default table unless `SGLANG_ENABLE_PP_SPEC` (`srt/layers/attention/linear/utils.py:110-121`, default False `srt/environ.py:1409`); table = `arange(pool_size)` int32 (`utils.py:124-157`) -> row 0 for the one request |
| `query_start_loc` | int32 [bs+1] = [0, 8] | `hybrid_linear_attn_backend.py:657-663, 896-898` |

### A.3 What is written, exactly

**Conv kernel** (`causal_conv1d_triton.py:607-1010`, grid (bs, 4608/256)):
* Reads `conv[slot][0..2]` (col0..col2, `697-712`) before anything is written by the same program.
* **Writes `conv[slot]` in place** (`760-785`): state_len = 3 (`1118`), seqlen = 8, so the masked load from the old state is
  empty and `new_conv_state[j] = x[5 + j]` -> `conv[slot] = [x5, x6, x7]` (the last 3 **pre-conv** inputs, bf16).
* Per token t: window after token t into `intermediate_conv_window[0][row][t]` = [col0, col1, col2] = the 3 newest pre-conv
  inputs, oldest first (`975-991`). The backend passes `icw.transpose(-1,-2)` (`kda_backend.py:1378`), so the memory layout of
  the per-layer view is **[rows=spec+1=9][steps=8][3][4608] bf16**, dense (KDA disables window dedup:
  `memory_pool.py:127-142, 903-907`; `srt/configs/mamba_utils.py:305`). Row = `intermediate_state_indices[0]` = 0.
* Output: `acc = sum_j col_j * w_j + x_t * w3` in fp32 (`957-978`), `silu` (`997`), stored to `out = empty_like(x)`
  (`1100`) -> **bf16**. The recurrence therefore reads bf16-rounded conv outputs.

**Recurrence kernel** (`fused_sigmoid_gating_recurrent.py:136-528`, grid (128/32, N=1, 12), BV=32 (`66`)):
* Initial state from `temporal[cache_indices[0]]` (`259-273`); `temporal` **not written** (`DISABLE_STATE_UPDATE`, `516-528`);
  `intermediate_ssm` is None under ReplaySSM (`memory_pool.py:866-868`) so no per-step state is stored (`486-497` off).
* Per token t (step_idx = 0..7, `295-506`):
  * `x = a + dt_bias` (fp32), `g = -5 * sigmoid(exp(A_log) * x)` per K channel (`331-334`);
  * `beta = 1 / (1 + exp(-b))` (`361`);
  * ring write (`371-424`), slot = `h0_indices[0]` = `mamba_cache_indices[0]`, position = **step_idx (0..7), no cursor**,
    dropped if step_idx >= L:
    * `replayssm_rawv[slot][hv][t][:]` = v (conv output, bf16 -> fp32 -> bf16, exact copy), `375-382`;
    * `replayssm_rawk[slot][h][t][:]` = k **before** the L2 norm (bf16 conv output), `384-392`;
    * `replayssm_g[slot][hv][t][:]` = g fp32 natural-log decay (per-K vector for KDA), `399-408`;
    * `replayssm_beta[slot][hv][t]` = sigmoid(b) fp32, `417-424`.
  * `q, k *= 1/sqrt(sum x^2 + 1e-6)` (`427-439`), `q *= 128^-0.5` (`441, 618-619`), `h *= exp(g)[k]` (`455`),
    `v -= sum_k h k` (`460`), `v *= beta` (`463`), `h += k (x) v` (`479`), `o = sum_k h q` (`482`) stored **bf16**
    (`o = q.new_empty`, `623, 483`) as [1, 8, 12, 128].
* After the backend: `o_norm` = `FusedRMSNormGated(128, eps 1e-5, sigmoid)` on bf16 o and bf16 g (`kimi_k3.py:2114-2116`;
  `_kda_fused_decode_ready` is False on HIP because `_prepare_fused_decode` returns early on `_is_hip` unless
  `SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter` (`kimi_k3.py:1896-1944`), so no fused o_norm), then `o_proj` (`kimi_k3.py:2117-2123`).

Ring layout (`memory_pool.py:747-835`): per layer `rawv` [size+1, 12, L, 128] bf16 (conv dtype), `rawk` [size+1, 12(k heads),
L, 128] bf16, `g` [size+1, 12, L, 128] fp32, `beta` [size+1, 12, L] fp32; `num_slots = size + 1` for KDA (`755-759`);
`record_len = L` for KDA (`771-778`); L = `--linear-replayssm-cache-len` default 16 (`srt/arg_groups/fields/exec_.py:481-484`),
power of 2 and >= 2 x draft tokens (`memory_pool.py:809-819`). The kernel uses `stride(0)` as slot pitch and
`MAX_CACHE_LEN = shape[-2]` (`fused_sigmoid_gating_recurrent.py:647-665`), i.e. contiguous [slot][head][pos][dim].

### A.4 Post-verify commit (DSPARK)

* `dspark_worker_v2.py:1158` -> `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`
  (`srt/layers/attention/hybrid_linear_attn_backend.py:1505`): slots = `linear_attn_backend.forward_metadata.mamba_cache_indices[:n]`
  (`1525-1530`), `replayssm_is_kda` -> `commit_kda_replayssm_after_verify(accept_lens = last_correct_step + 1, ...)`
  (`1548-1561`). (The generic `spec_utils.commit_mamba_states_after_verify` KDA branch, `srt/speculative/spec_utils.py:1092-1150`,
  does the same.)
* `kernels/ops/attention/fla/kda_replayssm_spec_decode.py:336-395`:
  1. fold, all layers in one launch (`257-333`, kernel `40-176`): `h = temporal[slot]`; for t in 0..accept_len-1 read ring
     position t (rawk -> fp32, `k / sqrt(sum k^2 + 1e-6)`, `h *= exp(g)`, delta rule with beta) -> `temporal[slot] = h`.
     Reads ring positions **0..accept_len-1** at the slot.
  2. conv rollback: `fused_conv_window_scatter_with_mask(conv, icw, slots, last_correct_step, src_indices_raw=None)`
     (`378-395`) -> `conv[slot] = icw[row][accept_len-1]` with `row = arange(bs)[i]` = 0 when `src_indices_raw` is None
     (`kernels/ops/mamba/mamba_state_scatter_triton.py:223-230`). This overwrites the in-place `conv[slot]` write of the verify
     conv kernel, so that write is not visible after the commit (inferred: no reader of `conv[slot]` between verify and
     commit; the DSPARK draft is dense MQA, `kimi_k3_hook.py:40-41`).

### A.5 Item-by-item vs our B300 integration (v0.5.19 contract)

| item | B300 (`sglang_k3_kda.py` `_pools()` + kernel) | v0.5.21 HIP | same? |
|---|---|---|---|
| slot | `mamba_cache_indices[:1]` int32 | same tensor, int32 (`hybrid_linear_attn_backend.py:620-626`) | yes |
| window row | `verify_intermediate_state_indices[:1]` | `select_verify_intermediate_state_indices(...)` = that table unless `SGLANG_ENABLE_PP_SPEC=1` | yes (env default) |
| `ring_k/v` | `[slot][h][0..7][128]` bf16 = bf16(silu(conv)) | same positions, same value (conv output already bf16) | yes |
| `ring_g` | fp32 `-5 sigmoid(exp(A_log)(f_b + dt_bias))` | same formula, fp32 | yes |
| `ring_beta` | fp32 sigmoid(b) | same | yes |
| `icw[row][t]` | [3][4608] bf16, 3 newest pre-conv inputs after t | same | yes |
| `temporal[slot]` | read, not written | read, not written | yes |
| `conv[slot]` | read, not written | **read, then overwritten in place with x5..x7** | **no** (state after commit identical, inferred) |
| conv output into recurrence | our kernel rounds to bf16 (`B300: tests/kda_reference.py:1-3`) | bf16 (`causal_conv1d_triton.py:1100, 1008`) | now same (v0.5.19 CuTe used fp32, `B300: kda_reference.py:31`) |
| core output before o_norm | v0.5.19 CuTe folds o_norm into the recurrence on fp32 o (`v519: kda_backend.py:1344-1385`, `v519: models/kimi_k3.py:1877-1894`) | o stored **bf16** then separate `rms_norm_gated` | **check our kernel** |

---------------------------------------------------------------------------------------------------------------------------

## B. Per-step metadata under HIP-graph replay

Graph runner: `srt/model_executor/runner/decode_cuda_graph_runner.py`. Capture builds the ForwardBatch from registry slots
`_slot(name) = registry.get_slot(name).slice_for(bs, num_tokens)` (`922-934, 1011-1022`): views at offset 0 of static buffers.
Replay: `load_batch` -> `buffer_registry.fill_from(forward_batch, ...)` copies the live batch into those buffers
(`1430-1437`; `srt/model_executor/cuda_graph_buffer_registry.py:388-460`), then `attn_backend.init_forward_metadata_out_graph(fb_view)`
(or the captured metadata glue graph) refreshes the backends' static buffers (`decode_cuda_graph_runner.py:1489-1517`). For
DSPARK, `load_batch` runs inside `prepare_for_verify` before the replay (`srt/speculative/dflash_info.py:111-121`). Static buffer
dtypes (`srt/model_executor/runner_utils/buffers.py:158-163`; `decode_cuda_graph_runner.py:592-593`).

### B.1 KDA

| tensor | dtype / shape | capture-stable? | refreshed where |
|---|---|---|---|
| `be.linear_attn_backend.forward_metadata.mamba_cache_indices` = `state_indices_list[bs-1]` | int32 [bs] | yes: allocated once in `init_cuda_graph_state` (`hybrid_linear_attn_backend.py:620-626`), handed to the captured `ForwardMetadata` (`_capture_metadata`, `707-742`) | `_replay_metadata` writes it in place: `fused_replay_state_indices(out_state_indices=...)` (`763-772`) or `copy_` (`774-783`); ReplaySSM-spec leaves `replayssm_write_pos_list` None (`_replayssm_enabled` is the decode-ring flag only, `576-590`), so the fused path is taken when `_fused_state_indices_ok` (`96-100`) |
| `be.linear_attn_backend.verify_intermediate_state_indices` | int32 [pool_size (+pad)] = arange | yes: built once in `KDAAttnBackend.__init__` (`kda_backend.py:524-529`), never rewritten | constant (row i = batch position i) |
| per-layer pools `req_to_token_pool.mamba2_layer_cache(lid)` (`memory_pool.py:1651-1652`, `527-540`) | views of persistent pool tensors | yes | n/a |

The forward_metadata Python object is rebuilt per replay but its tensors are the same objects; what a captured kernel binds is the
tensor seen at capture (`[:1]` slice = same data_ptr). In eager steps `_forward_metadata` builds fresh tensors
(`hybrid_linear_attn_backend.py:187-191`).

### B.2 MLA (triton backend, page_size 1)

| tensor | dtype / shape | meaning in verify | capture-stable? |
|---|---|---|---|
| `req_to_token_pool.req_to_token` | int32 [size+1, max_context_len] (`memory_pool.py:397-399`) | row r = token slot per position; positions seq_len..seq_len+7 already hold the 8 draft slots (`dspark_planner.py:841-849` reads them from it) | persistent pool tensor |
| `forward_batch.req_pool_indices` | int64 [max_bs] buffer, `[:bs]` (`buffers.py:160`) | request row | yes (registry slot) |
| `forward_batch.seq_lens` | int64 [max_bs] buffer (`buffers.py:161`) | KV length **before** the 8 tokens (`dspark_planner.py:836-846` prefix_lens; triton `seq_mask_len = T*(seq_lens+T)`, `triton_backend.py:625`; `dflash_info.py:142-143` "excludes the verify tokens") | yes |
| `forward_batch.out_cache_loc` | int64 [max_num_tokens] buffer, `[:8]` (`buffers.py:162`, `_cache_loc_dtype` int64 `decode_cuda_graph_runner.py:592-593`) | the 8 rows written = `req_to_token[req, seq_len:seq_len+8]` (`dspark_planner.py:841-849`, `dspark_verify.py:320`) | yes |
| `full_attn_backend.forward_metadata.kv_indptr` = `self.kv_indptr[:bs+1]` | int32 (`triton_backend.py:371-373`) | [0, seq_len] (`cumsum(seq_lens)`, `triton_backend.py:503-522`) | yes |
| `full_attn_backend.forward_metadata.kv_indices` = `cuda_graph_kv_indices` | int64 [max_num_tokens * max_context_len] (`triton_backend.py:1136-1143`) | `req_to_token[req, 0:seq_len]`, prefix only (plain pool: no translation, `srt/mem_cache/kv_index_translator.py:458-518`; unified pool only with `--enable-unified-memory`) | yes; refilled by `_update_target_verify_buffers` (`580-641`) from `_apply_cuda_graph_metadata` (`1311-1351`). Eager: fresh `kv_indices` each step (`941-962`) |
| `qo_indptr` | int64 [bs+1] = [0, 8] (`triton_backend.py:388-392, 590-597`) | | yes |

KV write and read in the verify forward (`DeepseekV2AttentionMLA` -> `MLA` -> `MLA_ROCM`,
`srt/models/deepseek_common/attention_backend_handler.py:239-258, 32-43`):
* `forward_absorb_rocm_core` else-branch (`forward_mla_rocm.py:966-992`; triton is not in
  `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS`, `srt/models/deepseek_common/utils.py:65-76`; `_skip_rope_for_aiter_fused_mla`
  False because rotary_emb is None, `forward_mla_rocm.py:1104-1120`): `q = cat(q_nope_out, q_pe)` [8,12,576],
  `k = cat(k_nope, k_pe)` [8,1,576], `v = k_nope` [8,1,512], `attn_mqa(q, k, k_nope, save_kv_cache=True)`.
* `TritonAttnBackend.forward_extend` (`triton_backend.py:1582`): `save_kv_cache` -> `_set_kv_buffer` (`1626-1635`) ->
  `MLATokenToKVPool.set_kv_buffer` (via the hybrid pool's layer-id map, `memory_pool.py:4325-4330, 4405+`):
  `kv_buffer[l][out_cache_loc] = k` (index_put, `memory_pool.py:4754-4786`). Buffer `[size+page_size, 1, 576]` bf16
  (`memory_pool.py:4671-4686`). Row = **[k_nope normed 512 | k_pe 64 unrotated]**, no RoPE anywhere (`skip_rope=True`,
  `kimi_k3.py:2159`; `deepseek_v2.py:2131-2144`).
* Attention: `use_verify_shared_kv` is True for K3 MLA on gfx95 with topk 1 (`triton_backend.py:90-96, 236-241`) ->
  `verify_shared_kv_fwd` (`kernels/ops/attention/verify_mla.py:725-819`): prefix rows through `kv_indices` / `kv_indptr`; the
  8 draft rows from `k_extend` / `v_extend` (the `k` / `k_nope` tensors), **not from the cache**. Details in C.3.

---------------------------------------------------------------------------------------------------------------------------

## C. MLA math (HIP, triton backend)

1. `latent = fused_qkv_a_proj_with_mqa(x)` [8, 2112] bf16 (`deepseek_v2.py:2474-2499`); split `[q 1536 | latent 576]`,
   `k_nope = latent[:, :512]` (`forward_mla_rocm.py:512-520`), `k_pe = latent[:, 512:]` (`forward_mla.py:191-193`).
2. Norms: HIP has no alt stream (`kimi_k3.py:2701`), `_use_aiter` -> aiter `fused_qk_rmsnorm_bf16(q, w_qa, eps, k_nope, w_kva, eps)`
   (`forward_mla_rocm.py:576-584`), bf16 out, eps 1e-5.
3. `q = q_b_proj(q_a)` -> view [8, 12, 192] (`deepseek_v2.py:2501-2513`, called `forward_mla_rocm.py:631`); `q_nope = [..., :128]`,
   `q_pe = [..., 128:]` (`forward_mla.py:191`).
4. Absorb: `w_kc` bf16 -> `torch.bmm(q_nope^T, w_kc)` bf16 out (`forward_mla_rocm.py:222-247`; the fp8/fp4 branches need
   uint8/fp8 `w_kc`), transposed to [8, 12, 512] (`716-719`). No fused bmm+rope+cache (`_can_fuse_bmm_rope_cat_and_cache` needs
   fp8 `w_kc` and a rotary_emb, `425-452`).
5. Attention (`verify_mla.py`), sm_scale = `layer.scaling` = 192^-0.5 (`deepseek_v2.py:2023`, no mscale with skip_rope), k_scale 1:
   * prefix, stage 1 (`56-222`): grid (bs, 12/4, num_splits); num_splits = min(32, 512 // 3) = 32 launched (`434-436`), active =
     max(1, min(32, seq_len // 64)) (`49-53`); split s covers `seq_len // active` rows, the last takes the remainder (`131-135`).
     Online softmax over 64-row blocks: `qk = dot(q_nope, c) + dot(q_pe, k_pe)` fp32, `* sm_scale` (`170-181`);
     `p = exp(qk - m)`; **P.V with p rounded to bf16** (`198`); `e_sum` = fp32 sum of the **unrounded** p (`199`); split output
     `acc / e_sum` **stored bf16** (`att_out` bf16, `213, 418-422`), `lse = m + log(e_sum)` fp32.
   * stage 2 (`225-376`): (a) fp32 LSE merge of the active splits' bf16 partials; (b) the 8x8 draft block **fully fp32**: q, k_extend,
     v_extend loaded bf16 -> fp32, scores `sum(q*k) * sm_scale`, causal `j <= i`, `pd = exp` unrounded, `o_draft = pd.v / sum pd`
     (`346-359`); (c) LSE merge prefix vs draft (`362-365`), output bf16 (`374`) into `o` [8, 12*512] bf16
     (`triton_backend.py:1609`).
6. `v = bmm(attn, w_vc)` bf16 -> [8, 1536] (`forward_mla_rocm.py:301-317, 371`).
7. Gate (`kimi_k3.py:2266-2287`): `gate = g_proj(hidden_states)` where `hidden_states` = the attention input (aggregation 1
   output) stashed in `forward` (`2340`), computed inline on HIP (no gate stream, `2321-2325`); `o_in = bf16(v * bf16(sigmoid(gate)))`
   via the JIT kernel (`kernels/ops/attention/mla_output_gate.py:34-53`) or `x * torch.sigmoid(gate)` (`kimi_k3.py:2278`): same
   rounding either way (bf16 sigmoid then bf16 product). Then `o_proj` (`forward_mla_rocm.py:1080`), `reduce_results=False`.

Differences from `B300: tests/mla_reference.py core()`:
* Attention core (C.5) differs from the trtllm-gen model in the reference (`mla_reference.py:57-68`): (i) the prefix is split into
  up to 32 parts whose normalized outputs are **rounded to bf16** and merged by LSE in fp32; (ii) the 8 draft keys use **fp32 p**
  (no bf16 rounding of exp) and fp32 V accumulation; (iii) online-softmax rescaling per 64-row block instead of one global max.
  The reference rounds `bf(e)` for every key and divides by one global fp32 sum.
* Everything else matches: latent bf16, RMSNorm bf16, q bf16, `q_abs` bf16, KV row `[k_nope | k_pe]` bf16 unrotated,
  scaling 192^-0.5, `v` bf16, gate bf16, `o_in = bf16(v * bf16(sigmoid(gate)))`, `o` bf16.
* Exact op implementations differ (aiter fused RMSNorm, aiter/torch GEMMs on HIP) - same rounding points, reduction order may differ.

---------------------------------------------------------------------------------------------------------------------------

## D. Weights after load + post_load_weights (compressed-tensors MXFP4 checkpoint: attention BF16)

### D.1 KDA (`KimiK3DeltaAttention`, `kimi_k3.py:1477-1780`)

| attribute | shape / dtype | notes |
|---|---|---|
| `fused_qkvg_proj.weight` | [6144, 7168] bf16, rows q 1536 \| k 1536 \| v 1536 \| g 1536 (this rank) | `1545-1561`; `split_sizes` [4608, 1536] |
| `b_proj.weight` | [12, 7168] bf16 | `1562-1571` |
| `f_a_proj.weight` | [128, 7168] bf16, replicated | `1572-1578` |
| `f_b_proj.weight` | [1536, 128] bf16 | `1579-1586` |
| `_bfa_w` | [144, 7168] = f_a 128 \| b 12 \| 4 zero rows; f_a/b weights become views | `_merge_bfa_weights` `1610-1647`; the ROCm whole-in-proj merge (`1649-1676`) needs `do_fuse_qkvbfg` = `quant_config is None` (`1537`), False for this checkpoint |
| `_bfa_f_b_w` | = `f_b_proj.weight` | `1645` |
| `qkv_conv1d.weight` | [4608, 1, 4] fp32, channels q \| k \| v (this rank's 1536 each); `attn.conv_weights` = [4608, 4] view; tap 3 = newest | `1699-1707, 1755`; bias None |
| `A_log` | [1, 1, 12, 1] fp32 (checkpoint 1-D narrowed at rank*12) | `1713-1733` |
| `dt_bias` | [1536] fp32 | `1692-1698` |
| `o_norm.weight` | [128], default dtype bf16 (inferred; file F32) | `1735-1737`; `FusedRMSNormGated` `kernels/ops/attention/fla/fused_norm_gate.py:408-433` |
| `o_proj.weight` | [7168, 1536] bf16, `reduce_results=False` | `1738-1754` |

`B300: kimi_k3/kda.py to_kernel_weights` takes `fused_qkvg_proj`, `f_a_proj`, `b_proj`, `f_b_proj`, conv [4608, 4] (squeezed),
`A_log` (reshape -1), `dt_bias`, `o_norm.weight`, `o_proj`, and `mlp_res_proj` / `mlp_res_norm` / `post_attention_layernorm`
weights: every one exists under the same name and shape in v0.5.21 (`f_a_proj` / `b_proj` are views into `_bfa_w`, values
unchanged). v0.5.19 has the same definitions (`v519: models/kimi_k3.py:1602-1646, 1740`). No difference.

### D.2 MLA (`KimiK3MLAAttention` over `DeepseekV2AttentionMLA`)

| attribute | shape / dtype | notes |
|---|---|---|
| `fused_qkv_a_proj_with_mqa.weight` | [2112, 7168] bf16 replicated, rows q_a 1536 \| kv_a 512 \| k_pe 64 | `deepseek_v2.py:2033-2039`; loader q_a -> rows [:1536], kv_a_proj_with_mqa -> rows [1536:] (`kimi_k3.py:3147-3167`) |
| `q_a_layernorm.weight` / `kv_a_layernorm.weight` | [1536] / [512], eps 1e-5 | `deepseek_v2.py:2041, 2127` |
| `q_b_proj.weight` | [2304, 1536] bf16, per head 192 rows: nope 128 \| rope 64 | `deepseek_v2.py:2042-2049`; split `forward_mla.py:191` |
| `kv_b_proj.weight` | [3072, 512] bf16 (12 x (128 k \| 128 v)) | `deepseek_v2.py:2109-2116` |
| `w_kc` | logical [12, 128, 512] bf16, strides (65536, 1, 128) (memory [12][512][128]) | `kimi_k3.py:3297-3300` |
| `w_vc` | logical [12, 512, 128] bf16, strides (65536, 1, 512) (memory [12][128][512]) | `kimi_k3.py:3301`; `w_scale` stays 1.0 on HIP (`3302-3317`) |
| `g_proj.weight` | [1536, 7168] bf16 | `kimi_k3.py:2238-2249` |
| `o_proj.weight` | [7168, 1536] bf16 | `deepseek_v2.py:2118-2125` |
| `scaling` | 192^-0.5; `rotary_emb` None | `deepseek_v2.py:2023, 2144` |

Same split code and layout in v0.5.19 (`v519: models/kimi_k3.py:3189-3195`). `B300: kimi_k3/mla.py to_kernel_weights` uses
`w_kc.reshape(12,128,512).transpose(1,2)` and `w_vc.reshape(12,512,128).transpose(1,2)` on the logical shapes, so the strides
do not matter. No difference.

---------------------------------------------------------------------------------------------------------------------------

## E. Every difference from the v0.5.19 contract our kernels implement

KDA:
1. Verify kernel: v0.5.19/B300 ran the CuTe `fused_kda_decode_mtp_dspark` (`v519: kda_backend.py:1356`); v0.5.21/HIP runs the
   Triton chain `causal_conv1d_update` + `fused_sigmoid_gating_delta_rule_update(cache_ring=True)`.
2. **`conv[slot]` is written during verify** on HIP (= pre-conv x5, x6, x7); v0.5.19 CuTe kept it read-only
   (`v519: kernels/ops/kimi_k3/kda_decode_mtp.py:965`). The commit overwrites it from `icw[row][accept_len-1]`, so after the commit
   the state matches whether or not we write it (inferred). Consequence for shadow mode: by the time `post()` runs, SGLang has
   already changed `conv[slot]`; the B300 `post()` docstring premise ("SGLang's verify does not write the pools' state or window")
   is false here. Shadow on AMD must snapshot `conv[slot]` in `pre()`.
3. Conv output is rounded to bf16 before the recurrence (v0.5.19 SGLang: fp32). Our kernel already rounds there.
4. Core output o is rounded to bf16 and normed by a separate `rms_norm_gated` (v0.5.19: o_norm fused into the CuTe recurrence).
   Whether our kernel rounds o before the norm: to check in the kernel source.
5. Window-row table goes through `select_verify_intermediate_state_indices`; same values unless `SGLANG_ENABLE_PP_SPEC=1`.
6. Commit gained an optional `src_indices_raw` (None by default); fold math unchanged (`diff` of
   `kda_replayssm_spec_decode.py`: only that argument).
Unchanged: ring tensor names/shapes/dtypes/positions/values, slot index, window layout and row, `temporal` read-only, weights.

MLA:
1. Backend: triton with page_size 1 instead of trtllm_mla with page_size 64. No `decode_trtllm_mla_metadata`, no
   `block_kv_indices`, no `_decode_kernel_loc`.
2. Length: `seq_lens` / `kv_indptr[1]` = rows **before** the 8 tokens (B300 `seq_lens_k` included them).
3. Row lookup: token slots, not pages: `req_to_token[req_pool_indices[0], pos]` (int32) or `kv_indices` (int64, prefix only);
   the 8 new rows are at `out_cache_loc[:8]` (int64) = `req_to_token[req, seq_len:seq_len+8]`.
4. Attention numerics (C.5): split prefix with bf16-rounded partials + fp32 draft block, versus the trtllm-gen model in
   `mla_reference.py`.
5. KV write path: torch index_put of `cat(k_nope, k_pe)` at `out_cache_loc` (`memory_pool.py:4754-4786`) - same bytes as before.
Unchanged: weights and their layouts, scaling, no RoPE, the gate formula and its rounding, q_abs / v bf16 rounding points.
