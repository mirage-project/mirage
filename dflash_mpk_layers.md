# DFlash on MPK — 分层（Layer）实现清单

> 目标：把 `vllm_dflash.md` 的每个阶段拆成 MPK 的一个 **layer**，列出每个 layer 内的
> **kernel**、是 **复用（REUSE）** 还是 **新建（NEW）**、以及 **输入 / 输出**。
>
> 参照实现：MPK 现有的 **Eagle3** 路径（算法最接近 DFlash）。
> - 复用算子分布在 `include/mirage/persistent_kernel/tasks/{ampere,hopper,blackwell,speculative_decoding}/`
> - Python 编排参照 `python/mirage/mpk/models/eagle3/builder.py`（`Eagle3Builder`）
>
> 记号（沿用 `vllm_dflash.md`，`z-lab/Qwen3-8B-DFlash-b16`，TP=1）：
> `B`=block=16，`s`=B-1=15，`K`=捕获层数=5，`L`=draft层数=5，`H_t`=4096(K2.6:7168)，
> `H_d`=4096，`n_q/n_kv/d`=32/8/128，`q_size/kv_size`=4096/1024，`I`=12288，`V`=151936，
> `MASK`=151669，`bs`=batch，`n`=单请求前缀长，`S=Σn`=总 context token 数。
> dtype：bf16（RMSNorm/RoPE/softmax 内部 fp32）。
>
> 状态图例：✅ REUSE（现成可用） · 🟡 REUSE+（现成算子，但需新编排/包装） · 🔴 NEW（需新写 kernel）

---

## 总览（每阶段 → 一个 layer）

| Layer | 阶段 | 触发频率 | 含 forward? | 主要状态 |
|---|---|---|---|---|
| L0 init | STAGE 0 权重预处理 | 启动一次 | 否 | 🟡 融合 KV 权重预拼为 NEW |
| L1 capture | §1 捕获 K 层 aux hidden | 每次 target 前向后 | 否 | ✅ |
| L2 fc | §2 `combine_hidden_states` | 每轮物化前 | 否 | ✅ |
| L3 input-prep | §3 `copy_and_expand_dflash_inputs` | 每轮 draft 前 | 否 | 🔴 |
| L4 materialize-kv | §4 `precompute_and_store_context_kv` | prefill 后 + 每轮 verify 后 | 否 | 🔴（写 KV）+ ✅（norm/proj/rope） |
| L5 draft-fwd | §5 非因果 block 前向 | 每轮 1 次 | **是** | 🔴（非因果 attn）+ ✅（其余） |
| L6 sample | §6 选 draft token | 每轮 1 次 | 否 | 🟡（gather）+ ✅（argmax/d2t） |
| L7 verify | §7 chain 验证 | 每次 target 验证后 | 否 | ✅ |

> 「重计算」只在 **L5（draft 前向）** 与 target 的 prefill/verify 前向里发生；
> L2/L4/L6 都是投影 + cache 写 + 选择，无 attention 前向。

---

## L0 · Init（启动一次，权重预处理）

把 draft 权重 shuffle 成 MPK 融合布局，并预拼出「KV 物化」要用的融合权重。

| kernel / 步骤 | 状态 | 说明 |
|---|---|---|
| QKV / gate-up shuffle 成 MPK 融合布局 | ✅ REUSE | `Eagle3Builder._prepare_weights` 同款（`shuffle_tensors`） |
| `fc` 权重落盘 (`K·H_t × H_d`) | ✅ REUSE | 仅形状比 Eagle3(3H→H) 宽 |
| `d2t` 表构建 | ✅ REUSE | `eagle3` 同款 |
| **`_fused_kv_weight` 预拼** = 各层 `qkv_proj.weight[q_size:]` concat → `[L·2·kv_size, H_d]` | 🔴 NEW | vLLM `_build_fused_kv_buffers`；MPK 无此融合 KV 权重缓冲，需新写预处理 |
| `_k_norm_weights[L]`、RoPE `cos_sin_cache` 预堆叠 | 🟡 REUSE+ | 算子有，需按「多层一次过」布局预堆叠 |

- **输入**：draft state_dict（`qkv_proj/k_norm/fc/d2t/...`）、`rope` 配置
- **输出**：`w_fc[H_d, K·H_t]`、`_fused_kv_weight[L·2·kv_size, H_d]`、`_k_norm_weights[L,d]`、
  `cos_sin_cache`、shuffle 后的 `w_qkv / w_gateup / w_o`、`d2t` 表

---

## L1 · Capture aux hidden（§1）

把 target K 个指定层的 hidden 拷进专用 aux buffer（MPK 逐层中间量会被复用，必须另存）。

| kernel | 状态 | 文件 |
|---|---|---|
| `copy_layer_kernel<T, BATCH, HIDDEN>` | ✅ REUSE | `speculative_decoding/eagle3_ops.cuh` |

- **输入**：target 第 `k` 层 hidden `[S, H_t]` bf16（k ∈ K 个捕获层）
- **输出**：`target_hidden_states [S, K·H_t]` bf16（K 层拼接；K2.6=`[S, 35840]`）
- 备注：Eagle3 捕获 3 层，DFlash K=5，仅调用次数/拼接宽度差异。

---

## L2 · fc projection（§2 `combine_hidden_states`）

把 K 层拼接 hidden 投影到 draft 隐藏维。

| kernel | 状态 | 文件 |
|---|---|---|
| `concat_layer`（若 capture 未拼好）| ✅ REUSE | `eagle3_ops.cuh: concat_kernel` |
| `linear_layer`（`ReplicatedLinear K·H_t→H_d`，无 bias）| ✅ REUSE | `ampere/linear.cuh`（或 hopper/blackwell 变体） |

- **输入**：`target_hidden_states [S, K·H_t]` bf16，`w_fc [H_d, K·H_t]` bf16
- **输出**：`ctx_hidden [S, H_d]` bf16
- 备注：`hidden_norm` **不在这里**，在 L4-① 内做。

---

## L3 · Input prep（§3 `copy_and_expand_dflash_inputs`）🔴 NEW

一个融合 kernel，grid `(bs, num_blocks)`，一次产出 6 个张量。MPK 无对应件
（Eagle3 不做 MASK-block 展开，也不在 kernel 内由 block_table 算 slot）。

| kernel | 状态 |
|---|---|
| `dflash_expand_inputs_kernel`（建议名）| 🔴 NEW（初期可先用 eager torch 顶上） |

- **输入**：`next_token_ids(bonus t₀) [bs]`、`target_positions [S]`、
  `block_table [bs, max_blk] int32`、`query_start_loc [bs+1]`、`seq_lens [bs]`、`block_size`
- **输出**（6 个）：
  - `out_input_ids [bs·B] int32` = 每请求 `[t₀, MASK×15]`
  - `context_positions [S] int64`（= `target_positions` 拷贝）
  - `query_positions [bs·B] int64` = `last_valid_pos+1+offset(0..15)`
  - `context_slot_mapping [S] int64` = `blk_id·block_size + pos%block_size`
  - `query_slot_mapping [bs·B] int64`（同上，对 query 位置）
  - `token_indices_to_sample [bs·s] int32` = 仅 15 个 MASK slot（跳过 bonus slot）

---

## L4 · Materialize KV（§4 `precompute_and_store_context_kv`）

不跑 draft 前向，直接把 target context hidden 投影成各 draft 层 K/V 并写入 draft KV cache。
六个子 op：算子大多现成，**唯独 ⑥「独立写 KV cache」是 NEW**（MPK 里 KV 一向是 attention
kernel 的副产物，没有脱离 attention 的写入算子）。

| 子op | kernel | 状态 | 文件 / 说明 |
|---|---|---|---|
| ① hidden_norm | `rmsnorm_layer`（fp32 内部，eps=1e-6）| ✅ REUSE | `ampere/rmsnorm.cuh` / `norm_*` |
| ② 融合 KV GEMM（所有 L 层一个大 GEMM）| `linear_layer` | 🟡 REUSE+ | `linear.cuh`；需配 L0 的 `_fused_kv_weight[L·2·kv_size, H_d]` |
| ③ reshape/permute → `all_k/all_v [L,S,n_kv,d]` | view/permute | 🟡 REUSE+ | 纯布局；MPK 可能需一个 layout/transpose 包装 |
| ④ 逐层 k_norm（×L）| `rmsnorm_layer` | ✅ REUSE | 同 ① |
| ⑤ 多层融合 RoPE（positions 重复 L 次，一次调用）| `rotary_embedding_layer` | 🟡 REUSE+ | `ampere/rotary_embedding.cuh`；需「所有层一次过、positions repeat L」编排 |
| ⑥ 逐层写 KV cache @ `context_slot_mapping` | `dflash_kv_cache_store_kernel`（建议名）| 🔴 NEW | **MPK 无独立 KV-write**；需新写「把投影好的 K/V 写进 paged cache 任意 slot」 |

- **输入**：`ctx_hidden [S, H_d]`、`_hidden_norm_weight [H_d]`、`_fused_kv_weight [L·2·kv_size, H_d]`、
  `_k_norm_weights [L, d]`、`cos_sin_cache`、`context_positions [S]`、`context_slot_mapping [S]`、
  draft `paged_k_cache / paged_v_cache`
- **输出**：写入 **draft KV cache** 的 L 层 context K/V（无张量返回，副作用写 cache）
- 触发：prefill 后（整段 prompt）+ 每轮 verify 后（1~B 个 commit token 重物化）。

---

## L5 · Draft non-causal forward（§5）— 含唯一的 forward

一次非因果前向，只处理 `bs·B=bs·16` 个 query token（context KV 已由 L4 预填）。
逐 op 看：除 **attention 非因果**外全部现成。

| op（每 DecoderLayer，×L=5）| kernel | 状态 | 文件 / 说明 |
|---|---|---|
| embed `[t₀,MASK×15]` | `embed_layer` | ✅ REUSE | `ampere/embedding.cuh`；MASK 是普通 token id |
| input_layernorm（fused add+norm）| `rmsnorm_layer` | ✅ REUSE | |
| qkv proj | `linear_layer` | ✅ REUSE | `w_qkv [q_size+2·kv_size, H_d]` |
| q_norm / k_norm（per-head RMSNorm d=128）| `rmsnorm_layer` | ✅ REUSE | |
| RoPE(q,k) | `rotary_embedding_layer` | ✅ REUSE | |
| **attention：16 query 双向看 [context S + block 16]** | `dflash_noncausal_attention_layer`（建议名）| 🔴 NEW | 现有 `single_batch_extend.cuh`/`multitoken_paged_attention` **全是 causal**（见 `single_batch_extend.cuh:444/454/486`）；需非因果 block 变体 |
| 写本 block 16 个 K/V @ `query_slot_mapping` | （并入上面的 attn kernel 副产物）| 🟡 REUSE+ | 与现有 attn 一样可在 kernel 内写；注意是非因果路径 |
| o_proj（+residual）| `linear_with_residual_layer` | ✅ REUSE | `RowParallel q_size→H_d` |
| post_attention_layernorm | `rmsnorm_layer` | ✅ REUSE | |
| MLP gate_up | `linear_layer` | ✅ REUSE | `[2·I, H_d]` |
| MLP silu·up | `silu_mul_layer` | ✅ REUSE | `ampere/silu_mul.cuh` |
| MLP down | `linear_layer` | ✅ REUSE | `[H_d, I]` |
| 末层 final norm | `rmsnorm_layer` | ✅ REUSE | |

- **输入**：`out_input_ids [bs·16]`、`query_positions [bs·16]`、`query_slot_mapping [bs·16]`、
  draft KV cache（含 L4 预填的 context）、各 draft 权重
- **输出**：`final_hidden [bs·16, H_d]` bf16
- 关键约束：attention 必须 **非因果**（vLLM 断言 `causal is False`）；16 个 MASK 靠绝对
  position 区分（同 embedding 不同 position）；slot j 的输出预测 slot j 处的 token（fill-in-the-mask，无 next-token 平移）。

---

## L6 · Draft token selection（§6）

只对 15 个 MASK slot 选 token。

| op | kernel | 状态 | 文件 / 说明 |
|---|---|---|---|
| gather 15 个 MASK slot：`final_hidden[token_indices_to_sample]` | `dflash_gather_sample_kernel`（建议名）/ 或复用 index-select | 🟡 REUSE+ | 小工；MPK 无现成命名件，可加薄 gather |
| TP-safe sharded argmax（或 lm_head + argmax）| `argmax_partial_layer` + `argmax_reduce_layer` | ✅ REUSE | `ampere/argmax.cuh`（`get_top_tokens` 同思路）|
| draft→target vocab remap | `eagle3_d2t_remap_kernel` | ✅ REUSE | `eagle3_ops.cuh`（b16 vocab 相等可跳过）|

- **输入**：`final_hidden [bs·16, H_d]`、`token_indices_to_sample [bs·15]`、`d2t`、（可选 `lm_head` 权重）
- **输出**：`draft_token_ids [bs, 15]`（target-vocab 空间），chain = `[t₀, d₁…d₁₅]`（16）

---

## L7 · Target verify（§7）

target 对 `[t₀..d₁₅]` 跑 chain-causal 前向（在 target 侧，非 draft），再按最长前缀验证。

| op | kernel | 状态 | 文件 / 说明 |
|---|---|---|---|
| chain 验证（最长匹配前缀 + bonus t₀′）| `target_verify_greedy_kernel` / `mtp_verify_strict` | ✅ REUSE | `speculative_decoding/target_verify.cuh` / `target_verify_mtp.cuh` |
| commit token 重物化 → 回到 L4 | （触发 L4）| — | 见 L4 |

- **输入**：`draft chain [bs, 16]`、target argmax `[bs, 16]`
- **输出**：`accepted_len [bs]`、`new_tokens`、新 bonus `t₀′`；并捕获 commit token 的 K 层
  hidden 送回 L1/L2/L4 做下一轮。

---

## 工作量小结

**需要新写的 kernel（🔴）共 3 个核心 + 1 小工：**
1. **L5 非因果 block attention** — DFlash 与 Eagle3 的根本区别，决定 draft 能否单次前向完成。
2. **L4-⑥ 独立 KV-cache 写入** — MPK 首个脱离 attention 的 KV 写算子（配 L0 融合 KV 权重预拼）。
3. **L3 input-prep 融合 kernel** — MASK 展开 + slot_mapping（初期可 eager torch）。
4. **L6 gather**（小） — 选 15 个 MASK slot。

**需要新编排但算子现成（🟡）：** L0 融合 KV 权重预拼、L4-②③⑤（融合 KV GEMM / 布局 / 多层 RoPE）。

**直接复用（✅）：** L1 capture、L2 fc、L4-①④（norm）、L5 除 attn 外全部、L6 argmax+d2t、L7 verify。

**Python 编排：** 仿 `Eagle3Builder` 写 `DFlashBuilder`，但 draft 循环是 **单次非因果前向**（非 Eagle3 的 K 步自回归 `build_draft_loop`），并接上 L3/L4。
