# DFlash (Kimi K2.6) on MPK — 测试进度

分支 `dflash-k26`（fork 自 `mpk`）。单卡，B200 / sm100。
Oracle = HF reference `dflash.py`（真权重）dump 出的张量；K2.6 target 超出范围。
设计 spec：`docs/superpowers/specs/2026-06-13-dflash-k26-design.md`
环境/坑：见 memory `dflash-k26-infra`。

配置（K2.6 b8）：B=8, ctx_len(测试)=16, K=6, L=6, H=7168, n_q/n_kv=64/8, d=128,
I=18432, eps=1e-5, YaRN RoPE, 0–4 层 sliding(2048)/第5层 full（ctx_len=16 时窗口不生效）。

## 测试矩阵

| 测试文件 | 阶段 | 验证内容 | 方式 | 结果 (relmax) | 状态 |
|---|---|---|---|---|---|
| `test_norm_rope.py` | kernel | per-head RMSNorm(1e-5)+YaRN RoPE 新 kernel | standalone wrapper | <0.002 | ✅ |
| `test_dflash_attn.py` | kernel | 非因果 sliding-window split-KV attention 新 kernel | standalone wrapper | <0.003 | ✅ |
| `test_dflash_attn_testmode.py` | kernel | 同上，但在真 megakernel 里跑（7/8-file wiring） | test-mode | 0.0011 | ✅ |
| `test_fc_hidden_norm_testmode.py` | layer | `fc`+`hidden_norm`（复用 linear sm100/cutlass + rmsnorm） vs dump | test-mode | fc 0.0312 / ctx 0.0156 (abs) | ✅ |
| `test_mlp_testmode.py` | layer | MLP gate_up→silu→down（复用） vs `out::layers.0.mlp` | test-mode | 0.0005 | ✅ |
| `verify_attention_ref.py` | oracle | 证明 torch attention reference 复现 dump（K3 的对齐基准） | torch | 0.0037 | ✅ |
| `test_draft_layer_testmode.py` | **layer** | **整层 DecoderLayer** vs `out::layers.0` | test-mode | **0.0040** | ✅ |
| `test_draft_model_testmode.py` | **e2e** | **整 6 层 draft 模型 final_hidden** vs `dumps6/final_hidden` | test-mode | **0.0148** | ✅ |
| `test_down_isolate.py` | debug | 隔离 down linear（调试用） | test-mode | <0.004 | ✅(辅助) |

`pytorch_reference.py` = 所有 op 的 canonical torch 参考（rms_norm/linear/silu_mul/
dflash_norm_rope/dflash_attention(_core)），两类测试共用。

## 三阶段对齐结论

- **Stage 1 kernel** ✅ — 两个新 kernel（`dflash_attention_sm100`、`dflash_norm_rope_sm100`）
  standalone + test-mode 都过；复用 kernel(linear/rmsnorm/silu_mul) 也对真权重 dump 验证过。
- **Stage 2 layer** ✅ — 整层对齐 `out::layers.0`，relmax 0.0040。
- **Stage 3 e2e（draft）** ✅ — 整 6 层模型 `final_hidden` 对齐，relmax 0.0148
  （6 层 bf16 累积，符合预期）。3004 tasks / megakernel。

## 怎么跑

```bash
# kernel standalone（需先 build 一次本地扩展）
cd tests/runtime_python/blackwell/sm100_dflash
python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=3 python test_norm_rope.py
CUDA_VISIBLE_DEVICES=3 python test_dflash_attn.py

# 重新生成 reference dump（1 层 / 6 层）
CUDA_VISIBLE_DEVICES=2 python demo/qwen3/dflash_correctness/ref_dump.py --num-layers 1 --ctx-len 16
CUDA_VISIBLE_DEVICES=2 python demo/qwen3/dflash_correctness/ref_dump.py --num-layers 6 --ctx-len 16 \
    --out demo/qwen3/dflash_correctness/dumps6

# test-mode（整层 / 整模型）；改了 .cc 后需先强制 relink core.so（见 memory）
CUDA_VISIBLE_DEVICES=2 python test_draft_layer_testmode.py
CUDA_VISIBLE_DEVICES=2 python test_draft_model_testmode.py
```

## 尚未做（非「kernel-layer-e2e」三阶段范围）

- **TP=8**（PD）：kernel 写法已 TP-aware，但 sharding / allreduce 接线 + TP8-vs-TP1 对齐未做。
- **context-KV materialize 搬进 MPK**：现在是 torch 物化后当 cache 输入喂进去（用已验证的
  reference 函数）；产品化要写独立的 MPK materialize pass（`mbt=ctx_len`）。
- **token-emit + verify 接线**：薄复用层；出 token 需要 target 的 lm_head（共享、超出范围），
  所以只对齐到 `final_hidden` 边界。
- 更长 ctx 下 sliding-window 的端到端验证（ctx_len=16 时窗口不生效；kernel 级已单独测过遮蔽）。

## 换到有权重的机器上：交接 / 下一步

有 K2.6 target 权重后能做：真 captured `target_hidden`、真 lm_head 出 token、对齐 vLLM、TP=8。

**0. 先确认分支在新机器能 build + 现有测试全绿（不需要 target，纯 dump 自洽）**
```bash
git checkout dflash-k26
# 重新生成 dump（需要 draft 权重 /raid/catalyst/models/Kimi-K2.6-DFlash-tmp，已 hf 下载）
python demo/qwen3/dflash_correctness/ref_dump.py --num-layers 1 --ctx-len 16
python demo/qwen3/dflash_correctness/ref_dump.py --num-layers 6 --ctx-len 16 --out demo/qwen3/dflash_correctness/dumps6
# build core（注意：改过 .cc 后 build_ext 不会自动 relink core.so，要强制）：
touch python/mirage/_cython/core.pyx
rm -f build/lib*/mirage/core*.so build/temp*/python/mirage/_cython/core.o python/mirage/core*.so
python setup.py build_ext --inplace          # 确认 core.so mtime 变了
# 跑 e2e
cd tests/runtime_python/blackwell/sm100_dflash && python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=<free> python test_draft_model_testmode.py   # 期望 relmax ~0.015 PASSED
```
注意：standalone wrapper 测试在某些机器上会 hang（cluster-launch 那类），优先用 test-mode 验证。
GPU0 在本机是坏的；新机器先 probe 一个能跑 `torch.randn(device='cuda')` 的卡。

**1. context-KV materialize 搬进 MPK**（去掉 torch 物化边界）：写一个 `mbt=ctx_len` 的
pass：`linear(ctx,k_w)`→`dflash_norm_rope`→ 存成 cache 张量；和现在 torch 物化的结果对齐即可。

**2. token-emit + verify**（需要 target lm_head 权重）：
`final_hidden[token_indices_to_sample(=B-1个 MASK槽)]` → `lm_head(target)` → TP-safe argmax →
（draft_vocab==target_vocab，b8 应无需 d2t）→ draft chain `[t0,d1..d_{B-1}]` →
复用 `target_verify` / `mtp_verify_strict`。新增小 kernel：K1 input-prep（MASK 展开+slot_mapping）、
K4 gather（选 MASK 槽）—— 详见 spec / `dflash_mpk_layers.md`。

**3. 对齐 vLLM（真 e2e token-match）**：vLLM 源在 `/home/letianr/vllm`（conda env `vllm`，
`vllm/v1/spec_decode/dflash.py`）。用真 target 跑一次 dflash，dump 出 `target_hidden` + draft
tokens；把同一份 `target_hidden` 喂给 MPK draft（替换现在 dump 的 noise/ctx 来源），比对 draft tokens。
oracle ≠ token-exact 的坑参考 memory `mtp-verify-oracle-not-sglang`。

**4. TP=8**：q/kv head 分片、fc 复制、lm_head vocab-parallel、allreduce；先 TP8 vs TP1 自对齐。
现有 kernel 模板参数已按 head 数推导，TP 下传 `q_size/TP`、`kv_size/TP` 即可，但 attention 的
GQA 分组和 o_proj 的 RowParallel + allreduce 需要接线。

提交记录（`git log mpk..dflash-k26`）：P0 → PA(kernels) → PA/PB(norm_rope) → PB(MLP/finding) →
PB(整层) → PC(整模型) → spec/status。
