# PR674 FP8 Group GEMM：BM=128 约束 + chunk prefill 适用性分析

## 问题 1：BM=128 + 每 BM block 一个 expert 是故意设计还是 bug？

**结论：test case coverage gap，不是有意的设计 trade-off。**

### 证据链

1. **约束在头文件有文档** (`fp8_group_gemm_sm100_common.cuh:34-35`)：注释明确写 "rows in [bm\*BM, (bm+1)\*BM) must share expert"。作者知道这个前提。

2. **kernel 实现严格依赖此前提** (`fp8_group_gemm_sm100_common.cuh:212-214`)：
   ```
   int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
   int on = expert_id * N + bn * BN;
   ```
   每个 BM tile 只读 `m_indices[bm*BM]` 一次，用第一行的 expert 加载 B 矩阵。若 [bm\*BM, (bm+1)\*BM) 行属于不同 expert，则后续行用错了 B 矩阵块 — **静默算错，无 assert**。

3. **测试只覆盖 balanced 场景** (`test_wrapper.py:61`)：
   ```python
   m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
   ```
   M_total = E × MPE，每个 expert 恰好连续 MPE 行。测试的 MPE ∈ {1, 4, 8, 16}，E=32，M_total ∈ {32, 128, 256, 512}。

4. **MPE ≥ 1 且 BM|MPE 时条件成立的充要条件**：若 MPE ≥ 128，则每个 BM block 天然在同一个 expert 内（balanced 分布）。若 MPE < 128，需要 128 能整除 MPE（即 MPE ∈ {1, 2, 4, 8, 16, 32, 64, 128}）且 routing 是 perfectly contiguous。测试用的 MPE=16 意味着一个 BM block 跨 128/16=8 个 expert——**测试 reference 也用同样的错误假设** (`test_wrapper.py:89`: `expert_id = m_indices[bm].item()`)，所以 kernel 和 reference 互相自洽，测不出 bug。

5. **commit message 明写测试范围** (`82bab699`)："E=32, gate\_up K=7168 N=4096 / down K=2048 N=7168, MPE 1-1024"。MPE 高达 1024 说明测了大 batch decode，但 E 始终 =32（DSv3 的 256 expert 被人工缩小了）。这个配置下 M_total=E×MPE 总是 balanced 连续分布。

**真实 DSv3 sparse routing 的 m_indices 不满足此前提**：top_k=8 routing 后，同一个 expert 的 token 不一定连续，且每个 expert 的 token 数可能从 0 到几十不等，远小于 BM=128。DSv3 builder 实际走的是 `moe_w13_fp8_sm100` (`TASK_MOE_W13_FP8_SM100=248`) 而非 PR674 新增的 `fp8_group_gemm_largem_sm100` (`TASK_FP8_GROUP_GEMM_LARGEM_SM100=312`)，所以 PR674 目前没被 builder 接入，尚未遇到 sparse routing 场景。

---

## 问题 2：DSv3 chunk prefill 能不能用 PR674 kernel？

**结论：在现实 sparse routing 下，无法通过调大 mbt 来让 PR674 kernel 正确工作。**

### (a) BM=128 要求的是什么？

BM=128 是**输入 A 的 M 维度分块大小**，要求每个 128 行 tile 内所有行属于同一个 expert。这与 chunk size（每次 prefill 处理多少 token）是不同的概念：

- chunk size = mbt（每次 scheduler 下发的 token 数量）
- BM=128 要求的是 token 按 expert 排序后连续分配，且每段至少 128 行

两者正交：即使 chunk size 是 128 的整数倍，sparse routing 也不能保证每个 expert 分到 ≥128 个 token。

### (b) 需要多大 mbt 才能保证每个 expert ≥128 token？

DSv3 参数：num_experts=256（EP=2 时本地 128），top_k=8，E=本地 expert 数。

理想均匀分布时：mbt × 8 / E ≥ 128，即 mbt ≥ 128 × E / 8。
- EP=2，E=128：mbt ≥ 2048
- EP=1，E=256：mbt ≥ 4096

但 routing 是 sparse 的，长尾分布不可避免。即使 mbt=4096，在实际负载下仍有 expert 只有 0-20 个 token，远低于 BM=128。**没有任何有限的 mbt 能保证 PR674 kernel 正确运行真实 sparse routing。**

### 正确路径

要让 PR674 类 kernel 用于 sparse prefill，需要在 Python 侧预处理：把 token 按 expert 分组、padding 到 BM 对齐，再排序输入到 kernel（类似 DeepGEMM 的 `m_grouped_fp8_gemm_nt_contiguous` 接口）。这正是现有 `moe_w13_fp8_sm100` 走的思路（通过 routing indices + mask 间接实现分组），PR674 的 kernel 目前没有这个前处理机制。
