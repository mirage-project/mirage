# PR674 `fp8_group_gemm_smallm/largem_sm100` 测试用例 Bug 报告

## 一、Bug 现象（一句话）

`test_mpk_smoke.py` 在 MPE ∈ {1, 4, 8} 等 sparse 场景下 **看似 PASS（max_err ≈ 0）**，但实际 kernel 在这些场景下产生**静默错误数据**。原因是 **测试用的 reference 实现 (`test_wrapper.py:torch_reference`) 跟 kernel 犯了同一个错误**，二者互相自洽地"通过"了测试，**等于没测**。

## 二、Kernel 的真实行为

Kernel 文件：`include/mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_sm100_common.cuh`

- 第 114 行：`constexpr int BM = 128, BK = 128`（BM 是硬编码，不是 template 参数）
- 第 121 行：`int const nm = (M_total + BM - 1) / BM;` —— 按 BM=128 分 tile
- 第 211-212 行：

  ```cpp
  int m_start = bm * BM;
  int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
  ```

  **每个 BM=128 行的 tile 只读 `m_indices[bm*BM]` 一次**（= 该 tile 第一行的 expert id），整个 tile 的 128 行都用这一个 expert。

- 第 214 行：`int on = expert_id * N + bn * BN;` —— B 矩阵地址用单一 `expert_id`，没有 per-row 切换的可能。

- 第 34-35 行注释：

  ```
  m_indices[M_total]  int32, expert id per row (rows in
                       [bm*BM, (bm+1)*BM) must share expert)
  ```

  作者知道这个约束，写了进去。但是 **runtime 没有任何 assert 来检查这个约束是否被满足**。

**结论**：kernel **只在 "每 128 行 A 输入属于同一个 expert" 的输入下才正确**。任何破坏这个对齐的 sparse 输入都会产生静默错误数据 —— 一个 BM=128 tile 里如果跨多个 expert，rows 1..127 会用 row 0 的 expert weight 算 GEMM。

## 三、测试为什么没发现

测试文件：`tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_wrapper.py`

- 第 61 行（`make_inputs`）：

  ```python
  m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
  ```

  这种构造方式让 `m_indices` 永远是"连续 MPE 个相同 expert id"的 balanced contiguous 排布。

- 第 87-89 行（reference 实现）：

  ```python
  for bm in range(0, M_total, BM):
      block_end = min(bm + BM, M_total)
      expert_id = int(m_indices[bm].item())
  ```

  reference **也只读 `m_indices[bm]`**，对整个 BM=128 tile 用同一个 expert id 算 GEMM —— 跟 kernel 做完全一样的"错误"近似。

- `test_mpk_smoke.py` 的 `CFGS` 字典只测了 E=32、MPE ∈ {1, 4, 8, 16}（共 8 个配置）：
  - MPE ∈ {4, 8, 16} 时 MPE 整除 128，BM=128 tile 内永远只有一个 expert，对齐天然满足，kernel 正确
  - MPE=1 时 M_total=32 < BM=128，kernel 只跑一个 partial tile，TMA OOB_FILL_NONE 把 rows 32..127 填 0、epilogue store 只写 32 行到 global memory，**碰巧也正确**（详见下面附注）

测试**完全没覆盖** DSv3 真实 top_k=8 sparse routing 那种"每个 expert 拿到 0..k 个 token、token 数不均衡"的输入。

## 四、commit message 里的 "MPE 1-1024 win vs DeepGEMM" 是误导

`82bab699` commit message 写了在 MPE 1-1024 范围内对 DeepGEMM 1.05-1.42x 全赢。这是在 **同一种 balanced contiguous m_indices + 小 E (=32)** 测试条件下得出的，跟真实 DSv3 prefill (E=256, top_k=8, 平均每 expert ~4 tokens 的高度 sparse 长尾分布) 的输入完全不同。这个 win 不能外推到 sparse 场景。

## 五、修复建议

请负责该 kernel 的同事按以下方式改：

### 5.1 修测试覆盖（必须）

`test_wrapper.py:make_inputs` 增加一个 "sparse routing" 输入模式。建议加 `routing="sparse"` 参数：

```python
def make_inputs(MPE, E, K, N, seed=0, routing="balanced"):
    # ...
    if routing == "balanced":
        # 现有行为
        m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
    elif routing == "sparse":
        # 模拟 DSv3 真实 top-k=8 routing:
        # M_total = batch * top_k, 每个 token 选 top_k 个 random expert
        # 然后按 expert 排序，得到不均衡的 m_indices
        batch = M_total // 8  # 假设 top_k=8
        topk = torch.randint(0, E, (batch, 8), device=device, dtype=torch.int32)
        # 排序，每行 token 复制 8 次配 8 个 expert id
        flat_token = torch.arange(batch, device=device).repeat_interleave(8)
        flat_expert = topk.flatten()
        sort_idx = torch.argsort(flat_expert, stable=True)
        m_indices = flat_expert[sort_idx]
        # A 也要相应 gather
        # ...
    elif routing == "tail":
        # 长尾：80% expert 拿到 < BM=128 tokens，20% expert 拿到 ≥ BM
        # ...
```

### 5.2 修 reference 实现（必须）

`test_wrapper.py:torch_reference` 改成 **per-row expert lookup**：

```python
def torch_reference(A_fp8, B_fp8, sa, sb_block, m_indices, MPE, E, K, N):
    M_total = A_fp8.shape[0]
    A = A_fp8.float()
    B = B_fp8.float()
    out = torch.zeros(M_total, N, dtype=torch.float32, device=A.device)
    sa_q = ue8m0_round_trip(sa)
    sb_block_q = ue8m0_round_trip(sb_block)
    sb_q_full = sb_block_q.repeat_interleave(128, dim=1)[:, :N, :]
    nk = K // 128
    for row in range(M_total):
        expert_id = int(m_indices[row].item())  # PER-ROW，不是 per-block
        for ki in range(nk):
            a = A[row, ki*128:(ki+1)*128]
            b = B[expert_id, :, ki*128:(ki+1)*128]
            partial = a @ b.T
            sa_s = sa_q[row, ki]
            sb_r = sb_q_full[expert_id, :, ki]
            out[row] += partial * sa_s * sb_r
    return out.to(torch.bfloat16)
```

注意：这个 reference 是"数学上正确"的，不再跟 kernel 自洽。**当 kernel 在 sparse 输入下跑时，它跟这个 reference 会有大差异 —— 那才是真正暴露 bug 的信号**。

### 5.3 改完测试后会看到的现象

- `routing="balanced"` 的现有 case：kernel 和 corrected reference 都正确 → 仍然 PASS。
- `routing="sparse"` 的新 case：kernel 用了错误的 expert weight → max_err 会很大（几个数量级以上），FAIL。

这就**暴露了 kernel 的真实限制**：它只支持 balanced contiguous m_indices，不支持 sparse routing。

### 5.4 Kernel 本身需不需要改？

**取决于使用场景**：

- 如果这个 kernel 只用于 "dense MoE / 大 batch prefill 已经 permute+pad 好" 的场景，那 kernel 现在的行为是 OK 的，只需要**把约束写进 docstring 里更显眼一些** + 在 wrapper 加 runtime assert 检查 `m_indices` 满足 BM=128 对齐 + 提供一个 helper 函数做 padding/permute 预处理。
- 如果想让它直接吃 DSv3 这种 sparse routing 的输入（不 padding），那 kernel 要重写：TMA 的 B 加载逻辑现在是按 `expert_id * N + bn * BN` 一次性 bulk-copy，per-row expert 切换需要 redesign（要么 per-row TMA 小 load 牺牲吞吐，要么放弃 group GEMM 形式）。

**推荐路径**：保持 kernel 现有设计 + 修测试以正确反映约束 + 在 Python wrapper 加 padding/permute 预处理 helper（让上游可以选择是否用这个 kernel）。

## 六、附注：为什么 MPE=1 测试也 "通过"

MPE=1, E=32, M_total=32 < BM=128：

- Kernel `nm = (32 + 127) / 128 = 1`，跑一个 partial BM tile
- A 的 TMA descriptor 用 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（`runtime_kernel_wrapper.cu:76`），rows 32..127 OOB 部分被自动零填到 shared memory
- B 用 `expert_id = m_indices[0] = 0` 加载 expert 0 的 weight
- MMA 计算 D[0..127, :] += A_zero_padded @ B[expert 0]
- Epilogue TMA store 用 `gd[2] = {N, M_total=32}` 的 extent（`runtime_kernel_wrapper.cu:113`），**硬件只 commit row 0..31 到 global memory**，rows 32..127 的"错误"输出被丢弃

但 **row 0..31 的输出仍然是用 expert 0 的 weight 算的**！而 reference 用 `m_indices[bm=0] = 0` 做同样的近似 → 二者再次自洽。

如果让 reference 正确地按 row 用 `m_indices[row]` 算（rows 1..31 用 expert 1..31 的 weight），跟 kernel 的输出会有巨大差异。这就是 5.2 节那个修复的核心。
