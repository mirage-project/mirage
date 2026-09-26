# GLM-5.3 / GLM-5.3-Flash Sparse MLA 实现规划

## 范围与状态

基于 `mpk` 分支 `1f3338f9`，新增 BF16 sparse MLA device 模板、Python layer、
任务注册、独立 CUDA launcher、reference 与 MPK 集成测试。decode 与分块 prefill
使用同一计算主体，每个 query 拥有独立索引。首版针对 SM100，优先考虑 decode。

当前开发环境为 macOS ARM64，无 CUDA 工具链和 GPU。实现的 CUDA 编译、数值、
同步安全与性能必须在 SM100 上另行验证；静态检查不等于 GPU 验证通过。

不包含 indexer、IndexShare 调度、IndexPool、KDA、权重加载、KV 量化或通信。
调用方须先完成 Q 投影、位置编码、KV cache 写入和最终 token 索引的生成。

## 接口与数学语义

`PersistentKernel.sparse_mla_layer(q, kv_cache, token_indices, index_counts,
output, *, softmax_scale, num_splits=1)`。

- `q`: BF16 `[T, H, 512 + R]`，已吸收 key 投影的 latent query。
- `kv_cache`: BF16 `[num_pages, page_size, 512 + R]`，latent 后接位置编码部分。
- `token_indices`: INT32 `[T, K_capacity]`，请求内逻辑 token 位置。
- `index_counts`: INT32 `[T]`，需要读取的索引前缀长度，允许前缀内有 `-1`。
- `output`: BF16 `[T, H, 512]`，latent 输出，后续 value/output 投影在调用方。
- `R=64/0`，分别对应 GLM-5.3 / Flash sparse MLA；本地 head 数支持 8/16/32/64。
- 张量必须连续；复用 MPK query indptr 和分页 metadata；scale 必须显式指定。
- cache 包含本次 query 对应的新 token。query 位置为历史长度加块内位置。
- 有效索引必须唯一（调用方将 IndexPool 结果展开并去重）；无效、越界和未来
  token 被屏蔽。空选择输出零，内部 LSE 为负无穷。

对每个 query/head，仅对选中且可见的 token 计算
`scores = scale * Q @ KV.T`、`P = softmax(scores)`、`O = P @ KV[:, :512]`。
scale 来自模型原始 head 定义，不能按 512/576 的 latent 维度自动推导。

## 实现结构

- 新增独立 `sparse_mla_sm100.cuh`，不改变现有 DeepSeek MLA 的计算路径。
- 每个 CTA 256 线程，处理一个 query、16 个 head 槽位和一个索引分片；KV tile 64。
- 直接通过逻辑索引和页表访问 KV，向量化搬入单阶段共享内存；不生成全局 gather
  缓冲。用 BF16 Tensor Core MMA、FP32 累加和在线 softmax。
- 编译时特化 head 数、RoPE 维度、page size 和 split 数；索引容量作为运行时步长，
  避免仅因容量变化重复实例化 CUDA 代码；R=0 时移除
  位置编码部分。所有共享内存尺寸静态核对 MPK worker 预算。
- `num_splits=1/2/4/8`；1 直接输出，其他模式输出 FP32 partial O/LSE，由 reduce
  task 稳定合并。每次调用完整覆盖自己的输出槽位，空 split 不保留旧值。
- 独立任务枚举/注册/名称映射/runtime metadata；无 TMA descriptor。
  输出作为 graph output 登记，保证 producer/consumer 依赖正确。

## 验证

独立 reference 直接 gather 后做完整 softmax，不复制 kernel 的 tile/split 算法。
CPU 使用 NumPy 验证索引和数学语义，GPU 使用 PyTorch FP32 验证 BF16 输出。

覆盖 R=0/64、head 8/16/32/64、split 1/2/4/8、decode、纯 prefill、有历史的分块
prefill、多请求、非连续页面、部分页面、不同 query 索引、空选择、边界长度、
无效/未来索引、索引重排、K_capacity=2048 及更大容量、重复调用工作区覆盖。
有效重复索引属于不合法输入，reference 验证器报错。

全选择应匹配 dense MLA，远距离选择应区别于滑动窗口。输出 `atol=0.02`、
`rtol=0.02`，整体相对 RMS <= 2%；零输出精确为零，不能出现 NaN/Inf。
GPU 上另跑 Compute Sanitizer、独立 launcher 和 MPK 集成，记录不同 split 延迟，
不预先承诺加速比。复用旧 MLA 的回归测试作为现有路径兼容检查。

## 执行顺序

1. 写入规划；2. reference 和测试；3. CUDA 主体及 reduce；4. MPK 注册与示例；
5. 本机检查与 GPU 验证说明。GPU 环境缺失时交付状态为“实现待 GPU 验证”。

## 本次实现与验证记录

- 已新增 CUDA sparse MLA / split reduce、Python layer、任务枚举与注册、runtime
  metadata 和 profiler 名称。decode 和分块 prefill 共用逐 query 的稀疏访问主体。
- 使用 BF16 WMMA（与现有 MLA 相同的 Tensor Core 计算类别），不复用连续 TMA
  描述符。首版是正确性基线，尚不能称为已经优化完成的 decode kernel。
- 新增独立 CUDA launcher、MPK 集成测试、CPU reference、接口契约测试、延迟脚本
  和验证 README。MPK 测试包含 reduce 依赖及下游 sparse MLA 消费者。
- 本机通过 9 项 CPU 测试；5 项 GPU 测试明确跳过。Python 语法检查和
  `git diff --check` 通过；新增 CUDA 文件按仓库 clang-format 15 配置格式化。
- 独立 GPU 测试覆盖历史 decode 和混合请求；MPK 集成测试当前覆盖 offline 的
  单 query/多 query纯 prefill，历史 decode 的 MPK 端到端集成仍需 GPU 环境补验。
- NVCC 编译、GPU 数值正确性、Compute Sanitizer、性能及既有 CUDA MLA 回归
  均未运行。没有执行完整 GLM 权重推理，也没有提交或推送代码。
