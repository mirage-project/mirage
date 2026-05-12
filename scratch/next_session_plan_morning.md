# Next session 优化执行计划 (2026-05-12 morning)

## TL;DR of 昨晚分析

Q/KV phase 100μs 慢的 root cause 找到了:
- DSv3 每层有 22 个 GEMM (vs Qwen3 9)
- FP8 dense smallm 是 persistent kernel (grid_dim=(128,1,1)), output_map=(-1,-1,-1)
- 这导致 event_dim GCD = 1 → coarse-grained event (num_triggers=128 等于所有 128 个 worker 都要完成)
- Worker queue FIFO, 不能 skip 已经 block 的 task → 在 phase 切换时 stall 严重
- 实测 worker 45 (busiest at 71.5%) 在 354μs 内有 200μs 是 idle gap (phase-transition bubbles)

## 三条实操路径 (按 ROI 排)

### Path 1: 融合 kv_a + kv_rope GEMM (MEDIUM effort, expect -10% Q/KV time)

**当前**: `kv_a_proj` (output 512) + `kv_a_proj_with_mqa rope part` (output 64→pad 128) = 2 个独立 FP8 GEMM, 共用 input `rmsnorm_out`.

**改法**: 拼接 weight 成 `[640, hidden]`, 单个 FP8 GEMM 输出 `[batch, 640]`, 然后 slice 给 `c_latent_out` 和 `k_pe_out`.

**改动文件**: `python/mirage/mpk/models/deepseek_v3/builder.py`
- 大约 line 1520-1550, 在 `_build_mla_attention_layer_with_prefix` (line 2376) 和 `_build_main_decoder_layer` (line ~1500)
- 在 weight load 时 concat (在 `_attach_fp8_weight` 之前): `w_kv_combined = torch.cat([w_kv_latent, w_kv_rope_padded], dim=0)`
- 同样 concat scale
- 输出用 一个 buffer `kv_combined_out = new_tensor(dims=(mbt, 640))`
- 然后 `c_latent_out` / `k_pe_out` 指向 `kv_combined_out[:, :512]` / `[:, 512:576]` (alias via `mpk.slice_layer` or direct buffer aliasing)

**风险**: alias 输出可能不被 builder 支持; 可能需要先写到 combined buffer 再 elementwise_copy 切分.

**验证**: 改后跑 `scripts/dpskv3_workload_compare.sh --tag T1 --prompt-len 50 --decode 8` 对比 token 准确性.

### Path 2: 调研 EVENT_LAUNCH_TASKS 启用 (HIGH risk, expect 大幅 util 提升)

**当前**: `src/kernel/runtime.cc:1011-1028` 把所有 EVENT_LAUNCH_TASKS 降级为 EVENT_EMPTY. 注释说 "selective-layer hangs".

**改法**: 注释掉降级, 跑测试看是否真的 hang. 如果只是 selective-layer 边界条件 bug, 可能能 fix.

**第一步实验**: 改 runtime.cc:1014 的 `if` 条件加一个 env-var gate: `if (getenv("MPK_TRY_FINEGRAINED_LAUNCH") && ...)` 那样测试时设 env 开关.

**风险**: 可能整个 megakernel 死锁. 必须在最简 config (layers 0-3, decode 1 token) 上先 sanity check.

**验证**: Qwen3 demo run (most contained); 然后 DSv3 layers 0-3 mbt=1.

### Path 3: 改 FP8 dense smallm 输出 partition 声明 (LOW effort, MIGHT WORK?)

**改法**: 在 `python/mirage/mpk/persistent_kernel.py:2337`:
```python
tb_graph.new_input(output, (-1, -1, -1), -1, True)
```
改成:
```python
tb_graph.new_input(output, (1, -1, -1), -1, True)  # 声明 grid_x partitions dim 1
```

**理论效果**: 给 runtime event_dim 计算更细的 GCD. 但 consumer 端 input_map 也必须用 (1, -1, -1) 才生效, 而 consumer 端可能也是 persistent. 所以可能没效果.

**风险**: 不影响 kernel 行为 (kernel 仍用 worker_idx 内部分配 work), 但可能影响 task 调度. 需要单测.

**验证**: 跑 layers 0-3 confirm 正确性 + 看 trace 中 num_triggers 分布是否变化.

## 不要先做的 (确认不值得)

- ❌ `MPK_ALLREDUCE_TILE_SIZE` tuning — 昨晚已确认 noise 大于 signal
- ❌ `MPK_MLA_TP4_V_SPLITS` tuning — V=2 看着略好但 1 sample 不可靠
- ❌ MoE W13 / W2 kernel — 已反馈同学
- ❌ Standalone bench FP8_DENSE_SMALLM — 没有现成 bench, 写一个要 ~1-2h

## 验证流程 (每次 fix 后)

1. Quick smoke: `python demo/qwen3/demo.py --use-mirage --max-new-tokens 8` — 确保不 hang/break Qwen3
2. DSv3 correctness: `bash scripts/dpskv3_workload_compare.sh --tag T<n> --prompt-len 100 --decode 32 --layers 0-19` — check tokens match
3. Re-run perfetto decode trace: 
   ```
   OUT=/home/muhengl/mirage/outputs/perfetto_decode_after_<change>_$(date +%H%M%S)
   ... (same args as outputs/perfetto_decode_fresh_20260511_235240/)
   ```
4. Use `mpk-perf-analyzer` agent (next session, after Claude restart) or manually re-analyze comparison file

## Reference

- `scratch/mpk_vs_vllm_perf_comparison.md` — 主要发现
- `~/.claude/agents/mpk-perf-analyzer.md` — 下次 session 自动跑分析
- 当前 baseline trace: `outputs/perfetto_decode_fresh_20260511_235240/`
- Qwen3 baseline trace: `outputs/qwen3_perfetto_20260512_001933/`
