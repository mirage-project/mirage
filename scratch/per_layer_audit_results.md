
## Run 2026-05-13 13:55:54

| layer | μs/call | cos | vLLM target μs | gap μs | shape |
|---|---|---|---|---|---|
| fp8_dense_smallm_q_a_baseline_n1536 | -1.00 | 0.0000 | 10.0 | -11.00 | ERROR: The size of tensor a (7168) must match the size of tensor b (128) at non-singleton dimension 2 |
| fp8_dense_smallm_qkv_a_fused_n2176 | -1.00 | 0.0000 | 13.0 | -14.00 | ERROR: The size of tensor a (7168) must match the size of tensor b (128) at non-singleton dimension 2 |
| quantize_fp8_slice_n1536_from_2176 | 9.38 | 1.0000 |  |  | quantize_fp8 slice (BATCH=128 HIDDEN=1536 GLOBAL=2176) |

## Run 2026-05-13 13:56:22

| layer | μs/call | cos | vLLM target μs | gap μs | shape |
|---|---|---|---|---|---|
| fp8_dense_smallm_q_a_baseline_n1536 | -1.00 | 0.0000 | 10.0 | -11.00 | ERROR: sb shape must be [N/128, K/128] |
| fp8_dense_smallm_qkv_a_fused_n2176 | -1.00 | 0.0000 | 13.0 | -14.00 | ERROR: sb shape must be [N/128, K/128] |
| quantize_fp8_slice_n1536_from_2176 | 9.89 | 1.0000 |  |  | quantize_fp8 slice (BATCH=128 HIDDEN=1536 GLOBAL=2176) |

## Run 2026-05-13 13:56:56

| layer | μs/call | cos | vLLM target μs | gap μs | shape |
|---|---|---|---|---|---|
| fp8_dense_smallm_q_a_baseline_n1536 | 58.72 | 1.0000 | 10.0 | +48.72 | fp8_gemm_dense_smallm (baseline q_a: M=128 N=1536 K=7168) |
| fp8_dense_smallm_qkv_a_fused_n2176 | 57.79 | 1.0000 | 13.0 | +44.79 | fp8_gemm_dense_smallm (qkv_a fused: M=128 N=2176 K=7168) |
| quantize_fp8_slice_n1536_from_2176 | 9.66 | 1.0000 |  |  | quantize_fp8 slice (BATCH=128 HIDDEN=1536 GLOBAL=2176) |
