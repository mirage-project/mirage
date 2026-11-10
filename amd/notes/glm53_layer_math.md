# GLM-5.3-Flash decode on 4 x MI355X (TP4): bytes per GPU and the time they bound

From the checkpoint headers (`/dev/shm/glm53f` on the box, 2026-11-07) and `config.json`: hidden 4096, 45 layers
(34 KDA, 11 DSA = layers 3, 7, ..., 43), layers 0-2 dense MLP (intermediate 12288), the rest MoE (288 routed experts,
top 8, 1 shared expert, intermediate 2048, routed scaling 2.5, sigmoid + bias), mHC with 4 residual streams.
TP4 without EP (user 11-07): every GPU holds 1/4 of every expert (512 of the 2048 intermediate columns).

Bandwidth used below: 6 TB/s per GPU (K3 measured ~6.3 TB/s for W13 streams, ~5.5 for W2; `amd-megakernel-development`
section 8).

## Weights read per layer per GPU

| part | tensors (dtype) | bytes per GPU | at 6 TB/s |
|---|---|---|---|
| KDA attention (x 34) | q, k, v [8192, 4096] bf16 split by heads: 3 x 16.8 MB; o_proj [4096, 8192] bf16 / 4: 16.8 MB; b [64, 4096], f_a / g_a [128, 4096], f_b / g_b [8192, 128] bf16: ~3 MB | ~70 MB | ~11.7 us |
| DSA attention (x 11) | q_a [1536, 4096] fp8 6.3 MB + kv_a [512, 4096] fp8 2.1 MB (replicated); q_b [16384, 1536] fp8 / 4: 6.3 MB; kv_b [32768, 512] bf16 / 4: 8.4 MB; o_proj [4096, 16384] fp8 / 4: 16.8 MB; indexer wq_b [4096, 1536] bf16 12.6 MB + wk, weights_proj, kpool gate ~2.3 MB (replicated) | ~55 MB | ~9.1 us |
| MoE (x 42), per expert touched | W13 [2 x 512, 4096] + W2 [4096, 512] fp8 = 6.3 MB (+ scales) | 6.3 MB x (unique experts + 1 shared) | |
| router (x 42) | gate [288, 4096] bf16 (replicated) | 2.4 MB | 0.4 us |
| mHC (x 45) | hc_attn_fn, hc_ffn_fn [24, 16384] bf16 | 1.6 MB | 0.3 us |
| dense MLP (x 3) | gate, up [12288, 4096], down [4096, 12288] fp8 / 4 | 37.7 MB | 6.3 us |
| lm_head (SGLang's, not ours) | [154880, 4096] bf16 / 4 | 317 MB | 53 us |

Unique routed experts per layer with T tokens x top 8 of 288 (uniform routing: 288 (1 - (280/288)^T); real routing
is more concentrated, K3: ~90 of 384 at T 8 on B300):

| T | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| unique experts | 8 | 15.8 | 31 | 59 | 103 | 171 | 241 |
| MoE bytes per GPU per layer (+ shared + router) | 59 MB | 108 MB | 205 MB | 383 MB | 657 MB | 1.09 GB | 1.53 GB |
| at 6 TB/s | 9.8 us | 18 us | 34 us | 64 us | 110 us | 182 us | 255 us |

## Per decode step per GPU (weights only, uniform routing)

| T | KDA 34 x 70 MB | DSA 11 x 55 MB | MoE 42 layers | dense + mHC + router | total | at 6 TB/s |
|---|---|---|---|---|---|---|
| 1 | 2.38 GB | 0.61 GB | 2.48 GB | 0.29 GB | 5.76 GB | 0.96 ms |
| 8 | 2.38 | 0.61 | 16.1 | 0.29 | 19.4 | 3.2 ms |
| 64 | 2.38 | 0.61 | 64.3 | 0.29 | 67.6 | 11.3 ms |

Plus state and cache traffic: KDA state [16 heads][128][128] fp32 = 1 MB per request per layer read and written (T 64:
128 MB per layer, 4.4 GB per step: as large as the KDA weights); DSA latent cache fp8 (the baseline's
`--kv-cache-dtype fp8_e4m3`): the top-2048 positions x 512 B = 1 MB per request per layer, the indexer's pooled keys
(1 per 4 positions) x 128 B over the whole context.

Consequences:
- At small T the KDA projections (bf16, 2.4 GB per step) are as large as all MoE layers together: bf16 KDA weights are
  the first thing to stream at full rate.
- At T 64 the MoE experts are ~85 % of the bytes; the expert queue and its rate decide T 32 / 64 (as in K3).
- The KDA state traffic grows with T (T 64 ~4.4 GB per step): same order as K3's (98 MB per layer at T 64).

(To be compared with the SGLang baseline's measured per-group times, `glmbase1`, and updated.)
