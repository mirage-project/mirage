# M2-I8 test artifacts

Machine-readable results from the runs that gated the Qwen3.5 registry builder +
weight loader, on `catalyst-B200` (GPU 1, exclusive, 3-sample idle guard) at
repo HEAD `b1e1e16` plus this issue's files.

| file | produced by | what it pins |
|---|---|---|
| `transforms.json` | `test_loader_transforms.py` | every §2.0 / §5.2 transform vs the M2-I3 oracle, **plus the q/k representation decision table** |
| `layer_gdn.json` | `test_qwen35_layer_testmode.py --phase gdn` | 21 op boundaries of layer 0 (GDN + MoE) through the real megakernel |
| `layer_attn.json` | `test_qwen35_layer_testmode.py --phase attn` | 21 op boundaries of layer 3 (full attention + MoE) |

## The one design decision these artifacts settle

MPK's SM100 attention rotates NeoX pairs `(i, i+128)`; Qwen3.5 rotates `(j, j+32)`.
The load-time column permutation that reconciles them (`v1-architecture.md` §4.4)
therefore *always* moves weight rows across 128-row `weight_scale_inv` block
boundaries — both members of a rotated pair live in the first 64 columns of a
head, i.e. inside one scale block, and MPK's partner is exactly one block away.
So the shipped `[N/128, K/128]` scale cannot be permuted alongside the rows, and
the attention q/k projections need a representation choice that the rest of the
dense path does not.

Measured against the oracle's own `q_proj_out` / `k_proj_out`
(`transforms.json` -> `qk representation decision`):

| representation | q | k |
|---|---|---|
| no permutation (unreachable floor) | **0.0 (bit-exact)** | **0.0 (bit-exact)** |
| fp8 rows permuted, per-block max rescale | 2.12e-3 | 1.55e-2 |
| exact dequant -> permute -> bf16 GEMM | 4.53e-3 | 1.74e-2 |

`fp8_permute` is the pinned default: closer on q, comparable on k, and it keeps
every dense projection on the one preserved-block-scale representation and the
fp8 byte budget. 24.9 % of q's (row, k-block) entries and 50 % of k's are
rescaled once; the rest come through unchanged.

The bit-exact floor is the notable number: MPK's `linear_fp8_blockscale_sm100`
reproduces HF's Triton `finegrained-fp8` output **exactly** on real checkpoint
weights when the weight is untouched. The residual delta above is entirely the
price of the RoPE permutation, and it is the largest single numerics term the
attention path carries into M2-I9.
