"""The dense MLP of GLM-5.3 layers 0 .. first_k_dense_replace - 1, on the static megakernel, for shard `rank` of `tp`.

The checkpoint stores gate_proj, up_proj [12288, 6144] and down_proj [6144, 12288] in e4m3 with one fp32 scale per 128 x 128
block. TP splits the intermediate dimension, as for the shared expert of a MoE layer: GPU `rank` holds gate / up rows and down
columns [rank IR, rank IR + IR), IR = 12288 / tp. The layer is the shared-expert path of demo/glm53_moe with no routed experts:

  group_quant (x) -> scaled_gemm_tile (gate_up) -> silu_and_mul -> group_quant (h) -> scaled_gemm_tile (down)
    -> allreduce_send -> sum_gpus -> residual_add (y = MLP output + residual, a graph tensor)

The tensors (mirage.mpk.fp8_fp32scale_weights piece layout, the e4m3 bytes and fp32 scales unchanged):
  gu [2 IR, 6144] uint8, gu_sf [IR / 128, 48, 2] fp32   gate / up rows, 64 gate rows then their 64 up rows in each piece
  dn [6144, IR] uint8, dn_sf [48, IR / 128] fp32        down
"""
import os

import torch
from mirage.core import bfloat16, float32, int64, uint8
from mirage.mpk.fp8_fp32scale_weights import pack_w13, pack_w2

T = int(os.environ.get("STATIC_TOKENS", "8"))   # tokens (rows) per step: the kernel's T (8, 16 or 32)


def dense_tensors(read, layer: int, tp: int = 8, rank: int = 0, device="cpu") -> dict:
    """Return the dense MLP tensors of `layer` for shard `rank` of `tp`, in the layouts of the module docstring."""
    p = f"model.layers.{layer}.mlp."
    get = lambda proj: (read(p + proj + ".weight").view(torch.uint8).to(device),
                        read(p + proj + ".weight_scale_inv").float().to(device))
    (gate, gs), (up, us), (down, ds) = get("gate_proj"), get("up_proj"), get("down_proj")
    assert gate.shape[0] % (128 * tp) == 0, f"intermediate {gate.shape[0]}: not whole 128-row blocks for each of {tp} GPUs"
    IR = gate.shape[0] // tp
    rows, blocks = slice(rank * IR, rank * IR + IR), slice(rank * IR // 128, (rank * IR + IR) // 128)
    gu, gu_sf = pack_w13(gate[rows], gs[blocks], up[rows], us[blocks])
    dn, dn_sf = pack_w2(down[:, rows].contiguous(), ds[:, blocks].contiguous())
    H = gate.shape[1]
    return {"gu": gu.view(2 * IR, H), "gu_sf": gu_sf, "dn": dn.view(H, IR), "dn_sf": dn_sf}


def build_graph(mpk, w: dict, next_norm=None):
    """Declare the 8 layer calls. w: the dense_tensors weights and the graph tensors x, residual, mlp_out (bf16 [T, H]).
    next_norm: (gamma, output, eps), torch tensors attached last ("norm_w", "x_next"): mlp_out is the new residual and the
    kernel also writes RMSNorm(mlp_out) gamma, the next sublayer's input."""
    if next_norm is not None:    # the norm's h32 scratch, gamma and output: more than config.cuh's 24 buffer slots
        mpk.max_bufs = max(getattr(mpk, "max_bufs", None) or 0, 32)
    t = {name: mpk.attach_input(torch_tensor=v, name=name) for name, v in w.items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    H, IR = w["dn"].shape
    xq, gu = nt((T, H), uint8, "x_fp8"), nt((T, 2 * IR), int64, "gate_up")
    h, hq, d = nt((T, IR), bfloat16, "act"), nt((T, IR), uint8, "act_fp8"), nt((T, H), float32, "down")
    mpk.group_quant_layer(input=t["x"], output=xq)                                                    # x in FP8, fp32 scales
    mpk.scaled_gemm_tile_layer(input=xq, weight=t["gu"], weight_scales=t["gu_sf"], output=gu, combine="add")   # gate_up
    mpk.silu_and_mul_layer(input=gu, output=h)                                                        # SiLU(gate) * up
    mpk.group_quant_layer(input=h, output=hq)                                                         # h in FP8, fp32 scales
    mpk.scaled_gemm_tile_layer(input=hq, weight=t["dn"], weight_scales=t["dn_sf"], output=d, combine="store")  # down
    sent, dsum = nt((T, H), bfloat16, "down_sent"), nt((T, H), bfloat16, "down_sum")
    mpk.allreduce_send_layer(input=d, output=sent)                                                    # to every GPU
    mpk.sum_gpus_layer(input=sent, output=dsum)                                                       # over the GPUs
    mpk.residual_add_layer(addend=dsum, residual=t["residual"], output=t["mlp_out"],                  # y = MLP + residual
                           counted=next_norm is not None)
    if next_norm is not None:
        gamma, out = (mpk.attach_input(torch_tensor=v, name=n) for v, n in zip(next_norm[:2], ("norm_w", "x_next")))
        mpk.row_rmsnorm_layer(input=t["mlp_out"], gamma=gamma, output=out, eps=next_norm[2])
