"""PB layer alignment: DFlash draft-layer MLP (SiLU gated) via MPK test-mode vs dump.

mlp(h) = down_proj( silu(gate_proj(h)) * up_proj(h) )

Input = out::layers.0.post_attention_layernorm (the normed MLP input from the HF dump).
Output compared to out::layers.0.mlp (before residual add).

Validates REUSED linear + silu_mul + shuffle_tensors at real DFlash dims (I=18432, H=7168).
Run: CUDA_VISIBLE_DEVICES=1 python test_mlp_testmode.py
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import load_weight, linear, silu_mul  # noqa: E402

DUMPS = os.path.join(os.path.dirname(__file__),
                     "../../../../demo/qwen3/dflash_correctness/dumps")


def grid_for_linear(out):
    """gate/up fused linear feeding silu_mul: tasks each emit 96 output rows
    (shuffle interleaving aligns to this). Used only for the gate_up linear."""
    if out % 96 == 0:
        return out // 96
    assert out % 64 == 0, f"unsupported linear out={out}"
    return out // 64


def grid_for_plain_linear(out):
    """Standalone linear (q/k/v/o/fc/down): fixed 96 or 64 tasks (matches
    mirage.mpk.models.utils.grid_for_rmsnorm_linear_layer; empirically the grid
    the cutlass linear accepts — out//64 task counts can hang/corrupt)."""
    if out % 96 == 0:
        return 96
    assert out % 64 == 0, f"unsupported linear out={out}"
    return 64


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS, fn))


def main():
    device, dtype = "cuda", torch.bfloat16

    h = load_dump("out::layers.0.post_attention_layernorm").to(device, dtype)  # [1,8,7168]
    ref_mlp = load_dump("out::layers.0.mlp").to(device, torch.float32)
    S = h.shape[0] * h.shape[1]
    H = h.shape[-1]
    h = h.reshape(S, H).contiguous()
    ref_mlp = ref_mlp.reshape(S, H)

    gate_w = load_weight("layers.0.mlp.gate_proj.weight")  # [I,H]
    up_w = load_weight("layers.0.mlp.up_proj.weight")      # [I,H]
    down_w = load_weight("layers.0.mlp.down_proj.weight")  # [H,I]
    I = gate_w.shape[0]

    # torch reference path
    gate_t = linear(h, gate_w)
    up_t = linear(h, up_w)
    act_t = silu_mul(gate_t, up_t)
    mlp_t = linear(act_t, down_w).to(torch.float32)
    print(f"[torch-ref vs dump] mlp maxerr {(mlp_t - ref_mlp).abs().max().item():.4f}")

    # MPK test-mode
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = S
    params["max_num_batched_requests"] = 1
    params["use_cutlass_kernel"] = True
    pk = PersistentKernel(**params)

    gateup_tasks = grid_for_linear(2 * I)   # split for shuffle
    down_tasks = grid_for_plain_linear(H)

    mlp_mid = torch.zeros(S, 2 * I, dtype=dtype, device=device)
    silu_out = torch.zeros(S, I, dtype=dtype, device=device)
    mlp_out = torch.zeros(S, H, dtype=dtype, device=device)

    h_dt = pk.attach_input(h, name="mlp_in")
    gw_dt = pk.attach_input(gate_w.contiguous(), name="gate_w")
    uw_dt = pk.attach_input(up_w.contiguous(), name="up_w")
    gu_dt = pk.shuffle_tensors(inputs=[gw_dt, uw_dt], shuffled_dim=0,
                               num_groups=gateup_tasks // 2, name="gateup_w")
    dw_dt = pk.attach_input(down_w.contiguous(), name="down_w")
    zero_res = torch.zeros(S, H, dtype=dtype, device=device)
    mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")
    silu_dt = pk.attach_input(silu_out, name="silu_out")
    res_dt = pk.attach_input(zero_res, name="zero_res")
    out_dt = pk.attach_input(mlp_out, name="mlp_out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_layer(input=h_dt, weight=gu_dt, output=mid_dt,
                    grid_dim=(gateup_tasks, 1, 1), block_dim=block_dim)
    pk.silu_mul_layer(input=mid_dt, output=silu_dt,
                      grid_dim=(gateup_tasks // 2, 1, 1), block_dim=block_dim)
    pk.linear_with_residual_layer(input=silu_dt, weight=dw_dt, residual=res_dt,
                                  output=out_dt, grid_dim=(down_tasks, 1, 1),
                                  block_dim=block_dim)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    # localize: intermediate gate_up (interleaved) and silu_out
    import torch as _t
    gi = gate_t.reshape(S, gateup_tasks // 2, I // (gateup_tasks // 2))
    ui = up_t.reshape(S, gateup_tasks // 2, I // (gateup_tasks // 2))
    mid_ref = _t.stack([gi, ui], dim=2).reshape(S, 2 * I)  # interleaved per group
    mid_err = (mlp_mid.to(_t.float32) - mid_ref).abs().max().item()
    silu_err = (silu_out.to(_t.float32) - act_t.to(_t.float32)).abs().max().item()
    err = (mlp_out.to(torch.float32) - ref_mlp).abs().max().item()
    print(f"[MPK vs ref] gate_up(interleaved) maxerr {mid_err:.4f}")
    print(f"[MPK vs ref] silu_out maxerr {silu_err:.4f}")
    print(f"[MPK vs dump] mlp maxerr {err:.4f}")
    pk.finalize()
    ok = err < 0.5
    print("PASSED" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
