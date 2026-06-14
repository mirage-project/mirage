"""PB layer alignment: DFlash `fc` + `hidden_norm` via MPK test-mode vs reference dump.

ctx = hidden_norm(fc(target_hidden))   (model-level, once per request)

Validates the REUSED linear (sm100) and rmsnorm kernels at real DFlash dims using the
real checkpoint weights and the HF-reference dump (demo/qwen3/dflash_correctness/dumps).

Run: CUDA_VISIBLE_DEVICES=1 python test_fc_hidden_norm_testmode.py
Prereq: ran ref_dump.py (--num-layers 1 --bs 1 --ctx-len 16) to produce the dumps.
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import load_weight, rms_norm, linear, EPS  # noqa: E402

DUMPS = os.path.join(os.path.dirname(__file__),
                     "../../../../demo/qwen3/dflash_correctness/dumps")


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS, fn))


def main():
    device, dtype = "cuda", torch.bfloat16

    # ---- reference inputs/outputs from the HF dump ----
    target_hidden = load_dump("in::target_hidden").to(device, dtype)  # [1,16,43008]
    ref_fc = load_dump("out::fc").to(device, torch.float32)           # [1,16,7168]
    ref_ctx = load_dump("out::hidden_norm").to(device, torch.float32)
    S = target_hidden.shape[0] * target_hidden.shape[1]
    KH = target_hidden.shape[-1]
    target_hidden = target_hidden.reshape(S, KH).contiguous()
    H = ref_fc.shape[-1]
    ref_fc = ref_fc.reshape(S, H)
    ref_ctx = ref_ctx.reshape(S, H)

    # ---- real weights ----
    fc_w = load_weight("fc.weight").contiguous()            # [H, KH]
    hn_w = load_weight("hidden_norm.weight").contiguous()   # [H]

    # ---- sanity: pure-torch reference path matches the dump ----
    fc_t = linear(target_hidden, fc_w).to(torch.float32)
    ctx_t = rms_norm(fc_t.to(dtype), hn_w, EPS).to(torch.float32)
    print(f"[torch-ref vs dump] fc maxerr  {(fc_t - ref_fc).abs().max().item():.4f}")
    print(f"[torch-ref vs dump] ctx maxerr {(ctx_t - ref_ctx).abs().max().item():.4f}")

    # ---- MPK test-mode graph ----
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

    fc_out = torch.zeros(S, H, dtype=dtype, device=device)
    ctx_out = torch.zeros(S, H, dtype=dtype, device=device)

    th_dt = pk.attach_input(target_hidden, name="target_hidden")
    fcw_dt = pk.attach_input(fc_w, name="fc_w")
    fco_dt = pk.attach_input(fc_out, name="fc_out")
    hnw_dt = pk.attach_input(hn_w, name="hn_w")
    ctxo_dt = pk.attach_input(ctx_out, name="ctx_out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_layer(input=th_dt, weight=fcw_dt, output=fco_dt,
                    grid_dim=(64, 1, 1), block_dim=block_dim)
    pk.rmsnorm_layer(input=fco_dt, weight=hnw_dt, output=ctxo_dt,
                     grid_dim=(S, 1, 1), block_dim=block_dim)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    fc_err = (fc_out.to(torch.float32) - ref_fc).abs().max().item()
    ctx_err = (ctx_out.to(torch.float32) - ref_ctx).abs().max().item()
    print(f"[MPK vs dump] fc  maxerr {fc_err:.4f}")
    print(f"[MPK vs dump] ctx maxerr {ctx_err:.4f}")
    pk.finalize()

    ok = fc_err < 0.1 and ctx_err < 0.05
    print("PASSED" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
