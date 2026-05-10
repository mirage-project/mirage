"""MPK end-to-end smoke for fp8_group_gemm_decode_layer at one config."""
import math
import os
import sys

import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_wrapper import (make_inputs, torch_reference)  # noqa: E402

device = "cuda"


CFGS = {
    "gate_up_M1":  (1,  32, 7168, 4096),
    "gate_up_M4":  (4,  32, 7168, 4096),
    "gate_up_M8":  (8,  32, 7168, 4096),
    "gate_up_M16": (16, 32, 7168, 4096),
    "down_M1":     (1,  32, 2048, 7168),
    "down_M4":     (4,  32, 2048, 7168),
    "down_M8":     (8,  32, 2048, 7168),
    "down_M16":    (16, 32, 2048, 7168),
}


def main():
    name = os.environ.get("CFG", "gate_up_M16")
    MPE, E, K, N = CFGS[name]
    print(f"shape {name}: MPE={MPE} E={E} K={K} N={N}")

    A_fp8, B_fp8, sfa_packed, sfb_packed, D, m_indices, sa, sb_block = make_inputs(MPE, E, K, N)
    A_u8 = A_fp8.view(torch.uint8)
    B_u8 = B_fp8.view(torch.uint8)
    M_total = E * MPE

    nw, nsch = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=nsch,
        max_num_batched_tokens=M_total,
        max_num_batched_requests=1,
        max_seq_length=1,
    )
    pk = PersistentKernel(**params)

    A_dt = pk.attach_input(A_u8, name="A_fp8")
    B_dt = pk.attach_input(B_u8, name="B_fp8")
    sfa_dt = pk.attach_input(sfa_packed, name="sfa_packed")
    sfb_dt = pk.attach_input(sfb_packed, name="sfb_packed")
    mi_dt = pk.attach_input(m_indices, name="m_indices")
    D_dt = pk.attach_input(D, name="D_bf16")

    pk.fp8_group_gemm_layer(
        a_fp8=A_dt,
        b_fp8=B_dt,
        sfa_packed=sfa_dt,
        sfb_packed=sfb_dt,
        m_indices=mi_dt,
        output=D_dt,
        num_workers=nw,
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    print("compiling...", flush=True)
    pk.compile(output_dir=folder)
    print("running...", flush=True)
    pk.run_test_mode()
    torch.cuda.synchronize()

    D_ref = torch_reference(A_fp8, B_fp8, sa, sb_block, m_indices, MPE, E, K, N)
    err = (D.float() - D_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    status = "OK" if (mean_err < 1e-2 and max_err < 1.0) else "FAIL"
    print(f"{name} max_err={max_err:.4f} mean_err={mean_err:.4f} [{status}]")
    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
