"""Kernel unit test for `sigmoid_gate_mul_add_sm100` (task id 238, M2-I7).

    r' = residual + sigmoid(x . w_sg^T) * shared

Two independent checks:

  A. SYNTHETIC -- against a torch reference that reproduces the kernel's declared
     cast positions (bf16 logit, bf16 sigmoid, one rounding on the epilogue), plus
     a COUNTERFACTUAL fp32-throughout reference. The counterfactual must be
     measurably further away; otherwise the test would pass for a kernel that got
     the cast positions wrong, which is the failure mode M2-I4/I5 kept hitting.

  B. ORACLE -- against HF's own dumps for MoE layers 0 and 3, decode and prefill:
     `shared_gate_logit`, `shared_gate_sigmoid` and `shared_output_gated` are all
     dumped, so the gate GEMV, the sigmoid and the multiply are each checked at
     their own boundary rather than only at the end.

Run:  python tests/runtime_python/blackwell/sm100_moe_block_qwen35/test_sigmoid_gate_mul_add.py
"""

import json
import os

import torch

import runtime_kernel_blackwell_moe_block_qwen35 as mk

DEVICE = "cuda"
HIDDEN = 2048
ORACLE = os.environ.get(
    "QWEN35_ORACLE_DUMPS", os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")
)


def ref_hf_casts(x, w_sg, shared, residual):
    """The kernel's contract: fp32 GEMV -> bf16 logit -> fp32 sigmoid -> bf16 gate
    -> fp32 multiply-add -> bf16 out. Matches `Qwen3_5MoeSparseMoeBlock.forward`,
    where `shared_expert_gate` is a bf16 linear and `F.sigmoid` on a bf16 tensor
    returns bf16."""
    logit = (x.float() @ w_sg.float().t()).to(torch.bfloat16)
    gate = torch.sigmoid(logit.float()).to(torch.bfloat16)
    return (residual.float() + gate.float() * shared.float()).to(torch.bfloat16), logit, gate


def ref_fp32_throughout(x, w_sg, shared, residual):
    """Counterfactual: no intermediate rounding anywhere."""
    logit = x.float() @ w_sg.float().t()
    gate = torch.sigmoid(logit)
    return (residual.float() + gate * shared.float()).to(torch.bfloat16)


def run(x, w_sg, shared, residual):
    out = torch.zeros_like(residual)
    mk.sigmoid_gate_mul_add_sm100(x, w_sg, shared, residual, out)
    torch.cuda.synchronize()
    return out


def frob(a, b):
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def load(mode, layer, key):
    man = json.load(open(os.path.join(ORACLE, mode, "manifest.json")))
    return torch.load(
        os.path.join(ORACLE, man["tensors"][f"{layer}.{key}"]["file"]),
        map_location=DEVICE,
    )


def main():
    torch.manual_seed(20260726)

    # ---------------- A. synthetic sweep ----------------
    for batch, out_size, hidden in ((1, 256, 256), (4, 256, 256), (3, 512, 256),
                                    (1, 2048, 2048), (2, 2048, 2048),
                                    (8, 2048, 2048), (16, 2048, 2048)):
        x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=DEVICE)
        w_sg = (torch.randn(1, hidden, dtype=torch.bfloat16, device=DEVICE) * 0.02)
        shared = torch.randn(batch, out_size, dtype=torch.bfloat16, device=DEVICE)
        residual = torch.randn(batch, out_size, dtype=torch.bfloat16, device=DEVICE)
        got = run(x, w_sg, shared, residual)
        ref, logit, gate = ref_hf_casts(x, w_sg, shared, residual)
        alt = ref_fp32_throughout(x, w_sg, shared, residual)
        e_ref, e_alt = frob(got, ref), frob(got, alt)
        # The GEMV order differs from torch's, so exact equality is not the bar;
        # agreement at the bf16 output floor is, and the counterfactual must not
        # be closer.
        floor = frob(ref.float().to(torch.bfloat16), ref)
        print(
            f"  [{batch},{out_size},{hidden}] frob_vs_hf_casts={e_ref:.3e} "
            f"frob_vs_fp32_throughout={e_alt:.3e} gate={gate.flatten()[0].item():.6f}"
        )
        assert e_ref <= 3e-3, f"kernel disagrees with its declared cast positions: {e_ref:.3e}"
        assert e_ref <= e_alt + 1e-9, (
            "the fp32-throughout counterfactual fits at least as well -- the test "
            "cannot distinguish the cast positions it claims to pin"
        )
        assert floor == 0.0

    # gate must actually depend on x: a zero gate weight gives sigmoid(0) = 0.5
    x = torch.randn(2, 2048, dtype=torch.bfloat16, device=DEVICE)
    zero_w = torch.zeros(1, 2048, dtype=torch.bfloat16, device=DEVICE)
    shared = torch.randn(2, 2048, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.zeros(2, 2048, dtype=torch.bfloat16, device=DEVICE)
    got = run(x, zero_w, shared, residual)
    torch.testing.assert_close(
        got.float(), (0.5 * shared.float()).to(torch.bfloat16).float(), rtol=0, atol=0
    )
    print("  zero-gate-weight identity (sigmoid(0)=0.5) exact")

    # ---------------- B. oracle ----------------
    for mode in ("decode", "prefill"):
        for layer in ("moe0", "moe3"):
            x = load(mode, layer, "layer_input")
            w_sg = load(mode, layer, "__weight.shared_expert_gate_weight")
            shared = load(mode, layer, "shared_down_proj_out")
            hf_logit = load(mode, layer, "shared_gate_logit")
            hf_sig = load(mode, layer, "shared_gate_sigmoid")
            hf_gated = load(mode, layer, "shared_output_gated")
            residual = torch.zeros_like(shared)
            got = run(x.contiguous(), w_sg.contiguous(), shared.contiguous(), residual)

            # boundary 1+2: the gate GEMV and the sigmoid, recomputed from the
            # same bytes with the kernel's declared casts
            logit = (x.float() @ w_sg.float().t()).to(torch.bfloat16)
            assert torch.equal(logit.view(torch.int16), hf_logit.view(torch.int16)), (
                f"{mode}/{layer}: recomputed gate logit != HF's dumped bf16 logit"
            )
            gate = torch.sigmoid(logit.float()).to(torch.bfloat16)
            assert torch.equal(gate.view(torch.int16), hf_sig.view(torch.int16)), (
                f"{mode}/{layer}: bf16 sigmoid != HF's dumped bf16 sigmoid -- the "
                f"cast position is wrong"
            )
            # boundary 3: the gated shared-expert output, residual = 0
            e = frob(got, hf_gated)
            print(f"  oracle {mode}/{layer}: frob_vs_shared_output_gated={e:.3e}")
            assert e <= 2e-3, f"{mode}/{layer}: gated output frob {e:.3e}"

            # boundary 4: the residual fold this task exists for
            resid = torch.randn_like(shared)
            got_r = run(x.contiguous(), w_sg.contiguous(), shared.contiguous(), resid)
            ref_r = (resid.float() + hf_gated.float()).to(torch.bfloat16)
            e_r = frob(got_r, ref_r)
            print(f"  oracle {mode}/{layer}: frob_vs_residual_plus_gated={e_r:.3e}")
            assert e_r <= 2e-3

    print("SIGMOID_GATE_MUL_ADD UNIT TEST PASSED")


if __name__ == "__main__":
    main()
