"""Test-mode coverage for the clamped-alpha SwiGLU and the per-expert biases
on both expert GEMMs.

Three layers in one task graph, each checked independently so a failure names
its own kernel:

  swiglu  (clamp(up) + 1) * clamp(gate) * sigmoid(gate * alpha)
  w13     [B, K] @ [E, 2I, K].T + bias[E, 2I]
  w2      [B, topk, I] @ [E, K, I].T + bias[E, K]

Biases differ per expert and per column, so bias indexing that drops the
expert fails.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_EXPERTS = 8
NUM_TOPK = 4
BATCH = 8
HIDDEN = 512
INTER = 256
LIMIT = 7.0
ALPHA = 1.702


def make_routing(device):
    """Round-robin: token i -> experts (i*topk+s) % E, slot s.

    routing[e, t] = s + 1 when token t reaches expert e in slot s, else 0.
    mask holds the activated expert ids, with the count in its last entry.
    """
    routing = torch.zeros(NUM_EXPERTS, BATCH, dtype=torch.int32, device=device)
    for t in range(BATCH):
        for s in range(NUM_TOPK):
            routing[(t * NUM_TOPK + s) % NUM_EXPERTS, t] = s + 1
    activated = [e for e in range(NUM_EXPERTS) if routing[e].any()]
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing, mask


def expert_of(routing, token, slot):
    hits = (routing[:, token] == slot + 1).nonzero().flatten().tolist()
    assert len(hits) == 1
    return hits[0]


def group_gemm_ref(x, w, b, routing):
    """x[B, (topk,) K] @ w[E, N, K].T + b[E, N] -> [B, topk, N]."""
    out = torch.zeros(BATCH, NUM_TOPK, w.shape[1], dtype=torch.float32,
                      device=x.device)
    for t in range(BATCH):
        for s in range(NUM_TOPK):
            e = expert_of(routing, t, s)
            row = x[t, s] if x.dim() == 3 else x[t]
            out[t, s] = row.float() @ w[e].float().T + b[e].float()
    return out


def swiglu_ref(mid):
    gate, up = mid[..., :INTER].float(), mid[..., INTER:].float()
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (up + 1.0) * gate * torch.sigmoid(gate * ALPHA)


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
    )
    pk = PersistentKernel(**params)

    routing, mask = make_routing(device)
    rt = pk.attach_input(routing, name="routing")
    mk = pk.attach_input(mask, name="mask")

    # --- clamped SwiGLU, twice: once below the clamp so the gating formula
    # is checked on its own, once straddling it in both directions.
    swiglu_cases = []
    for tag, scale in (("swiglu_small", 0.5), ("swiglu_clamped", 6.0)):
        mid = torch.randn(BATCH, NUM_TOPK, 2 * INTER, dtype=dtype,
                          device=device) * scale
        out = torch.zeros(BATCH, NUM_TOPK, INTER, dtype=dtype, device=device)
        pk.moe_clamped_swiglu_layer(
            input=pk.attach_input(mid, name=f"{tag}_in"),
            output=pk.attach_input(out, name=f"{tag}_out"),
            grid_dim=(BATCH, NUM_TOPK, 1), block_dim=(256, 1, 1),
            limit=LIMIT, alpha=ALPHA,
        )
        swiglu_cases.append((tag, mid, out))

    # --- w13 with a per-expert bias
    x = torch.randn(BATCH, HIDDEN, dtype=dtype, device=device) * 0.1
    w13 = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=dtype,
                      device=device) * 0.05
    b13 = torch.randn(NUM_EXPERTS, 2 * INTER, dtype=dtype, device=device)
    w13_out = torch.zeros(BATCH, NUM_TOPK, 2 * INTER, dtype=dtype, device=device)
    pk.moe_w13_linear_layer(
        input=pk.attach_input(x, name="x"),
        weight=pk.attach_input(w13, name="w13"),
        moe_routing_indices=rt, moe_mask=mk,
        output=pk.attach_input(w13_out, name="w13_out"),
        grid_dim=(10, (2 * INTER) // 128, 1), block_dim=(256, 1, 1),
        bias=pk.attach_input(b13, name="b13"),
    )

    # --- w2 with a per-expert bias
    act = torch.randn(BATCH, NUM_TOPK, INTER, dtype=dtype, device=device) * 0.1
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=dtype,
                     device=device) * 0.05
    b2 = torch.randn(NUM_EXPERTS, HIDDEN, dtype=dtype, device=device)
    w2_out = torch.zeros(BATCH, NUM_TOPK, HIDDEN, dtype=dtype, device=device)
    pk.moe_w2_linear_layer(
        input=pk.attach_input(act, name="act"),
        weight=pk.attach_input(w2, name="w2"),
        moe_routing_indices=rt, moe_mask=mk,
        output=pk.attach_input(w2_out, name="w2_out"),
        # 64 columns per task: 2880 is not divisible by 128.
        grid_dim=(8, HIDDEN // 64, 1), block_dim=(256, 1, 1),
        bias=pk.attach_input(b2, name="b2"),
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    checks = [(tag, out, swiglu_ref(mid)) for tag, mid, out in swiglu_cases]
    checks += [
        ("w13", w13_out, group_gemm_ref(x, w13, b13, routing)),
        ("w2", w2_out, group_gemm_ref(act, w2, b2, routing)),
    ]
    for name, got, ref in checks:
        diff = (got.float() - ref).abs().max().item()
        # bf16 output, so the tolerance scales with magnitude: one ulp is
        # ~0.4% of the value.
        tol = max(0.02, 0.01 * ref.abs().max().item())
        print(f"[{name}] max |kernel - reference| = {diff:.4f} "
              f"(tol {tol:.4f}, |ref|max {ref.abs().max().item():.2f})")
        if diff >= tol:
            ok = False
            print(f"[{name}] FAILED")

    # The two swiglu cases only mean what they claim if one clamps and the
    # other does not.
    for tag, mid, _ in swiglu_cases:
        frac = ((mid[..., :INTER].float() > LIMIT) |
                (mid[..., INTER:].float().abs() > LIMIT)).float().mean().item()
        print(f"[{tag}] fraction of values hitting the clamp: {frac:.3f}")
        if tag == "swiglu_clamped" and frac < 0.05:
            print("FAILED: the clamped case never reaches the clamp")
            ok = False
        if tag == "swiglu_small" and frac > 0.0:
            print("FAILED: the small case was supposed to avoid the clamp")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: clamped SwiGLU and both per-expert biases match the "
          "reference")


if __name__ == "__main__":
    main()
