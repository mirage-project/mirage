"""Router unit test: `topk_softmax_sm100` vs the HF oracle, boundary by boundary.

The probe (demo/qwen3_5/accept/probes/moe/p5_router_semantics.py) CHARACTERIZES
the kernel; this is the regression that keeps the characterization true. It asserts,
on the real dumps and on crafted rows:

  1. expert SET equals HF's `topk_ids` on every dumped token, including the two
     rows whose top-8 boundary is a genuine fp32 tie;
  2. renormalized weights equal HF's fp32 renormalization to fp32 rounding;
  3. with `round_weights=True` they equal HF's SHIPPED bf16 weights
     (`topk_renorm_weights`) BIT-EXACTLY -- which is what pins the cast position;
  4. the task's row capacity, so the mbt=16 build cannot regress back to the
     silent 8-row drop probe P5 found.

Run:  python tests/runtime_python/blackwell/sm100_moe_block_qwen35/test_router_oracle.py
"""

import json
import os

import torch
import torch.nn.functional as F

import runtime_kernel_blackwell_moe_block_qwen35 as mk

NUM_EXPERTS = 256
TOPK = 8
DEVICE = "cuda"
ORACLE = os.environ.get(
    "QWEN35_ORACLE_DUMPS", os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")
)


def run_router(logits, round_weights=False, vpt=0):
    rows = logits.shape[0]
    cap = mk.topk_softmax_rows_per_task(vpt)
    assert rows <= cap, f"{rows} rows exceeds the task capacity {cap}"
    g = logits.clone().contiguous()
    w = torch.zeros(rows, TOPK, dtype=torch.float32, device=DEVICE)
    routing = torch.zeros(NUM_EXPERTS, rows, dtype=torch.int32, device=DEVICE)
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    mk.topk_softmax_sm100(g, w, routing, mask, vpt, round_weights)
    torch.cuda.synchronize()
    ids = torch.full((rows, TOPK), -1, dtype=torch.int64, device=DEVICE)
    nz = routing.nonzero()
    ids[nz[:, 1], routing[nz[:, 0], nz[:, 1]].long() - 1] = nz[:, 0]
    return ids, w, mask


def load(mode, layer, key):
    man = json.load(open(os.path.join(ORACLE, mode, "manifest.json")))
    return torch.load(
        os.path.join(ORACLE, man["tensors"][f"{layer}.{key}"]["file"]),
        map_location=DEVICE,
    )


def main():
    torch.manual_seed(20260726)

    # ---- 1/2/3: every dumped MoE block, both layers, decode and prefill ----
    n_rows = n_tie = 0
    for mode in ("decode", "prefill"):
        for layer in ("moe0", "moe3"):
            logits = load(mode, layer, "router_logits")
            hf_probs = load(mode, layer, "router_probs")
            hf_ids = load(mode, layer, "topk_ids")
            hf_raw = load(mode, layer, "topk_weights_raw")
            hf_bf16 = load(mode, layer, "topk_renorm_weights")
            rows = logits.shape[0]
            n_rows += rows

            # the kernel's own softmax, recomputed the way HF does it
            probs = F.softmax(logits, dtype=torch.float32, dim=-1)
            torch.testing.assert_close(probs, hf_probs, rtol=0, atol=0)

            ids, w, mask = run_router(logits)
            for b in range(rows):
                assert set(ids[b].tolist()) == set(hf_ids[b].tolist()), (
                    f"{mode}/{layer} row {b}: expert set differs\n"
                    f"  mpk={sorted(ids[b].tolist())}\n"
                    f"  hf ={sorted(hf_ids[b].tolist())}"
                )
                srt, _ = torch.sort(hf_probs[b], descending=True)
                if srt[TOPK - 1] == srt[TOPK]:
                    n_tie += 1
            assert int(mask[NUM_EXPERTS].item()) == int(torch.unique(ids).numel())

            # weights compared in HF's own id order
            pos = (ids.unsqueeze(2) == hf_ids.unsqueeze(1)).float().argmax(dim=1)
            w_hf_order = torch.gather(w, 1, pos)
            hf_fp32 = hf_raw / hf_raw.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(w_hf_order, hf_fp32, rtol=4e-7, atol=0)

            ids_r, w_r, _ = run_router(logits, round_weights=True)
            assert torch.equal(ids_r, ids), "rounding must not change selection"
            w_r_hf_order = torch.gather(w_r, 1, pos)
            assert torch.equal(
                w_r_hf_order.to(torch.bfloat16).view(torch.int16),
                hf_bf16.view(torch.int16),
            ), (
                f"{mode}/{layer}: round_weights=True must reproduce HF's shipped "
                f"bf16 topk_renorm_weights bit-for-bit"
            )
            # and the stored fp32 must itself be an exactly-representable bf16
            assert torch.equal(
                w_r_hf_order, w_r_hf_order.to(torch.bfloat16).float()
            ), "round_weights=True must store a bf16-exact float32"
            print(
                f"  {mode}/{layer}: {rows} rows, expert sets match, "
                f"fp32 weights match, bf16 weights bit-exact"
            )
    print(f"  total token rows {n_rows}, of which {n_tie} have a top-8 boundary tie")
    assert n_tie >= 1, "the oracle is supposed to contain real boundary ties"

    # ---- 4: row capacity, the failure probe P5 found -------------------
    assert mk.topk_softmax_default_vpt() == 8
    assert mk.topk_softmax_rows_per_task(8) == 8
    assert mk.topk_softmax_rows_per_task(16) == 16
    logits16 = (torch.randn(16, NUM_EXPERTS, device=DEVICE) * 2).to(torch.bfloat16)
    ref = torch.topk(F.softmax(logits16, dtype=torch.float32, dim=-1), TOPK, dim=-1)[1]
    ids16, _, _ = run_router(logits16, vpt=16)
    for b in range(16):
        assert set(ids16[b].tolist()) == set(ref[b].tolist()), f"row {b} at VPT=16"
    # VPT=8 covers only half of them -- the reason the registration now picks VPT
    g = logits16.clone()
    w8 = torch.zeros(16, TOPK, dtype=torch.float32, device=DEVICE)
    r8 = torch.zeros(NUM_EXPERTS, 16, dtype=torch.int32, device=DEVICE)
    m8 = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    mk.topk_softmax_sm100(g, w8, r8, m8, 8, False)
    torch.cuda.synchronize()
    assert int(r8[:, 8:].sum().item()) == 0, (
        "VPT=8 is expected to leave rows 8..15 unrouted; if this fires the "
        "kernel's row capacity changed and the registration logic must follow"
    )
    print("  capacity: VPT=8 -> 8 rows (rows 8..15 unrouted), VPT=16 -> 16 rows")

    print("ROUTER ORACLE TEST PASSED")


if __name__ == "__main__":
    main()
