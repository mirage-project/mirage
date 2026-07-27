"""Router unit test: `topk_softmax_sm100` vs the HF oracle, boundary by boundary.

The probe (demo/qwen3_5/accept/probes/moe/p5_router_semantics.py) CHARACTERIZES
the kernel; this is the regression that keeps the characterization true. It asserts,
on the real dumps and on crafted rows:

  1. expert SET equals HF's `topk_ids` on every dumped token, including the two
     rows whose top-8 boundary is a genuine fp32 tie;
  2. renormalized weights equal HF's fp32 renormalization to fp32 rounding;
  3. with `round_weights=True` they equal HF's SHIPPED bf16 weights
     (`topk_renorm_weights`) BIT-EXACTLY -- which is what pins the cast position;
  4. the task's row coverage, so the build cannot regress back to the silent
     row drop probe P5 found. Since M3-I5b the kernel loops over row tiles, so
     the assertion is the opposite of what it was: EVERY row must be routed at
     EVERY legal VPT, including well past one tile.

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

    # ---- 4: row coverage, the failure probe P5 found -------------------
    # Rows-per-PASS is unchanged (8 at VPT=8, 16 at VPT=16); what changed in
    # M3-I5b is that the kernel now repeats the pass, so neither is a cap.
    assert mk.topk_softmax_default_vpt() == 8
    assert mk.topk_softmax_rows_per_task(8) == 8
    assert mk.topk_softmax_rows_per_task(16) == 16

    # 16 rows must be routed identically at BOTH VPTs -- one pass at VPT=16,
    # two row tiles at VPT=8. Before M3-I5b the VPT=8 arm left rows 8..15 at
    # zero, which is exactly what this used to assert.
    logits16 = (torch.randn(16, NUM_EXPERTS, device=DEVICE) * 2).to(torch.bfloat16)
    ref = torch.topk(F.softmax(logits16, dtype=torch.float32, dim=-1), TOPK, dim=-1)[1]
    ids16_v16, w16_v16, _ = run_router(logits16, vpt=16)
    ids16_v8, w16_v8, _ = run_router(logits16, vpt=8)
    for b in range(16):
        assert set(ids16_v16[b].tolist()) == set(ref[b].tolist()), f"row {b} at VPT=16"
        assert set(ids16_v8[b].tolist()) == set(ref[b].tolist()), (
            f"row {b} at VPT=8: the row-tile loop must route rows past the "
            f"first 8-row pass (M3-I5b; this is the M2-I9 regression)"
        )
    assert torch.equal(w16_v8, w16_v16), (
        "the two VPTs must agree bit-for-bit on the same rows"
    )

    # Rows that are NOT a whole number of tiles, at both sub-group widths, so
    # the partial-warp shuffle mask is exercised in a LATER tile too.
    for rows in (1, 7, 9, 17, 33):
        lg = (torch.randn(rows, NUM_EXPERTS, device=DEVICE) * 2).to(torch.bfloat16)
        rf = torch.topk(F.softmax(lg, dtype=torch.float32, dim=-1), TOPK, dim=-1)[1]
        for vpt in (8, 16):
            ids_n, _, mask_n = run_router(lg, vpt=vpt)
            for b in range(rows):
                assert set(ids_n[b].tolist()) == set(rf[b].tolist()), (
                    f"{rows} rows at VPT={vpt}: row {b} differs"
                )
            assert int(mask_n[NUM_EXPERTS].item()) == int(torch.unique(ids_n).numel())
    print("  coverage: 1/7/9/16/17/33 rows all routed at VPT=8 and VPT=16")

    print("ROUTER ORACLE TEST PASSED")


if __name__ == "__main__":
    main()
