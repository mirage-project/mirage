#!/usr/bin/env python3
"""M2-I9 divergence probe: which prefill row does the megakernel's first
generated token come from?

Ledger context. At bs=1 every prompt's MPK stream is
``<|im_end|> \\n <|im_start|> [assistant] \\n`` followed by the reference answer
verbatim from its position 0. So the 40-layer decode path reproduces the
reference; only the FIRST generated token -- the one produced by the prefill
pass -- is wrong. This probe reads the exposed ``argmax_in`` logits buffer after
a prefill-only run and reports the argmax of EVERY prefill row, so the two
candidate mechanisms separate cleanly:

  H-ROW   the right logits exist but the wrong row is consumed
          -> some row r < plen-1 has argmax == reference output_ids[0]
  H-MATH  the prefill pass itself computes different logits than decode
          -> no row matches, and the last row's top-1 is the emitted token

Usage:
    python probe_prefill.py --prompt-id p01-history [--rows 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ACCEPT = HERE.parent
REPO = ACCEPT.parents[2]
sys.path.insert(0, str(REPO / "python"))

import mirage as mi                                                # noqa: E402
from mirage.mpk.models.qwen3_5.builder import Qwen35Builder        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prompt-id", default="p01-history")
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--rows", type=int, default=6,
                    help="How many trailing prefill rows to report in detail.")
    ap.add_argument("--out", default=str(HERE / "probe_prefill.json"))
    ap.add_argument("--expose-all", action="store_true",
                    help="Expose EVERY op boundary and report per-row L2 norms, "
                         "so the first layer whose rows >= 16 break is visible "
                         "in one run (the M2-I9 bisection).")
    ap.add_argument("--force-prefix-pos", type=int, default=None,
                    help="Teacher-force the reference's own first N generated "
                         "tokens, then let MPK emit exactly one more, so the "
                         "exposed lm-head buffer holds MPK's REAL logit vector "
                         "at generated position N -- the per-position evidence "
                         "goal.md AC-3 requires before any tie-flip claim.")
    args = ap.parse_args()

    ref = json.load(open(ACCEPT / "reference" / "reference_outputs.json"))
    r = ref["results"][args.prompt_id]
    input_ids, out_ids = r["input_ids"], r["output_ids"]
    ref_topk_ids = ref_topk_logits = None
    if args.force_prefix_pos is not None:
        pos = args.force_prefix_pos
        ref_topk_ids = r["topk_ids_per_step"][pos]
        ref_topk_logits = r["topk_logits_per_step"][pos]
        input_ids = input_ids + out_ids[:pos]
        out_ids = out_ids[pos:]
    plen = len(input_ids)
    # Prefill + exactly one decode step.
    max_seq_length = plen + 1
    mbr = 1
    pages = max(-(-max_seq_length // args.page_size) + 4, 8)

    dev = "cuda"
    meta = {
        "step": torch.zeros(mbr, dtype=torch.int32, device=dev),
        "tokens": torch.zeros((mbr, max_seq_length), dtype=torch.long, device=dev),
        "input_tokens": torch.zeros((args.mbt, 1), dtype=torch.long, device=dev),
        "output_tokens": torch.zeros((args.mbt, 1), dtype=torch.long, device=dev),
        "num_new_tokens": torch.ones(mbr, dtype=torch.int32, device=dev),
        "prompt_lengths": torch.full((mbr,), plen, dtype=torch.int32, device=dev),
        "qo_indptr_buffer": torch.zeros(mbr + 1, dtype=torch.int32, device=dev),
        "paged_kv_indptr_buffer": torch.zeros(mbr + 1, dtype=torch.int32, device=dev),
        "paged_kv_indices_buffer": torch.zeros(pages, dtype=torch.int32, device=dev),
        "paged_kv_last_page_len_buffer": torch.zeros(mbr, dtype=torch.int32, device=dev),
        "paged_kv_indices_snapshot": torch.zeros(pages, dtype=torch.int32, device=dev),
    }
    meta["tokens"][0, :plen] = torch.tensor(input_ids, dtype=torch.long, device=dev)

    nw, ns = mi.get_configurations_from_gpu(0)
    torch.set_default_dtype(torch.bfloat16)
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0, num_workers=nw,
        num_local_schedulers=ns, num_remote_schedulers=0,
        max_seq_length=max_seq_length, max_num_batched_requests=mbr,
        max_num_batched_tokens=args.mbt, max_num_pages=pages,
        page_size=args.page_size, eos_token_id=-1, meta_tensors=meta,
        profiler_tensor=None, trace_name="", spec_decode_config=None,
        use_cutlass_kernel=True)
    builder = Qwen35Builder(mpk)
    if args.expose_all:
        builder.expose_intermediates = True
    else:
        builder.expose_logits = True
    builder.build_from_model(model_name=args.model, model_path=args.model_path)
    mpk.compile(output_dir=str(HERE / "probe_kernel"))
    mpk()
    torch.cuda.synchronize()

    logits = builder.buffers["argmax_in"].float()
    emitted = int(meta["tokens"][0, plen].item())
    want = out_ids[0]

    # `argmax_in` is indexed by the row's position WITHIN THE CURRENT CHUNK, not
    # by absolute sequence position: the runtime feeds ceil(plen / mbt) prefill
    # chunks and the buffer only ever holds the last one. So the final prompt
    # token lives at (plen - 1) % mbt, and only that many rows are meaningful.
    last_row = (plen - 1) % args.mbt
    n_rows = min(plen, last_row + 1)
    rows = []
    hit_row = None
    for i in range(n_rows):
        v, idx = torch.max(logits[i], dim=-1)
        top = int(idx.item())
        rows.append({"row": i, "argmax": top, "logit": float(v.item()),
                     "is_reference_first_token": top == want})
        if top == want and hit_row is None:
            hit_row = i

    report = {
        "prompt_id": args.prompt_id, "prompt_len": plen,
        "reference_output_0": want, "mpk_emitted_token": emitted,
        "final_step": int(meta["step"][0].item()),
        "row_with_reference_first_token": hit_row,
        "last_rows": rows[-args.rows:],
        "all_rows_argmax": [x["argmax"] for x in rows],
        "hypothesis": ("H-ROW: correct logits exist at row "
                       f"{hit_row} but row {plen - 1} was consumed"
                       if hit_row is not None and hit_row != plen - 1 else
                       ("H-MATH: no prefill row predicts the reference token"
                        if hit_row is None else
                        "prefill row plen-1 is correct; look downstream")),
    }
    if ref_topk_ids is not None:
        # Side-by-side at the SAME candidate ids. Both engines argmax over bf16
        # logits (every stored reference logit is exactly bf16-representable, so
        # generate_reference.py's `.float()` is an exact upcast that cannot move
        # a comparison), which makes these two vectors directly comparable and
        # the gap expressible in bf16 ULPs.
        last = logits[last_row]
        ulp = float(torch.tensor(ref_topk_logits[0], dtype=torch.bfloat16)
                    .float().item())
        ulp = 2.0 ** (torch.tensor(abs(ulp)).log2().floor().item() - 7)
        cmp = []
        for cid, clog in zip(ref_topk_ids, ref_topk_logits):
            m = float(last[cid].item())
            cmp.append({"token_id": cid, "ref_logit": clog, "mpk_logit": m,
                        "delta": m - clog, "delta_ulps": (m - clog) / ulp})
        mv, mi_ = torch.max(last, dim=-1)
        report["tie_evidence"] = {
            "position": args.force_prefix_pos,
            "bf16_ulp_at_this_magnitude": ulp,
            "ref_top1": ref_topk_ids[0], "mpk_argmax": int(mi_.item()),
            "mpk_argmax_logit": float(mv.item()),
            "mpk_argmax_in_ref_topk": int(mi_.item()) in ref_topk_ids,
            "ref_margin_top1_top2": ref_topk_logits[0] - ref_topk_logits[1],
            "mpk_margin_over_ref_top1":
                float(mv.item()) - float(last[ref_topk_ids[0]].item()),
            "candidates": cmp,
        }
        print(json.dumps(report["tie_evidence"], indent=2))
    if args.expose_all:
        # Per-row L2 norm of every exposed boundary. A boundary whose rows >= 16
        # are the first to lose structure is the culprit; everything upstream of
        # it is intact.
        prof = {}
        for name, buf in builder.buffers.items():
            if buf.dim() < 2 or buf.shape[0] < plen:
                continue
            x = buf[:plen].reshape(plen, -1).float()
            nrm = x.norm(dim=-1)
            lo = float(nrm[:16].mean().item())
            hi = float(nrm[16:].mean().item()) if plen > 16 else None
            prof[name] = {
                "mean_norm_rows_0_15": lo,
                "mean_norm_rows_16plus": hi,
                "ratio": (hi / lo) if (hi is not None and lo) else None,
                "n_zero_rows_16plus": int((nrm[16:] == 0).sum().item()),
                "per_row_norm": [round(float(v), 4) for v in nrm.tolist()],
            }
        report["boundary_profile"] = prof
        flagged = sorted(
            ((k, v) for k, v in prof.items() if v["ratio"] is not None
             and (v["ratio"] < 0.5 or v["ratio"] > 2.0 or v["n_zero_rows_16plus"])),
            key=lambda kv: kv[0])
        report["flagged_boundaries"] = [
            {"name": k, "ratio": v["ratio"],
             "zero_rows_16plus": v["n_zero_rows_16plus"]} for k, v in flagged]
        print("FLAGGED BOUNDARIES (rows>=16 vs rows<16):")
        for f in report["flagged_boundaries"][:40]:
            print(" ", f)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("all_rows_argmax", "boundary_profile",
                                   "tie_evidence")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
