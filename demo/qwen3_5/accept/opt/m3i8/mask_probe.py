#!/usr/bin/env python3
"""Read the MoE router's ACTIVATED-EXPERT MASK straight out of the megakernel.

This is M3-I8's primary mechanism instrument, and the one artifact that can
falsify the whole issue in a single number. M3-I1 inferred the activated-group
count indirectly, from how many grouped-GEMM tasks ran longer than 1 us. This
probe reads the count the router itself wrote:

    layer_i_moe_mask[num_experts]  == num_activated for the LAST iteration
    layer_i_routing[e, row]        == topk slot + 1, 0 when not routed

`builder.expose_intermediates` makes every intermediate a host-visible torch
tensor, so no kernel change is needed to read them.

All `bs` requests are given prompts TRUNCATED TO A COMMON LENGTH: same length
keeps every request in lockstep (so the final iteration is a clean decode with
exactly `bs` live rows), different content keeps their routing genuinely
different (so the measured union is a real one, not `bs` copies of one row's
top-8).

Expected, per layer, at bs 1/2/4/8/16:
    MOE_GATE_PADDING_ROWS=False   56.4 / 59.4 / 60.2 / 70.1 / 86.7  (M3-I1)
    MOE_GATE_PADDING_ROWS=True     8.0 / ~14.7 / ~24.6 / ~47.9 / 86.7
and, decisively, `<= min(256, 8*bs)` in the gated arm -- a hard cap, not a fit.

Usage (on the B200, under the GPU guard, one wave, no profiler):
    python3 mask_probe.py --batch-size 1 --out mask_bs1_v1.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ACCEPT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ACCEPT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--kernel-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import mirage as mi
    from mirage.mpk.models.qwen3_5.builder import Qwen35Builder, \
        MOE_GATE_PADDING_ROWS

    ref = json.load(open(ACCEPT / "reference" / "reference_outputs.json"))
    ids = sorted(ref["results"])[:a.batch_size]
    prompts = [ref["results"][i]["input_ids"] for i in ids]
    plen = min(len(p) for p in prompts)
    prompts = [p[:plen] for p in prompts]          # common length, distinct text

    bs = a.batch_size
    max_seq_length = plen + a.new_tokens + 1
    pages = max(bs * (-(-max_seq_length // a.page_size)) + 4, 8)
    dev = "cuda"
    meta = {
        "step": torch.zeros(bs, dtype=torch.int32, device=dev),
        "tokens": torch.zeros((bs, max_seq_length), dtype=torch.long, device=dev),
        "input_tokens": torch.zeros((a.mbt, 1), dtype=torch.long, device=dev),
        "output_tokens": torch.zeros((a.mbt, 1), dtype=torch.long, device=dev),
        "num_new_tokens": torch.ones(bs, dtype=torch.int32, device=dev),
        "prompt_lengths": torch.full((bs,), plen, dtype=torch.int32, device=dev),
        "qo_indptr_buffer": torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        "paged_kv_indptr_buffer": torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        "paged_kv_indices_buffer": torch.zeros(pages, dtype=torch.int32, device=dev),
        "paged_kv_last_page_len_buffer": torch.zeros(bs, dtype=torch.int32, device=dev),
        "paged_kv_indices_snapshot": torch.zeros(pages, dtype=torch.int32, device=dev),
    }
    for i, p in enumerate(prompts):
        meta["tokens"][i, :plen] = torch.tensor(p, dtype=torch.long, device=dev)

    nw, ns = mi.get_configurations_from_gpu(0)
    torch.set_default_dtype(torch.bfloat16)
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0, num_workers=nw,
        num_local_schedulers=ns, num_remote_schedulers=0,
        max_seq_length=max_seq_length, max_num_batched_requests=bs,
        max_num_batched_tokens=a.mbt, max_num_pages=pages,
        page_size=a.page_size, eos_token_id=-1, meta_tensors=meta,
        profiler_tensor=None, trace_name="", spec_decode_config=None,
        use_cutlass_kernel=True)
    builder = Qwen35Builder(mpk)
    builder.expose_intermediates = True
    builder.build_from_model(model_name=a.model, model_path=a.model_path)
    mpk.compile(output_dir=a.kernel_dir)
    mpk()
    torch.cuda.synchronize()

    cfg = builder.config
    n_exp, topk = cfg.num_experts, cfg.num_experts_per_tok
    live = int(meta["qo_indptr_buffer"][bs].item())
    per_layer = []
    for i in range(cfg.num_layers):
        mask = builder.buffers[f"layer_{i}_moe_mask"].to("cpu").tolist()
        routing = builder.buffers[f"layer_{i}_routing"].to("cpu")
        activated = int(mask[n_exp])
        per_row = [int((routing[:, r] > 0).sum().item()) for r in range(a.mbt)]
        per_layer.append(dict(layer=i, activated=activated,
                              experts_per_row=per_row))
    acts = [p["activated"] for p in per_layer]
    cap = min(n_exp, topk * live)
    rows_marked = [sum(1 for r in range(a.mbt) if p["experts_per_row"][r] > 0)
                   for p in per_layer]

    report = dict(
        batch_size=bs, mbt=a.mbt, prompt_ids=ids, common_prompt_len=plen,
        live_rows_last_iter=live, num_experts=n_exp, topk=topk,
        gate_padding_rows=bool(MOE_GATE_PADDING_ROWS),
        activated_mean=sum(acts) / len(acts),
        activated_min=min(acts), activated_max=max(acts),
        hard_cap=cap,
        cap_respected=max(acts) <= cap,
        rows_marked_mean=sum(rows_marked) / len(rows_marked),
        expected_rows_marked=live if MOE_GATE_PADDING_ROWS else a.mbt,
        per_layer=per_layer)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"},
                     indent=1))
    # A gated build whose mask still exceeds the cap means the runtime scalar
    # never reached the kernel -- the lever is void until that is explained.
    if MOE_GATE_PADDING_ROWS and not report["cap_respected"]:
        print("FALSIFIED: gated build activated more than min(256, topk*live)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
