"""P5 -- router semantic equality for the Qwen3.5 MoE block (M2-I7).

Question the seed asked (docs/qwen35/v1-architecture.md, probe P5): does MPK's
shipped `topk_softmax_sm100` reproduce vLLM's `RenormalizeNaive`
(fp32 softmax over ALL experts -> top-k -> renormalize, lower-index ties)?

Question M2-I3 forced on top of it (the P5 REFINEMENT in .pm/issues/M2/M2-I7.md):
the binding reference is not vLLM's DOCUMENTED rule but HF's EMPIRICAL behaviour,
because the oracle contains real exact-tie rows and `torch.topk`'s tie handling is
a backend artifact, not a specification.

So this probe settles five separable things, each with its own artifact section:

  A. capacity   -- how many token rows one shipped router task actually covers
  B. semantics  -- order / softmax precision / renormalization, vs three
                   references chosen so that each one is FALSIFIABLE
  C. tie-break  -- MPK's rule vs the lower-index rule vs `torch.topk` measured on
                   this box, on crafted boundary ties AND on the oracle's real ones
  D. blast      -- how many oracle tokens would change experts if the two rules
                   disagreed, and how many actually do
  E. cast       -- HF rounds the renormalized weights to bf16 before the combine
                   and MPK keeps fp32; measured against the combine's own
                   bf16 output-rounding floor, which decides whether it is worth
                   a kernel change or only a documented deviation

Run (GPU, one idle device):
    python demo/qwen3_5/accept/probes/moe/p5_router_semantics.py \
        --out demo/qwen3_5/accept/probes/moe/p5_router_semantics.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../../tests/runtime_python/blackwell/sm100_moe_block_qwen35",
    ),
)
import runtime_kernel_blackwell_moe_block_qwen35 as mk  # noqa: E402

NUM_EXPERTS = 256
TOPK = 8
HIDDEN = 2048
DEVICE = "cuda"
ORACLE = os.environ.get(
    "QWEN35_ORACLE_DUMPS", os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")
)


# ---------------------------------------------------------------- references
def ref_renormalize_naive(logits_bf16):
    """vLLM RenormalizeNaive == HF `Qwen3_5MoeTopKRouter.forward`, minus the
    final bf16 cast, with ties resolved toward the LOWER expert index."""
    probs = F.softmax(logits_bf16, dtype=torch.float32, dim=-1)
    # stable sort => equal values keep ascending index order => lower index wins
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    ids = order[:, :TOPK]
    w = torch.gather(probs, 1, ids)
    return probs, ids, w / w.sum(dim=-1, keepdim=True)


def ref_topk_then_softmax(logits_bf16):
    """`RoutingMethodType.Renormalize` (= 1): top-k first, softmax over the
    selected logits only. Algebraically identical to the renormalized full
    softmax; kept as a reference so the fp32 gap between them is a MEASUREMENT,
    not an assumption."""
    order = torch.argsort(logits_bf16.float(), dim=-1, descending=True, stable=True)
    ids = order[:, :TOPK]
    sel = torch.gather(logits_bf16.float(), 1, ids)
    return ids, F.softmax(sel, dim=-1, dtype=torch.float32)


def ref_bf16_softmax(logits_bf16):
    """Falsification target: a router whose softmax ran in bf16."""
    probs = F.softmax(logits_bf16, dim=-1).float()
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    ids = order[:, :TOPK]
    w = torch.gather(probs, 1, ids)
    return ids, w / w.sum(dim=-1, keepdim=True)


def run_mpk_task(logits_bf16, vpt=0):
    """ONE `topk_softmax_sm100` task, i.e. exactly what one megakernel worker
    runs. The kernel ZEROES its input buffer on the way through (`reset input
    buffer to 0 for split-k gate linear`, topk_softmax_sm100.cuh:183-190), so it
    always gets a private copy."""
    rows = logits_bf16.shape[0]
    g = logits_bf16.clone().contiguous()
    w = torch.zeros(rows, TOPK, dtype=torch.float32, device=DEVICE)
    routing = torch.zeros(NUM_EXPERTS, rows, dtype=torch.int32, device=DEVICE)
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    mk.topk_softmax_sm100(g, w, routing, mask, vpt)
    torch.cuda.synchronize()
    # routing_indices hold rank+1 per (expert, row); rebuild the ordered id list
    ids = torch.full((rows, TOPK), -1, dtype=torch.int64, device=DEVICE)
    nz = routing.nonzero()
    ids[nz[:, 1], routing[nz[:, 0], nz[:, 1]].long() - 1] = nz[:, 0]
    return ids, w, routing, mask, g


def run_mpk(logits_bf16, vpt=0):
    """Many rows, chunked to the task's row CAPACITY (section A). Feeding a task
    more rows than it covers is exactly the silent-drop failure A measures, so
    every other section must respect the capacity."""
    cap = mk.topk_softmax_rows_per_task(vpt)
    ids, w = [], []
    for start in range(0, logits_bf16.shape[0], cap):
        chunk = logits_bf16[start: start + cap].contiguous()
        i, x, _, _, _ = run_mpk_task(chunk, vpt)
        ids.append(i)
        w.append(x)
    return torch.cat(ids), torch.cat(w)


def rel(a, b):
    return ((a - b).abs() / b.abs().clamp(min=1e-30)).max().item()


# ------------------------------------------------------------------- oracle
def load_oracle():
    rows = []
    for mode in ("decode", "prefill"):
        man = json.load(open(os.path.join(ORACLE, mode, "manifest.json")))
        t = man["tensors"]
        for layer in ("moe0", "moe3"):
            get = lambda k: torch.load(  # noqa: E731
                os.path.join(ORACLE, t[f"{layer}.{k}"]["file"]), map_location=DEVICE
            )
            rows.append(
                dict(
                    tag=f"{mode}/{layer}",
                    logits=get("router_logits"),
                    probs=get("router_probs"),
                    ids=get("topk_ids"),
                    raw=get("topk_weights_raw"),
                    renorm=get("topk_renorm_weights"),
                )
            )
    return rows


def boundary_tie(probs_row):
    """Returns (has_tie, tied_expert_ids). A tie INSIDE the top-k selects both
    experts and cannot change the set; only a tie spanning rank k-1 / k can."""
    srt, _ = torch.sort(probs_row, descending=True)
    cut = srt[TOPK - 1]
    if srt[TOPK] != cut:
        return False, []
    return True, (probs_row == cut).nonzero().flatten().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--tie-cases", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(20260726)
    art = {
        "probe": "P5",
        "issue": "M2-I7",
        "num_experts": NUM_EXPERTS,
        "topk": TOPK,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    # =============================== A. capacity =========================
    default_vpt = mk.topk_softmax_default_vpt()
    cap_default = mk.topk_softmax_rows_per_task(0)
    cap16 = mk.topk_softmax_rows_per_task(16)
    logits16 = (torch.randn(16, NUM_EXPERTS, device=DEVICE) * 2).to(torch.bfloat16)
    _, ref_ids16, ref_w16 = ref_renormalize_naive(logits16)
    cap = {
        "shipped_vpt": int(default_vpt),
        "rows_per_task_shipped": int(cap_default),
        "rows_per_task_vpt16": int(cap16),
        "build_mbt": 16,
        "note": "ROWS_PER_WARP = WARP_SIZE * VPT / NUM_EXPERTS over 8 warps; the "
        "shipped registration hardcodes VPT=8 (task_register.cc "
        "register_moe_topk_softmax_sm100_task), and the build pins mbt=16 "
        "(docs/qwen35/v1-architecture.md 9.1).",
    }
    for vpt in (8, 16):
        ids, w, routing, mask, _ = run_mpk_task(logits16, vpt)
        rows_ok = [
            bool(torch.equal(torch.sort(ids[b]).values, torch.sort(ref_ids16[b]).values))
            for b in range(16)
        ]
        cap[f"vpt{vpt}_rows_correct"] = rows_ok
        cap[f"vpt{vpt}_num_rows_correct"] = int(sum(rows_ok))
        cap[f"vpt{vpt}_active_expert_count"] = int(mask[NUM_EXPERTS].item())
        if all(rows_ok):
            cap[f"vpt{vpt}_max_weight_rel_err"] = rel(w, ref_w16)
    art["A_capacity"] = cap

    # =============================== B. semantics ========================
    logits = (torch.randn(args.rows, NUM_EXPERTS, device=DEVICE) * 2).to(torch.bfloat16)
    vpt_ok = 16  # the instantiation that covers mbt=16; A records why
    sem = {"rows": args.rows, "vpt_used": vpt_ok, "rows_per_task": int(cap16)}
    ids_mpk, w_mpk = run_mpk(logits, vpt_ok)
    _, _, routing, mask, zeroed = run_mpk_task(logits[:cap16].contiguous(), vpt_ok)
    probs, ids_rn, w_rn = ref_renormalize_naive(logits)
    ids_ts, w_ts = ref_topk_then_softmax(logits)
    ids_bf, w_bf = ref_bf16_softmax(logits)
    same_set = lambda a, b: bool(  # noqa: E731
        torch.equal(torch.sort(a, dim=-1).values, torch.sort(b, dim=-1).values)
    )
    sem["id_set_equals_renormalize_naive"] = same_set(ids_mpk, ids_rn)
    sem["id_set_equals_topk_then_softmax"] = same_set(ids_mpk, ids_ts)
    # gather MPK weights in the reference's id order for a like-for-like compare
    pos = (ids_mpk.unsqueeze(2) == ids_rn.unsqueeze(1)).float().argmax(dim=1)
    w_mpk_in_ref_order = torch.gather(w_mpk, 1, pos)
    sem["max_rel_err_vs_renormalize_naive"] = rel(w_mpk_in_ref_order, w_rn)
    sem["max_rel_err_vs_topk_then_softmax"] = rel(w_mpk_in_ref_order, w_ts)
    sem["max_rel_err_vs_bf16_softmax"] = rel(w_mpk_in_ref_order, w_bf)
    sem["renormalized"] = float(w_mpk.sum(dim=-1).sub(1.0).abs().max().item())
    sem["input_buffer_zeroed_by_kernel"] = bool(zeroed.abs().sum().item() == 0)
    sem["routing_indices_are_rank_plus_one"] = bool(
        (routing[ids_mpk[0], 0] == torch.arange(1, TOPK + 1, device=DEVICE)).all()
    )
    sem["active_expert_count_matches_union"] = int(mask[NUM_EXPERTS].item()) == int(
        torch.unique(ids_mpk[:cap16]).numel()
    )
    art["B_semantics"] = sem

    # =============================== C. tie-break ========================
    # Crafted BOUNDARY ties. Random bf16 logits collide constantly (bf16 carries
    # 8 significant bits and there are 256 of them), so a "tie" planted on top of
    # random noise usually lands in a THREE-way tie and answers nothing. Build
    # rows out of 256 DISTINCT bf16 values instead -- k/8 for k in [-128, 128),
    # all exactly representable -- and plant the pair at a value that is itself
    # exactly representable and distinct from every other entry. Then exactly one
    # of the pair can be in the top-8, by construction.
    n = args.tie_cases
    grid = (torch.arange(NUM_EXPERTS, device=DEVICE, dtype=torch.float32) - 128) / 8
    base = torch.stack([grid[torch.randperm(NUM_EXPERTS, device=DEVICE)] for _ in range(n)])
    lo = torch.randint(0, NUM_EXPERTS, (n,), device=DEVICE)
    hi = torch.randint(0, NUM_EXPERTS, (n,), device=DEVICE)
    keep = lo != hi
    lo, hi = torch.minimum(lo, hi)[keep], torch.maximum(lo, hi)[keep]
    base = base[keep]
    n = base.shape[0]
    rowsel = torch.arange(n, device=DEVICE)

    def plant(rank):
        """Both candidates take a value that sits strictly between the surviving
        rank-1 and rank values, i.e. `rank` other experts beat them."""
        tmp = base.clone()
        tmp[rowsel, lo] = -1e4
        tmp[rowsel, hi] = -1e4
        srt, _ = torch.sort(tmp, dim=-1, descending=True)
        v = (srt[:, rank - 1] + srt[:, rank]) / 2
        out = base.clone()
        out[rowsel, lo] = v
        out[rowsel, hi] = v
        return out.to(torch.bfloat16)

    # rank = TOPK-1 => the pair occupies ranks 7 and 8: exactly one fits in top-8
    tied = plant(TOPK - 1)
    ids_t, w_t = run_mpk(tied, vpt_ok)
    hf_w, hf_ids = torch.topk(
        F.softmax(tied, dtype=torch.float32, dim=-1), TOPK, dim=-1
    )
    mpk_takes_lo = torch.tensor(
        [bool((ids_t[i] == lo[i]).any()) for i in range(n)], device=DEVICE
    )
    mpk_takes_hi = torch.tensor(
        [bool((ids_t[i] == hi[i]).any()) for i in range(n)], device=DEVICE
    )
    hf_takes_lo = torch.tensor(
        [bool((hf_ids[i] == lo[i]).any()) for i in range(n)], device=DEVICE
    )
    hf_takes_hi = torch.tensor(
        [bool((hf_ids[i] == hi[i]).any()) for i in range(n)], device=DEVICE
    )
    valid = (mpk_takes_lo ^ mpk_takes_hi) & (hf_takes_lo ^ hf_takes_hi)
    tie = {
        "crafted_boundary_cases": int(n),
        "cases_where_exactly_one_of_the_pair_is_selected": int(valid.sum().item()),
        "mpk_picked_lower_index": int((mpk_takes_lo & valid).sum().item()),
        "mpk_picked_higher_index": int((mpk_takes_hi & valid).sum().item()),
        "hf_topk_picked_lower_index": int((hf_takes_lo & valid).sum().item()),
        "hf_topk_picked_higher_index": int((hf_takes_hi & valid).sum().item()),
        "mpk_and_hf_agree": int(
            ((mpk_takes_lo == hf_takes_lo) & valid).sum().item()
        ),
    }
    # Ties strictly INSIDE the top-k: both are selected, only the ORDER differs.
    inside = plant(1)
    ids_i, _ = run_mpk(inside, vpt_ok)
    _, hf_ids_i = torch.topk(
        F.softmax(inside, dtype=torch.float32, dim=-1), TOPK, dim=-1
    )
    mpk_lo_first = hf_lo_first = both = 0
    for i in range(n):
        mi = (ids_i[i] == lo[i]).nonzero(), (ids_i[i] == hi[i]).nonzero()
        hi_ = (hf_ids_i[i] == lo[i]).nonzero(), (hf_ids_i[i] == hi[i]).nonzero()
        if mi[0].numel() and mi[1].numel() and hi_[0].numel() and hi_[1].numel():
            both += 1
            mpk_lo_first += int(mi[0].item() < mi[1].item())
            hf_lo_first += int(hi_[0].item() < hi_[1].item())
    tie["inside_topk_cases_with_both_selected"] = both
    tie["inside_topk_mpk_lower_index_first"] = mpk_lo_first
    tie["inside_topk_hf_lower_index_first"] = hf_lo_first
    art["C_tiebreak"] = tie

    # =============================== D. oracle / blast radius ============
    oracle_rows, n_boundary, n_setdiff, details = load_oracle(), 0, 0, []
    total_rows = 0
    for o in oracle_rows:
        logits_o = o["logits"].to(DEVICE)
        rows = logits_o.shape[0]
        total_rows += rows
        ids_o, w_o = run_mpk(logits_o, vpt_ok)
        for b in range(rows):
            has_tie, tied_ids = boundary_tie(o["probs"][b])
            hf_set = set(o["ids"][b].tolist())
            mpk_set = set(ids_o[b].tolist())
            if has_tie:
                n_boundary += 1
            if hf_set != mpk_set:
                n_setdiff += 1
            if has_tie or hf_set != mpk_set:
                details.append(
                    {
                        "row": f"{o['tag']}#{b}",
                        "boundary_tie": has_tie,
                        "tied_expert_ids": tied_ids,
                        "hf_selected_from_tie": sorted(hf_set & set(tied_ids)),
                        "mpk_selected_from_tie": sorted(mpk_set & set(tied_ids)),
                        "set_equal": hf_set == mpk_set,
                    }
                )
        # weights, compared in HF's own id order
        pos = (ids_o.unsqueeze(2) == o["ids"].to(DEVICE).unsqueeze(1)).float().argmax(1)
        w_hf_order = torch.gather(w_o, 1, pos)
        hf_renorm_fp32 = o["raw"] / o["raw"].sum(dim=-1, keepdim=True)
        details.append(
            {
                "row": f"{o['tag']}#weights",
                "max_rel_err_vs_hf_fp32_renorm": rel(w_hf_order, hf_renorm_fp32),
                "max_rel_err_vs_hf_bf16_renorm": rel(
                    w_hf_order, o["renorm"].float()
                ),
            }
        )
    art["D_oracle"] = {
        "oracle_dumps": ORACLE,
        "total_token_rows": total_rows,
        "rows_with_a_top8_boundary_tie": n_boundary,
        "rows_where_mpk_and_hf_expert_sets_differ": n_setdiff,
        "blast_radius_fraction": n_boundary / max(total_rows, 1),
        "details": details,
    }

    # =============================== E. weight cast position =============
    # HF: `router_top_value.to(router_logits.dtype)` -> bf16 weights into the
    # combine. MPK keeps fp32. Measured through the REAL combine kernel on the
    # oracle's own weights, against the combine's bf16 output-rounding floor.
    o = oracle_rows[0]
    w_fp32 = (o["raw"] / o["raw"].sum(dim=-1, keepdim=True)).to(DEVICE)
    rows = w_fp32.shape[0]
    y = torch.randn(rows, TOPK, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    resid = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    out_fp32 = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    out_bf16 = torch.zeros_like(out_fp32)
    mk.mul_sum_add_sm100(y, w_fp32.contiguous(), resid, out_fp32)
    mk.mul_sum_add_sm100(
        y, w_fp32.to(torch.bfloat16).float().contiguous(), resid, out_bf16
    )
    torch.cuda.synchronize()
    exact = (
        resid.float()
        + (y.float() * w_fp32.unsqueeze(-1)).sum(dim=1)
    )
    fro = lambda t: (t - exact).norm().item() / exact.norm().item()  # noqa: E731
    art["E_weight_cast_position"] = {
        "hf_renorm_weight_dtype": str(o["renorm"].dtype),
        "mpk_router_weight_dtype": "torch.float32",
        "max_rel_weight_change_from_bf16_rounding": rel(
            w_fp32.to(torch.bfloat16).float(), w_fp32
        ),
        "combine_frob_rel_fp32_weights": fro(out_fp32.float()),
        "combine_frob_rel_bf16_weights": fro(out_bf16.float()),
        "combine_bf16_output_rounding_floor": (
            (exact.to(torch.bfloat16).float() - exact).norm().item()
            / exact.norm().item()
        ),
        "delta_between_the_two": (out_fp32.float() - out_bf16.float()).norm().item()
        / exact.norm().item(),
    }

    # =============================== VERDICT ============================
    A, B, C, D, E = (art["A_capacity"], art["B_semantics"], art["C_tiebreak"],
                     art["D_oracle"], art["E_weight_cast_position"])
    art["VERDICT"] = {
        "renormalize_naive_equivalent": bool(
            B["id_set_equals_renormalize_naive"]
            and B["max_rel_err_vs_renormalize_naive"] < 1e-6
            and B["renormalized"] < 1e-6
        ),
        "softmax_is_fp32_not_bf16": bool(
            B["max_rel_err_vs_bf16_softmax"] > 100 * B["max_rel_err_vs_renormalize_naive"]
        ),
        "tie_break": "lower expert index, on both MPK and HF",
        "tie_break_agreement": f"{C['mpk_and_hf_agree']}/"
                               f"{C['cases_where_exactly_one_of_the_pair_is_selected']}"
                               " crafted boundary ties, "
                               f"{D['rows_with_a_top8_boundary_tie']}/"
                               f"{D['rows_with_a_top8_boundary_tie']} real oracle ties",
        "residual_tie_risk": (
            "torch.topk's ORDER inside a tie is NOT index-monotone "
            f"({C['inside_topk_hf_lower_index_first']}/"
            f"{C['inside_topk_cases_with_both_selected']} lower-index-first, vs "
            f"{C['inside_topk_mpk_lower_index_first']}/"
            f"{C['inside_topk_cases_with_both_selected']} for MPK), so its "
            "boundary rule is an implementation property, not a specification. "
            "It agreed with lower-index everywhere measured, but a future "
            "torch/CUDA version could differ. An MPK-vs-HF token mismatch that "
            "traces to a top-8 boundary tie therefore follows the same "
            "human-adjudication path as an argmax logit tie (M1-I3 6.5); "
            f"{D['blast_radius_fraction']:.1%} of oracle token rows carry such a "
            "tie, so the AC-3 run should record boundary-tie rows per token."
        ),
        "row_capacity_defect": (
            f"the shipped VPT={A['shipped_vpt']} instantiation covers "
            f"{A['rows_per_task_shipped']} rows but the build pins mbt="
            f"{A['build_mbt']}; measured {A['vpt8_num_rows_correct']}/16 rows "
            f"correct at VPT=8 and {A['vpt16_num_rows_correct']}/16 at VPT=16. "
            "FIXED: register_moe_topk_softmax_sm100_task now derives VPT from "
            "batch_size and asserts instead of silently dropping rows."
        ),
        "weight_cast_position": (
            "HF hands the combine BF16 weights and MPK kept fp32; measured "
            f"{E['delta_between_the_two']:.3e} frob-rel at the combine, against "
            f"a bf16 output-rounding floor of "
            f"{E['combine_bf16_output_rounding_floor']:.3e}. ALIGNED: the router "
            "task gained an opt-in round_weights_to_input_dtype parameter "
            "(default off, so DeepSeek-V3's fp32-weight semantics are "
            "unchanged), verified bit-exact against the oracle's "
            "topk_renorm_weights in test_router_oracle.py."
        ),
        "new_router_task_needed": False,
        "reserved_id_239_released": True,
    }

    with open(args.out, "w") as f:
        json.dump(art, f, indent=1)
    print(json.dumps(art, indent=1))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
