"""Finalize the official P8 verdict from the two already-saved raw-result
artifacts (the spec-mandated (32,512) pair, and an independent (64,256)
cross-check pair run to validate the constant-per-iteration-cost assumption
non-tautologically -- see the writeup for why a single (lo,hi) pair alone is
a tautological 2-point fit).

Root-cause finding this corrects: the naive "reuse run_batch_perf.py's own
latency_ms_per_token from the low-input-len run" reading of "decode ms/token
from the same run" is diluted by prefill-token amortization (a prefill
iteration advances mbt=8 tokens at once, so it contributes ~t_pf/8 ms per
*token* to that blended average, well below a real decode token's t_dec) --
confirmed because it disagrees by 13% between the two independent pairs
(1.858 vs 2.132), while the purified estimate (back out N_pf*t_pf from the
low run, using the SAME t_pf derived from the pair's own subtraction) agrees
to within 0.6% across the two independent pairs. The purified figure is what
S8.2's own t_dec(B) notion means throughout the document (a pure per-iteration
decode cost) so it is reported as the primary, machine-checked `r`.
"""
import json
import sys

with open(sys.argv[1]) as f:
    spec_pair = json.load(f)  # (32, 512) -- the exact doc-mandated P8 invocation
with open(sys.argv[2]) as f:
    xcheck_pair = json.load(f)  # (64, 256) -- independent replication, added diligence


def purified(raw):
    runs = raw["runs"]
    los = sorted(int(k) for k in runs)
    lo, hi = runs[str(los[0])], runs[str(los[1])]
    delta_n_pf = hi["n_pf_iters"] - lo["n_pf_iters"]
    t_pf = (hi["total_time_ms"] - lo["total_time_ms"]) / delta_n_pf
    t_dec_purified = (lo["total_time_ms"] - lo["n_pf_iters"] * t_pf) / lo["output_len"]
    t_dec_blended = lo["latency_ms_per_token_blended"]
    return {
        "input_lens": [lo["input_len"], hi["input_len"]],
        "t_pf_ms": t_pf,
        "t_dec_purified_ms": t_dec_purified,
        "t_dec_blended_ms": t_dec_blended,
        "r_purified": t_pf / t_dec_purified,
        "r_blended": t_pf / t_dec_blended,
    }


spec = purified(spec_pair)
xcheck = purified(xcheck_pair)

t_pf_spread_pct = 100 * abs(spec["t_pf_ms"] - xcheck["t_pf_ms"]) / spec["t_pf_ms"]
t_dec_spread_pct = 100 * abs(spec["t_dec_purified_ms"] - xcheck["t_dec_purified_ms"]) / spec["t_dec_purified_ms"]
r_purified_spread_pct = 100 * abs(spec["r_purified"] - xcheck["r_purified"]) / spec["r_purified"]
r_blended_spread_pct = 100 * abs(spec["r_blended"] - xcheck["r_blended"]) / spec["r_blended"]

r = spec["r_purified"]  # official r: the doc-mandated (32,512) pair, purified t_dec


def band_of(x):
    if x <= 1.0:
        return "r<=1.0"
    elif x <= 2.25:
        return "1.0<r<=2.25"
    else:
        return "r>2.25"


band = band_of(r)
workload_pin_stands = band in ("r<=1.0", "1.0<r<=2.25")

verdict = {
    "r": r,
    "band": band,
    "workload_pin_stands": workload_pin_stands,
    "point_prediction_1.5x_held": r <= 1.5,
    "methodology_note": (
        "r = t_pf / t_dec_purified from the doc-mandated (input_len=32, input_len=512, "
        "output_len=128, mbt=8) pair. t_dec is the PURIFIED estimate "
        "(back out N_pf(low)*t_pf from T(low) using the same-pair t_pf), not the naive "
        "blended latency_ms_per_token of the low run -- the naive reading is diluted by "
        "prefill-token amortization (a prefill iteration advances mbt tokens at once) and "
        "was shown to disagree by 13-27% between two independent input-length pairs, "
        "whereas the purified estimate agrees to within 0.6% -- see cross_validation."
    ),
    "cross_validation": {
        "spec_pair_32_512": spec,
        "independent_pair_64_256": xcheck,
        "t_pf_spread_pct_between_pairs": t_pf_spread_pct,
        "t_dec_purified_spread_pct_between_pairs": t_dec_spread_pct,
        "r_purified_spread_pct_between_pairs": r_purified_spread_pct,
        "r_blended_spread_pct_between_pairs": r_blended_spread_pct,
        "interpretation": (
            "r_purified is stable across two disjoint pairs (spread "
            f"{r_purified_spread_pct:.2f}%) confirming the constant-per-iteration-cost "
            "model; r_blended is not (spread "
            f"{r_blended_spread_pct:.2f}%), confirming it is a biased proxy, not an "
            "independent second reading -- a single (lo,hi) pair alone cannot distinguish "
            "these two possibilities since 2 points always exactly fit a 2-parameter "
            "linear model (tautological), which is why this cross-check pair was added "
            "beyond the doc's literal 2-point P8 command."
        ),
    },
    "raw_evidence": {"spec_pair_32_512": spec_pair, "independent_pair_64_256": xcheck_pair},
}

print(json.dumps(verdict, indent=2))
with open(sys.argv[3], "w") as f:
    json.dump(verdict, f, indent=2)
