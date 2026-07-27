#!/usr/bin/env python3
"""Analytic memory-roofline characterisation of the vLLM decode kernels MPK must beat.

Substitute for the Nsight Compute SpeedOfLight section, which could not be collected on
catalyst-B200 (see ncu/NCU_UNAVAILABLE.md).  Every byte count below is derived from the exact
shapes in docs/qwen35/vllm-graph.md 3.3 / 4.1 / 4.2; every microsecond is the measured median
from tables/bs*_kernels.csv.  Achieved bandwidth = bytes / measured time, and the roof is B200
HBM3e at 8.0 TB/s.

This tells a ferret task the thing NCU's SOL page would have told it: how far the incumbent
kernel is from the memory roof, i.e. how much room a better kernel actually has.
"""
import csv
import json
from pathlib import Path

HBM_TB_S = 8.0            # B200 HBM3e peak, TB/s
OUT = Path(__file__).resolve().parent.parent / "ncu"

MB = 1024.0 * 1024.0

# (label, bytes moved per decode step, measured us/step bs1, bs8, bs16, note)
ROWS = [
    ("dense fp8 projections (160 sites)",
     (30 * 12288 * 2048 + 30 * 2048 * 4096 + 10 * 9216 * 2048 + 10 * 2048 * 4096
      + 40 * 1024 * 2048 + 40 * 2048 * 512),          # fp8 weights, 1 byte each
     1353.50, 1434.72, 1370.96,
     "weight-streaming dominated; activations negligible at M<=16"),
    ("  - in_proj_qkvz [12288,2048] x30", 30 * 12288 * 2048, 30 * 9.45, None, None,
     "per site 9.45 us at bs1 (ordinal analysis)"),
    ("  - gdn out_proj [2048,4096] x30", 30 * 2048 * 4096, 30 * 10.75, None, None,
     "per site 10.75 us at bs1"),
    ("  - qkv(g)_proj [9216,2048] x10", 10 * 9216 * 2048, 10 * 9.45, None, None, ""),
    ("  - o_proj [2048,4096] x10", 10 * 2048 * 4096, 10 * 10.75, None, None, ""),
    ("  - shared gate_up [1024,2048] x40", 40 * 1024 * 2048, 40 * 7.80, None, None, ""),
    ("  - shared down [2048,512] x40", 40 * 2048 * 512, 40 * 5.80, None, None, ""),
    ("MoE routed w13 (top-8 of 256) x40", 40 * 8 * 1024 * 2048, 336.56, 879.83, 1123.23,
     "bs8/bs16 read more experts: bs8 ~<=64 groups, bs16 ~<=128 groups"),
    ("MoE routed w2 (top-8 of 256) x40", 40 * 8 * 2048 * 512, 300.66, 699.07, 755.94, ""),
    ("lm_head [248320,2048] bf16 x1", 248320 * 2048 * 2, 150.72, 152.88, 155.13,
     "the reference point: this one IS at the roof"),
    ("GDN recurrent state rd+wr x30", 30 * 2 * 32 * 128 * 128 * 4, 163.65, 270.43, 462.74,
     "fp32 state [32,128,128] per layer per request; scales with batch"),
    ("GDN conv1d state rd+wr x30", 30 * 2 * 3 * 8192 * 2, 89.61, 91.68, 95.61, ""),
    ("quantize / fp8 casts (200 sites)",
     (30 * 2048 + 30 * 4096 + 10 * 2048 + 10 * 4096 + 40 * 2048 + 40 * 512) * 3,
     559.28, 620.50, 606.23,
     "read bf16 + write fp8 + fp32 scales, M=1 row per site at bs1 - about 1 MB of traffic"),
]

BATCH_SCALE = {"bs1": 1, "bs8": 8, "bs16": 16}
SCALES_WITH_BATCH = {"GDN recurrent state rd+wr x30", "GDN conv1d state rd+wr x30",
                     "quantize / fp8 casts (200 sites)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for label, nbytes, u1, u8, u16 in [(r[0], r[1], r[2], r[3], r[4]) for r in ROWS]:
        note = next(r[5] for r in ROWS if r[0] == label)
        rec = {"kernel_or_stage": label, "bytes_per_step_bs1": nbytes,
               "MB_per_step_bs1": round(nbytes / MB, 2), "note": note}
        for bs, us in (("bs1", u1), ("bs8", u8), ("bs16", u16)):
            if us is None:
                continue
            b = nbytes * (BATCH_SCALE[bs] if label in SCALES_WITH_BATCH else 1)
            tbs = (b / 1e12) / (us / 1e6)
            rec[f"us_per_step_{bs}"] = round(us, 2)
            rec[f"MB_per_step_{bs}"] = round(b / MB, 2)
            rec[f"achieved_TB_s_{bs}"] = round(tbs, 3)
            rec[f"pct_of_HBM_roof_{bs}"] = round(100 * tbs / HBM_TB_S, 2)
            rec[f"roofline_us_{bs}"] = round((b / 1e12) / HBM_TB_S * 1e6, 2)
            rec[f"headroom_x_{bs}"] = round(us / ((b / 1e12) / HBM_TB_S * 1e6), 2)
        out.append(rec)

    (OUT / "roofline.json").write_text(json.dumps(
        {"hbm_peak_TB_s": HBM_TB_S, "source_us": "tables/bs*_kernels.csv (this issue)",
         "source_shapes": "docs/qwen35/vllm-graph.md 3.3/4.1/4.2", "rows": out}, indent=2))
    cols = ["kernel_or_stage", "MB_per_step_bs1", "us_per_step_bs1", "achieved_TB_s_bs1",
            "pct_of_HBM_roof_bs1", "roofline_us_bs1", "headroom_x_bs1",
            "us_per_step_bs16", "achieved_TB_s_bs16", "pct_of_HBM_roof_bs16",
            "headroom_x_bs16", "note"]
    with open(OUT / "roofline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"{'MB/step':>9}{'us/step':>9}{'TB/s':>8}{'% roof':>8}{'roof us':>9}{'x off':>7}  kernel/stage")
    for r in out:
        if "achieved_TB_s_bs1" not in r:
            continue
        print(f"{r['MB_per_step_bs1']:9.1f}{r['us_per_step_bs1']:9.1f}"
              f"{r['achieved_TB_s_bs1']:8.2f}{r['pct_of_HBM_roof_bs1']:8.1f}"
              f"{r['roofline_us_bs1']:9.1f}{r['headroom_x_bs1']:7.1f}  {r['kernel_or_stage']}")
    print("\nbs16:")
    for r in out:
        if "achieved_TB_s_bs16" not in r:
            continue
        print(f"{r['MB_per_step_bs16']:9.1f}{r['us_per_step_bs16']:9.1f}"
              f"{r['achieved_TB_s_bs16']:8.2f}{r['pct_of_HBM_roof_bs16']:8.1f}"
              f"{r['roofline_us_bs16']:9.1f}{r['headroom_x_bs16']:7.1f}  {r['kernel_or_stage']}")
    print(f"\nwrote {OUT / 'roofline.csv'}")


if __name__ == "__main__":
    main()
