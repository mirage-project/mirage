import json, os, sys, csv, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep3 as S
rows=[]
for geom, getter, extra in (("A", lambda q,b,r: (S.sample_A(q,b,r) or (None,None))[0], "sum_wave_wall_ms"),
                            ("B", lambda q,b,r: S.sample_BC("B",q,b,r), "wave_wall_ms"),
                            ("C", lambda q,b,r: S.sample_BC("C",q,b,r), "wave_wall_ms")):
    for bs in S.BSL:
        a={q:S.agg([getter(q,bs,r) for r in S.REPS]) for q in S.ARMS}
        if any(a[q] is None for q in S.ARMS): continue
        rows.append(dict(geometry=geom, metric=extra, bs=bs,
            pass4_median_ms=round(a[4]["median"],1), pass4_spread_pct=round(a[4]["pct"],2), pass4_n=a[4]["n"],
            pass2_median_ms=round(a[2]["median"],1), pass2_spread_pct=round(a[2]["pct"],2), pass2_n=a[2]["n"],
            pass1_median_ms=round(a[1]["median"],1), pass1_spread_pct=round(a[1]["pct"],2), pass1_n=a[1]["n"],
            ratio_2_over_4=round(a[2]["median"]/a[4]["median"],4),
            ratio_1_over_4=round(a[1]["median"]/a[4]["median"],4),
            ratio_1_over_2=round(a[1]["median"]/a[2]["median"],4)))
if S.discarded: print("DISCARDED:", S.discarded, file=sys.stderr)
w=csv.DictWriter(open(sys.argv[2],"w",newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", sys.argv[2], "rows", len(rows), "discarded", len(S.discarded))
