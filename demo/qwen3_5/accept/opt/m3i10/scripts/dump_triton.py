#!/usr/bin/env python3
"""Map inductor triton kernel NAMES (as they appear in a kineto trace) to their real ops.

Inductor emits, immediately before each kernel registration:
    # Topologically Sorted Source Nodes: [...], Original ATen: [...]
    NAME = async_compile.triton('NAME', ''' ... ''', device_str='cuda')

vLLM piecewise-compiles each layer slice into its own subgraph artifact, and inductor numbers
kernels PER SUBGRAPH - so the same short name (e.g. triton_poi_fused_3) can mean different
things in different subgraphs.  A kineto trace only carries the name, so this script reports
EVERY artifact that defines each name and flags provenance disagreement.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
NAMES = sys.argv[2:]

REG = re.compile(r"^([A-Za-z_0-9]+) = async_compile\.triton\(", re.M)
TOPO = re.compile(r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]", re.S)

texts = {}
for f in sorted(ROOT.glob("artifact_compile_range_*")):
    try:
        texts[f.name] = f.read_text(errors="replace")
    except Exception:
        pass

report = {}
for name in NAMES:
    per_art = []
    for fn, t in texts.items():
        regs = [(mm.start(), mm.group(1)) for mm in REG.finditer(t)]
        for i, (pos, nm) in enumerate(regs):
            if nm != name:
                continue
            prev_end = regs[i - 1][0] if i else 0
            head = t[prev_end:pos]
            tm = None
            for tm in TOPO.finditer(head):
                pass  # keep the LAST provenance comment before this registration
            if tm:
                per_art.append({"artifact": fn,
                                "source_nodes": " ".join(tm.group(1).split()),
                                "original_aten": " ".join(tm.group(2).split())})
            else:
                per_art.append({"artifact": fn, "source_nodes": "?", "original_aten": "?"})
    distinct = sorted({p["original_aten"] for p in per_art})
    report[name] = {"n_artifacts": len(per_art),
                    "n_distinct_provenance": len(distinct),
                    "artifacts": sorted({p["artifact"] for p in per_art}),
                    "variants": per_art[:200]}
    print("=" * 110)
    print(f"### {name}   [{len(per_art)} definitions, {len(distinct)} distinct provenance]")
    seen = set()
    for p in per_art:
        key = p["original_aten"]
        if key in seen:
            continue
        seen.add(key)
        print(f"  artifact {p['artifact'].replace('artifact_compile_range_1_16384_', '')}")
        print(f"    source_nodes : {p['source_nodes'][:700]}")
        print(f"    original_aten: {p['original_aten'][:500]}")
    if len(distinct) > 1:
        print("  *** NAME COLLISION: this trace name aggregates DIFFERENT kernels ***")

Path(sys.argv[1] + "/../../../triton_name_map.json") if False else None
print("\n\nJSON:")
print(json.dumps(report, indent=1)[:200])
with open("/home/muhengl/mpk-qwen35/m3i10-profile/out/triton_name_map.json", "w") as f:
    json.dump(report, f, indent=1)
