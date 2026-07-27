# Raw profiler traces — archival pointers

All raw captures live OUTSIDE the repo (multi-GB); derived tables in-tree are regenerable
from them via the committed scripts.

| set | location | contents |
|---|---|---|
| vLLM torch-profiler chrome traces (the per-kernel source) | `/home/catalyst/mpk-artifacts/m3i10-vllm-traces/main/` (21 files, 1.1G; box original `~/mpk-qwen35/m3i10-profile/traces/main/`) | 18 decode windows (6 per bs x {1,8,16}) + 3 prefill |
| MPK matched-geometry npz (arms A/B) | `/home/catalyst/mpk-artifacts/m3i10-remeasure/{armA,armB}/` | profiled rep0 per bs |
| MPK late-context npz | `/home/catalyst/mpk-artifacts/m3i10-remeasure/armAlate/` | profiled rep0 per bs |
| I8 F1 oracle npz | `/home/catalyst/mpk-artifacts/m3i8-f1-raw/`, `/home/catalyst/mpk-artifacts/m3i8-f1ext/` | bs1/2 + bs4/8/16 |

Reproduce: vLLM tables via `scripts/` (see comparison.md §9); MPK tables via
`remeasure/scripts/` against the npz.
