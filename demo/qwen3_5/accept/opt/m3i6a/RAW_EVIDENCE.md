# M3-I6a raw evidence — locations

Per-rep metadata, timings, gate logs, and every script are committed IN-TREE under
`raw_meta/` (3.5 MB): per-arm/per-rep `meta_*.json` (each carries `gpu_before` — the
contamination audit), `iters.csv`, the gate logs (`gates/`), the context-curve and analysis
scripts, the GPU guard + drain gate (`gpu_guard_i6a.sh`, `claim_and_run.sh`), and the
DISCARDED contaminated arm with its audit trail.

Raw profiler captures (6 npz, 4.9 GB — too large for git) are archived at
`/home/catalyst/mpk-artifacts/m3i6a/` (box originals under `~/mpk-qwen35/i6a/`). Every
committed table regenerates from them with the in-tree scripts.

`head_ac3/` holds the AC-3 per-case byte-diff rerun on INTEGRATED HEAD (i.e. including
M3-I11's TMA fence `0cdd52f0` and this issue's `a86b1eb1`), which is the milestone rule's
integrated-HEAD gate; the pre-integration run at `170ab325` is retained beside it.
