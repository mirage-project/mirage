# M3-I11 — MPK run-to-run nondeterminism: what it is, and what it is not

Pinned commit for every run below: **65e42ee8** (the box clone
`~/mpk-qwen35/mirage` was verified clean at that SHA before and after). GPU 6,
sole tenant, claimed with the 3-sample guard. The two commits that landed in the
shared repo during this window (`0c8b4cf5` M3-I5c router prep, `68f93b3a`
tie-aware router tests) are **not** in any measurement here; `0c8b4cf5` is
discussed below because it fixes one of the two mechanisms this issue found.

## Summary

Three findings, in order of confidence.

1. **A live run-to-run-varying write in the decode path is confirmed on
   hardware, and it is the MoE router's expert-list compaction.** The compacted
   `mpk_active_expert_ids` list comes out in a **different order in 767 of 800**
   same-input comparisons across five independent processes. Its **set and count
   are identical in all 800**. This is the `atomicAdd` compaction M3-I5c
   diagnosed by source reading and host model; this is its first hardware
   evidence. Its fix is already in the tree (`0c8b4cf5`).

2. **That write is value-neutral at the geometries we run**, and so is
   everything else: **42 runs / 420 trajectories / 8176 state-level comparisons
   produced zero differences**, all matching the M3-I9b census consensus md5
   `be346b6d868ef8e7e46980426e662722`.

3. **The census's token divergences did not reproduce.** The m3i9 window saw
   5 anomalous runs in 27; 42 runs here produced none. If the rate were
   unchanged, P(0 in 42) ≈ 2 × 10⁻⁴. Something about that window, not the code
   at this SHA, carried the anomalies. **M3-I11 is therefore not closed as
   "fixed"** — see "What is still open".

A second, independent defect was found by source reading and is reported but
**deliberately not patched here**: `linear_sm100_mpk.cuh` signals task
completion with a TMA wait that does not cover the store's destination write.

## The instrument (the part worth reusing)

Comparing emitted token ids is a poor detector for numerical nondeterminism: a
perturbation is only visible when it happens to cross an argmax margin, which
the census measured at ~2% of trajectories. So `scripts/e2_fingerprint.py`
fingerprints the **paged KV cache** instead. It is a builder-owned torch tensor
holding K and V for every position of every attention layer, written once per
position and never rewritten, so the first `(layer, cache slot)` at which two
runs differ is the first moment their arithmetic differed — ULP or not, argmax
flip or not. `scripts/e4_full.py` snapshots it at every wave boundary and also
emits the standard `bs<N>.json` dump, so a run is directly comparable to the
committed census md5s.

The fingerprint is bitwise (bf16/fp32 reinterpreted as unsigned integers, two
independent index-weighted mixes folded together), so a 1-ULP change moves it.

## Evidence

### Reproduction attempt (E4/E5) — `results/determinism_campaign.json`

Config identical to `plan_m3i9.sh` stage 7: the ten reference prompts,
`--max-seq-length 1280 --max-new-tokens 1024`, warm reuse of the very same
cached `.so` files the m3i9 runs loaded.

| bs | runs | trajectories | pairwise array comparisons | differing |
|---:|-----:|-------------:|---------------------------:|----------:|
| 1  | 9    | 90           | 1800                       | **0** |
| 4  | 23   | 230          | 5566                       | **0** |
| 8  | 10   | 100          | 810                        | **0** |
| **total** | **42** | **420** | **8176**              | **0** |

All 42 token dumps hash to the census consensus `be346b6d…`. The comparison
covers per-wave KV/GDN state fingerprints *and* the token arrays.

### Single-prompt campaigns (E1/E2/E3)

28 further trajectories of one prompt at bs1, all bit-identical, used to refute
specific mechanisms:

- **H1, uninitialised intermediates — refuted at bs1.** Every megakernel
  intermediate is `cudaMalloc`ed and never zeroed
  (`src/kernel/runtime.cc:1274`), so this was the leading hypothesis.
  `scripts/e3_churn.py` dirties 3 GiB in launcher-sized blocks immediately
  before the launcher's allocations, with a chosen fill byte. Runs differing
  **only** in that byte (0xAA vs 0x55, plus a repeat and two clean controls)
  are bit-identical, 12/12.
- **Cold vs warm — refuted at bs1.** An in-process `compile()` into a fresh
  kernel dir matches the cached-load runs exactly. (This was the census's
  strongest correlate: 3 of its 6 anomalies were the rep that compiled.)
- **ECC / hardware — refuted.** Zero volatile and zero aggregate uncorrected
  ECC errors on all eight B200s; no pending row remaps.

### Router compaction order (E7) — `results/router_mask_order.txt`

`layer_<i>_moe_mask` *is* `mpk_active_expert_ids`. Building the graph with
`expose_intermediates` makes it a readable torch tensor, so all 40 layers' lists
can be compared across processes (`scripts/e7_router_mask.py`).

Five processes, two waves each, same prompt, 80 mask samples per run:

```
pairs compared: 800
  count differs : 0
  order differs : 767   (of which SET differs: 0)
  token dumps differing: 0
  non-ascending lists: 77-80 of 80 per run
```

Example (`p1` vs `p2`, layer 0): `[33 56 136 78 89 241 14 18]` vs
`[14 18 136 78 89 241 33 56]`.

So the list order is decided by CTA arrival order, exactly as
`topk_softmax_sm100.cuh` / `topk_sigmoid_sm100.cuh` Phase 7 implies
(`int pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1)` before
`0c8b4cf5`). **Why it is benign today:** the grouped GEMM consumes the list with
`for (int ae = expert_offset; ae < num_activated; ae += expert_stride)` and then
`int const expert = d_mask[ae]`, addressing everything by expert **id**
(`moe_fp8_blockscale_sm100.cuh:196-197`). Permuting the list only re-assigns
experts to CTAs; the strided loop means no expert is dropped at any list length.

**Why it is not benign forever.** The same pre-`0c8b4cf5` code has the
unbarriered read-then-scatter, whose compacted entries alias the marks of
experts `[0, n_active)`. At bs1 `n_active` is 8, the aliasing window sits
entirely inside warp 0, and warp 0 issues all its loads before any of its
stores — which is why 0 of 800 samples showed a count or set difference. At
bs16 `n_active` is ~87 (M3-I8's measurement), the aliasing window spans three
warps that both read and scatter, and the race has a real window. A phantom
expert is then processed with a **stale `routing` row** — `routing` is a
`cudaMalloc`ed intermediate written per active expert — and would gather live
token rows and write `down[token, slot]` slots the real expert also owns. M4's
final gate runs bs16.

## The second defect (reported, NOT patched here)

`include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh:720-723`
ends a task with

```cpp
    if (warp_idx == 0 && cute::elect_one_sync()) {
      cute::tma_store_wait<0>();
    }
```

`cute::tma_store_wait<N>` emits `cp.async.bulk.wait_group.read N`
(`deps/cutlass/include/cute/arch/copy_sm90_tma.hpp:1248`). The `.read` qualifier
retires when the bulk copies have finished **reading their source shared
memory** — i.e. when smem may be reused. It does not wait for the destination
global write. By construction the read completes *before* the write.

In an ordinary CUTLASS kernel that is sufficient, because the CTA then exits and
kernel completion flushes the async proxy. Under the persistent runtime the CTA
does not exit: it immediately release-increments the task's trigger event
(`persistent_kernel.cuh:1063`, `atom_add_release_gpu_u64`) and another worker
CTA that acquires that counter (`persistent_kernel.cuh:1004`) reads the tensor
with ordinary loads. `atom.add.release.gpu` orders generic-proxy accesses; it
does not order in-flight **async-proxy** writes, and `__syncthreads()` is a
CTA-scope generic-proxy barrier. So a consumer can observe the event while the
producer's last output atom is still in flight.

On Blackwell this reaches Qwen3.5 through `linear_layer` only — `linear_sm100`
is the registered task (`task_register.cc:1851`) and the call sites are the
lm_head, the MoE router projection and GDN `in_proj_ba`. The dense fp8 path
(`linear_fp8_blockscale_sm100.cuh`) uses plain global stores and is unaffected.

`python/mirage/mpk/models/utils.py:3-15` already carries an unexplained TODO for
this same kernel — "*both MPK ptx and cutlass version will output unexpected
result (not same output for same prompt) if the OUTPUT_SIZE is too big*" — and
works around it by capping the per-task output at 256 columns.

The fix is one line (`cp.async.bulk.wait_group 0` without `.read`, plus
`fence.proxy.async.global`, before the task's release). It is not landed here
because it could not be shown to change anything: with no reproduction, the only
evidence obtainable was "42 more runs still identical", which does not
distinguish a fix from a no-op. It needs its own scoped change with a
bit-exactness run and a perf comparison.

## What is still open

The census phenomenon is real (M3-I9b's `census_window2.json`, 6 anomalous
dumps of 80) and is **not explained** by anything measured here. What the
divergences look like, from the committed dumps:

- They are **whole-wave** events. `s7_cap_bs4_rep2` diverges on p07/p05/p08/p10
  — exactly the four prompts of wave 2, and only those. `s7_base_bs4_rep1`
  diverges on p03+p02 = exactly wave 3. `s7_base_bs8_rep1` likewise on its
  wave 2. One perturbation per wave, hitting every live slot.
- p03-python position 632 flips `198 → 271` in **two independent runs**
  (`s7_base_bs4_rep1` and `s7_base_bs8_rep1`, different batch sizes, different
  kernels), and in both the next 40 tokens are identical. A reproducible
  position with a reproducible alternative.
- Every anomaly landed on GPU 1, 4 or 7 during a heavily contended window; all
  42 clean runs here were on GPU 6 (and the box was at 100% on GPUs 0-5 for most
  of the campaign, so box load alone is not the difference).

Recommended next steps, in order:

1. **Sequence M3-I5c's hardware validation.** Its `prep.md` pre-registers it and
   `stress_compaction.py` is written. E7 above supplies the pre-fix baseline:
   767/800 order differences and 77-80/80 non-ascending lists. Post-fix both
   must be **0** — a sharp, cheap falsifier.
2. **Re-run the census on GPUs 1/4/7** with `e4_full.py` attached. This is the
   only untested difference between the anomalous window and this one, and the
   fingerprint makes each run ~40× more informative than a token dump.
3. **Land the TMA-store wait fix** as its own reviewed change.
4. **M4's gate must require ≥2 same-config reps** regardless — the protocol
   amendment below is in force whether or not the cause is found.

## Protocol amendments (landed with this issue)

- `harness/run_ac3.py` now refuses an ambiguous `--engine-dump-dir` tree
  (`resolve_dump_tree`): more than one `bs<N>.json` candidate anywhere in the
  tree, or a single candidate that is not at the top level, is an integrity
  error (exit 2) listing every candidate. M3-I9 pointed the flag at a parent of
  twenty run directories and one anomalous dump out of eighty was reported as a
  policy effect. Regression test: `harness/tests/test_dump_tree.py` (5 cases,
  including the exact M3-I9 tree shape).
- `docs/qwen35/bench-protocol.md`'s determinism section is updated with the
  findings above; its ≥2-rep rule for any "policy X changed the tokens" claim
  was already in force and now cites this issue's evidence.

## Files

- `scripts/` — the detector and every campaign runner, as executed.
- `results/determinism_campaign.json` — the 42-run / 420-trajectory result.
- `results/router_mask_order.txt` — the E7 router-order measurement.
- Raw fingerprints: `/home/catalyst/mpk-artifacts/m3i11/`.
