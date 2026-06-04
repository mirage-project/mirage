# MPK v2 runtime — known debt & follow-ups

Status: the v2/v3 path is at v1 parity (Qwen3-8B, all 8 task types, ~3.38 ms/token
on B200, 40/40 hang-stress clean) but still **experimental**. This file tracks
what is intentionally unfinished, ordered by priority.

## P0 — correctness / design debt

1. **Single source of truth for the per-SM schedule.**
   The round-robin schedule exists twice: `build_v2_worker_task_queues`
   (python/mirage/mpk/v2_task_schedule.py, feeds the SMEM planner) and
   `build_v2_plan` (persistent_kernel_v2.cuh, feeds the kernel). They are
   kept in lockstep only by cross-referenced comments. Fix: the generated
   init code already parses `task_graph.json` for the page regions
   (runtime.cc:645) — make it also read `v2_worker_task_queues` and build
   `v2_per_sm_task_offsets/positions` from that; delete the C++ event-walk.
   Decision (2026-06): Python stays the source of truth because scheduling
   policy and page planning will be co-designed for cross-task overlap.
   The runtime must FAIL LOUDLY if the queues are missing from the JSON —
   no silent fallback to recomputing.

2. **`tiles_per_task > 1` is a latent deadlock.**
   Partial-tile tasks leave inconsistent barrier state (the 2026-05-30
   lm_head hang), and the `bounds_fail` early-returns in linear_v3
   loader/launcher/consumer skip the launcher's blanket page-free →
   page-parity desync if ever reached. Unreachable today (tpt=1 makes
   task count == tile count). Either fix the barrier/page protocol for
   partial tiles or remove the template parameter.

3. **Paged attention v2 SMEM is monolithic.**
   The task declares one big region; the device body hand-rolls offsets and
   ignores `smem_region_offset` (planner registration is accounting-only,
   task_register.cc ~4297). Restore per-stage regions and address through
   `smem_region_offset` like linear v3, so attention pages can participate
   in cross-task overlap.

## P1 — Phase E: cross-task page overlap (`CROSS_TASK_PAGES=false` today)

4. **Planner: `_find_physical_run` ignores `preferred_order`.**
   Multi-page contiguous regions (the W stages — the main overlap
   beneficiaries) always take the lowest free run instead of following the
   fill-fresh-then-earliest-released rule. Harmless while overlap is off;
   must be fixed before flipping `CROSS_TASK_PAGES` on.

5. **Flip `CROSS_TASK_PAGES` on and verify.**
   The device plumbing exists (SmemRing acquire/release/owns, W-ring
   per-stage release in linear_v3's launcher, loader first-touch acquire).
   Needs (4), a page-aware scheduler (see 1), a fresh 40-run hang
   stress + perf measurement, AND the codegen loader page-prefix must skip
   ring-owned pages — today it waits ALL 14 pages up front, which would
   serialize against the prior task and defeat the per-stage acquire.

6. **Event-trigger latency.**
   The controller triggers intra-stream producer events only at slot reuse
   (up to RING-1 instructions of latency; see comment in
   runtime_v2.cuh controller loop). If profiling shows this hot, move event
   triggering into the storer warp.

## P2 — cleanup

7. **demo.py duplicates `compile()`'s planning dance.**
   demo/qwen3/demo.py generates + plans the task graph itself (for the
   inspection JSON / --profiling figure) and then `mpk.compile()` does it
   all again internally. Generate once; reuse.

8. **`fence.proxy.async` in the controller is unproven hardening.**
   Isolation testing showed the linear_v3 launcher `__syncwarp` is the
   load-bearing hang fix (syncwarp-only 40/40; fence-only still hung). The
   fence closes a real PTX proxy-ordering gap vs v1's `__syncthreads`, so
   it stays — but if it ever shows up in a profile, it can be re-evaluated.

9. **Planner dead scaffolding.** `invalid_tasks` counter is never
   incremented and `planned_smem_valid` is always True (the planner raises
   instead). Wire them up or remove them.

10. **`linear_with_residual_layer_v2` registers with empty params**
    (persistent_kernel.py ~1409), silently ignoring `tiles_per_task`,
    while the non-residual variant passes `[-1, 1, tiles_per_task]`.
    Harmonize.

11. **`register_variant_smem_size` AND `get_variant_smem_size`
    (task_register.h/.cc) are dead** — nothing calls either; all paths use
    `register_variant_smem_info`/`get_variant_smem_info`. Remove both and
    the stale "future per-SM allocator" comment above them.

12. **Cross-language page-count invariant is unchecked.**
    `MAX_SMEM_PAGES_PER_TASK=14` (runtime_header.h) ↔ `NUM_PAGES=14`
    (v2_smem_planner.py) is enforced by nothing. Emit the constant into the
    JSON (or a generated header the planner reads) and assert.

13. **linear v2 vs v3.** Both task paths ship for now (decision 2026-06).
    Once v3 has soaked, delete the v2 task + spec and its registration
    wiring (task_register.cc/h, graph.cc, dispatch_v2.cuh).

14. **core.pyx `register_task` ignores its `bgraph` arg** and relies on
    `operators.back()` being the op just customized — fragile implicit
    contract; either use the arg or drop it.

15. **Consolidate the PTX wrappers.** The same mbarrier instructions are
    wrapped in 4 places (runtime_v2.cuh `mbar_*` on `uint64_t*`;
    sm100_ptx.cuh `mbar_*` on int addrs; linear_device.cuh `mbarrier_*`;
    channel.cuh's own `mbar_arrive`/`cp_async_bulk_tensor_3d`), plus
    one-off fences/loads in linear_sm100_v3.cuh. Promote sm100_ptx.cuh to
    the single home (int-addr convention, suspend hint as a parameter to
    merge wait/poll), keep thin using-aliases at the old sites, leave
    op-specific asm (mma_k_block, tcgen05_ld_16) in the ops. MUST be
    SASS-diff-verified (cuobjdump byte-identical) — several variants'
    details were load-bearing during the 2026-05/06 hang debugging
    (suspend hints, forceinline reinit path, .release-vs-plain arrive).
    Own PR; do not mix with functional changes.
