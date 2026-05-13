// DRAFT — Option A per-task barrier for the AllReduce kernel.
//
// This file is intentionally OUTSIDE the build to allow review before patching
// it into `include/mirage/persistent_kernel/tasks/blackwell/allreduce.cuh`.
//
// Design: replace the existing `mpkar_sync_block(team)` call (which uses
// `sync_counter[2*team_idx]` and `pSync[mype]` — shared across all 56
// concurrent task-callers per AR phase per PE) with `mpkar_sync_block_per_task`
// (one private counter+slot quad per task_offset). This eliminates the 56-way
// contention measured in the 2026-05-12 phase-isolation experiment
// (barrier=91 μs/task, removing barrier saves 18.5 ms / 460 μs/phase e2e).
//
// Memory layout in team's psync_pool region (per psync_len_per_team formula):
//   slots 0 .. 2*MPKAR_NVSHMEMI_SYNC_SIZE-1 -> existing legacy barrier
//   (untouched) slots 2*SYNC_SIZE + 0 .. + MAX_AR_TASKS-1 -> per-task counters
//   slots 2*SYNC_SIZE + MAX_AR_TASKS + ... -> per-task pSync (2 buffer × npes
//   per task)
//
// MAX_AR_TASKS = 128 (covers DSv3 hidden=7168 / tile=128 = 56, with headroom
// for future tile-size sweeps).
// Per-team budget = 128 (counters) + 128 * NPES_MAX(8) * 2 = 128 + 2048 = 2176
// longs = 17 KB. Well within psync_len_per_team's reserved
// 2*SYNC_SIZE-3*SYNC_SIZE reduce/bcast region (55 KB available there).
//
// IMPORTANT: this REUSES psync_pool slots reserved by NVSHMEM for reduce/bcast
// ops. MPK does not use those ops, so this is safe — but if anyone adds a
// non-tile NVSHMEM collective later, this layout collides.

#pragma once

// ---- assumed defined in allreduce.cuh ----
// MPKAR_NVSHMEMI_SYNC_SIZE          : 2 * 27648 longs
// mpkar_team_get_psync_sync(teami)  : returns long* (base of team's psync
// region) mpkar_team_translate_pe(teami, i) : i -> world-PE
// mpkar_signal_for_barrier(dst, val, peer) : P2P volatile store
// mpkar_wait_until_ge(addr, val)    : spin

namespace kernel {

static constexpr int MAX_AR_TASKS = 128;

// Per-task contention-free dissemination barrier (block-scope, k = team->size).
// Replaces `mpkar_sync_block(team)` when called inside the AR tile kernel where
// task_offset is the unique linear bid -> bid_offset and team index.
//
// Stationarity assumption (audited 2026-05-12 in scratch/ar_rewrite_design.md
// open question #1): each AR call fires the SAME (task_offset, team) pair on
// EVERY PE exactly once. DSv3 gate-mode early-return is consistent across PEs
// (gated on runtime request state shared via NVSHMEM, not local state).
// Verified safe for: _allreduce_residual (4 sites), MTP MoE allreduce, decode
// AR. NOT safe if a future call site asymmetrically gates a single rank.
static __device__ __forceinline__ void
    mpkar_sync_block_per_task(nvshmem_team_t team, int task_offset) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;

  // For non-P2P teams (job_connectivity > GPU_LDST) the legacy
  // dissemination_generic path is used; we'd need a separate per-task version
  // for that. For now, fall back to the existing barrier if the team isn't
  // fully P2P-connected. (DSv3 TP groups always are.)
  if (!teami->are_gpus_p2p_connected) {
    // Fallback — caller should never hit this on NVLS teams, but guard anyway.
    mpkar_sync_block(team);
    return;
  }
  if (task_offset < 0 || task_offset >= MAX_AR_TASKS) {
    // task_offset out of bounds → conservatively use legacy barrier.
    mpkar_sync_block(team);
    return;
  }

  long volatile *psync_base = (long volatile *)mpkar_team_get_psync_sync(teami);
  long volatile *per_task_region = psync_base + 2 * MPKAR_NVSHMEMI_SYNC_SIZE;
  long volatile *task_counter = per_task_region + task_offset;
  long volatile *task_pSync_base =
      per_task_region + MAX_AR_TASKS + task_offset * size * 2;

  // Atomic increment to claim a unique phase number for THIS task on THIS PE.
  // First call sees counter=0 (psync_pool is zero-initialized by NVSHMEM at
  // bootstrap) and signals phase=1. The +1 is critical: psync slots are also
  // zero-initialized, so wait_until_ge(slot, 0) would trivially pass for the
  // first call's "would-be" signal value 0 → no actual cross-PE sync. By
  // signaling >=1, the receiver actually waits for the peer's increment.
  long my_phase;
  if (threadIdx.x == 0) {
    my_phase = (long)atomicAdd(reinterpret_cast<unsigned long long *>(
                                   const_cast<long *>(task_counter)),
                               (unsigned long long)1) +
               1;
  }
  __shared__ long s_my_phase;
  if (threadIdx.x == 0) {
    s_my_phase = my_phase;
  }
  __syncthreads();
  my_phase = s_my_phase;

  // Slot pair: 2-buffered by phase parity. Slot index = (phase & 1) * size +
  // pe.
  long volatile *pSync = task_pSync_base + (my_phase & 1) * size;

  // For P2P-connected teams, k = size. TP=4 => 3 signals + 3 waits per phase,
  // 1 phase total. Each block uses its own private slot, so 56 concurrent
  // callers no longer contend.
  int k = size;
  int const my_pe = teami->my_pe;
  int const world_my_pe = nvshmemi_device_state_d.mype;

  // Signal phase: write `my_phase` to each peer's slot
  //   `task_pSync_base_on_peer[(phase & 1) * size + world_my_pe]`
  //
  // NOTE: we COULD parallelize across threads (threadIdx.x in [1, k-1]), but
  // for k=4 there are at most 3 peers and the latency-bound signal store
  // doesn't benefit from extra threads. Keep it on lane 0 for clarity and to
  // make ordering across the 3 writes obvious to NVLink's P2P FIFO.
  if (threadIdx.x == 0) {
    for (int j = 1; j < k; j++) {
      int to_nbr_idx = my_pe + j;
      if (to_nbr_idx >= size) {
        to_nbr_idx -= size;
      }
      int to_nbr = mpkar_team_translate_pe(teami, to_nbr_idx);
      // Remote pSync address = same offset relative to peer's heap_base.
      long volatile *remote_dest = pSync + world_my_pe;
      mpkar_signal_for_barrier((long *)remote_dest, my_phase, to_nbr);
    }

    // Wait phase: spin until each peer has written >= my_phase to MY slot
    // (indexed by their world_my_pe relative to MY task_offset's slot pair).
    for (int j = 1; j < k; j++) {
      int from_nbr_idx = my_pe - j;
      if (from_nbr_idx < 0) {
        from_nbr_idx += size;
      }
      int from_nbr = mpkar_team_translate_pe(teami, from_nbr_idx);
      mpkar_wait_until_ge(pSync + from_nbr, my_phase);
    }
  }
  __syncthreads();
}

} // namespace kernel

// ---- planned patch to allreduce.cuh::nvshmem_tile_allreduce_impl ----
//
// (around line 558, after __threadfence())
//
//   __threadfence();
// #if defined(MPK_AR_PER_TASK_BARRIER)
//   mpkar_sync_block_per_task(team, task_offset);
// #elif !defined(MPK_AR_SKIP_BARRIER)
//   mpkar_sync_block(team);
// #endif
//
// Gating order: MPK_AR_PER_TASK_BARRIER takes priority over the legacy
// path; MPK_AR_SKIP_BARRIER (committed 2026-05-12 in d6d1730a) remains as a
// measurement-only gate that elides both.

// ---- planned patch to python/mirage/mpk/persistent_kernel.py ----
//
// (in the same env→-D section that handles MPK_AR_SKIP_BARRIER)
//
//   if os.environ.get("MPK_AR_PER_TASK_BARRIER") == "1":
//       common_cmd.append("-DMPK_AR_PER_TASK_BARRIER")

// ---- correctness validation plan (live, requires GPU) ----
//
// 1. Qwen3 small smoke (TP=4, layers 0-3, mbt=1): verify same output token IDs
//    with vs without MPK_AR_PER_TASK_BARRIER=1.
// 2. DSv3 layers 0-3 prefill-128: per-layer cosine vs FP8 ref ≥ 0.99 with
//    MPK_AR_PER_TASK_BARRIER=1.
// 3. DSv3 prefill-128 layers 0-19 prefill+1-decode: per-token latency and
//    AR per-task wallclock with vs without.
//
// Expected outcomes if Option A succeeds:
//   - per-AR-task wallclock drops to ~290-310 μs (down from 370, mostly
//     barrier contention removed but reduce + dep-spin unchanged)
//   - e2e drops to 165-170 ms (recovers most of the 18.5 ms barrier saving)
// If Option A is noise: contention isn't the dominant cost → re-design
// (likely Option B or revisit the reduce path).
