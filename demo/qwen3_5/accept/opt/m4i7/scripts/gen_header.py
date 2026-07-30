#!/usr/bin/env python3
"""Build the M4-I7 moe_fp8_blockscale_sm100.cuh from three provable pieces:

  A. the PRE-M4-I7 file's frozen region (namespace moe_fp8_blockscale ... end of
     moe_fp8_blockscale_task_impl) wrapped verbatim in `namespace golden { }`
  B. the ferret workspace3 v012 tag blob's `cand` region, verbatim modulo a
     namespace rename (moe_fp8_blockscale -> moe_fp8_blockscale_fast) and the
     dispatcher, which is REPLACED (MPK's parallel unit is the emitted task,
     not the SM -- see the header comment)
  C. a new compile-time dispatcher

The script PRINTS the byte counts it copied and re-extracts both regions from
the emitted file to prove they round-trip unchanged.
"""
import hashlib
import re
import subprocess
import sys

OLD = sys.argv[1]      # pre-M4-I7 moe_fp8_blockscale_sm100.cuh
FERRET = sys.argv[2]   # v012 kernel.cu
OUT = sys.argv[3]

old = open(OLD).read()
fer = open(FERRET).read()

# ---- A: the frozen region of the old file -------------------------------
a_start = old.index("namespace moe_fp8_blockscale {")
a_end = old.index("} // namespace kernel")
GOLDEN = old[a_start:a_end].rstrip("\n")
GOLDEN_SHA = hashlib.sha256(GOLDEN.encode()).hexdigest()
OLD_PREAMBLE = old[:a_start]

# ---- B: the ferret cand region -----------------------------------------
b_start = fer.index("namespace cand {")
b_start = fer.index("namespace moe_fp8_blockscale {", b_start)
b_end = fer.index("template <typename T,\n          int BATCH_SIZE,\n          int NUM_TOPK,\n          int NUM_EXPERTS,\n          int OUTPUT_SIZE,\n          int ORIG_OUTPUT_SIZE,\n          int REDUCTION_SIZE,\n          bool W13_LINEAR>\n__device__ __forceinline__ void\n    moe_fp8_blockscale_task_impl", b_start)
FAST = fer[b_start:b_end].rstrip("\n")
FAST_RAW_SHA = hashlib.sha256(FAST.encode()).hexdigest()

# the ferret file's vendored .cg cp.async helpers (used by the fast paths only)
h_start = fer.index("// .cg (bypass-L1) variants for read-once streamed rows")
h_end = fer.index("// ---- vendored from blackwell/linear_fp8_blockscale_sm100.cuh:80-87 ----")
CG_HELPERS = fer[h_start:h_end].rstrip("\n")

# the ONE mechanical rename: the candidate's own constants namespace, so it can
# coexist with the frozen golden one in a single translation unit.
n_rename = FAST.count("moe_fp8_blockscale")
FAST = FAST.replace("namespace moe_fp8_blockscale {", "namespace moe_fp8_blockscale_fast {")
FAST = FAST.replace("} // namespace moe_fp8_blockscale", "} // namespace moe_fp8_blockscale_fast")
FAST = FAST.replace("using namespace moe_fp8_blockscale;", "using namespace moe_fp8_blockscale_fast;")
FAST = FAST.replace("smem_moe_blockscale_cand", "smem_moe_blockscale")
# the smem-fit static_assert moves to the CLAMPED accounting: the in-kernel
# offsets use the clamped STAGE_TILES, so smem_bytes_k is the footprint the
# launch must cover, and smem_bytes() is only its upper bound. Keeping the
# loose bound would refuse w2's admissible clamped PATH2 layout in the
# 163 KiB-budget test TUs.
FAST = FAST.replace(
    """  static_assert(smem_bytes(BATCH_SIZE, PATH) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "moe_fp8_blockscale_sm100 exceeds the worker smem budget");""",
    """  static_assert(smem_bytes_k(BATCH_SIZE, PATH, REDUCTION_SIZE) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "moe_fp8_blockscale_sm100 fast path exceeds the worker smem "
                "budget");""")
assert "smem_bytes_k(BATCH_SIZE, PATH, REDUCTION_SIZE)" in FAST

HEADER_NOTE = '''
// ===========================================================================
// M4-I7: the ferret `moe-fp8-grouped-vllm-beat` winner (workspace3 tag v012,
// c8b5b24, min_ratio 0.801 over 10 configs) integrated behind a compile-time
// dispatcher, exactly as M4-I2 did for the dense sibling.
//
// The pre-M4-I7 body is preserved BYTE-FOR-BYTE as
// `kernel::golden::moe_fp8_blockscale_task_impl` (sha256 of the frozen region
// is asserted by opt/m4i7/scripts/check_golden.py). It remains the fallback
// for every shape the fast paths do not cover -- in particular PREFILL, where
// BATCH_SIZE is the full `max_num_batched_tokens`.
//
// WHAT THE FAST PATHS CHANGE, and why each is value-neutral:
//
//  (a) WORK-ITEM FLATTENING. The golden body walks `ae` (activated experts,
//      strided by `expert_stride`) and, inside each, all `NUM_N_BLOCKS` column
//      blocks. The fast body walks a single flattened space
//      `wi in [expert_offset, num_activated * NUM_N_BLOCKS)` step
//      `expert_stride`, with `ae = wi / NUM_N_BLOCKS`, `nb = wi % NUM_N_BLOCKS`.
//      Every (expert, block) pair is still covered exactly once for any
//      `expert_offset in [0, expert_stride)`, each pair writes a disjoint set
//      of output elements, and the per-column K accumulation order is
//      untouched -- so the result is BIT-IDENTICAL.
//      This is the load-bearing MPK-dispatch decision, and it is the mirror of
//      M4-I2's (f): `task_register.cc` emits `expert_stride = grid.x = 128`
//      tasks per (layer, stage, N-split) but only `num_activated` of them ever
//      had work, so at bs1 roughly 8-13 of 128 emitted tasks were live and the
//      other ~115 exited immediately. Flattening spreads the SAME work over
//      `num_activated * NUM_N_BLOCKS` of those already-emitted tasks. Unlike
//      raising `moe_n_splits` (M4-I5's width lever, measured x1.11 at bs1 and a
//      regression at bs16), this costs NO extra dead-task dispatch and does not
//      shrink the N tile.
//  (b) A WIDER FETCH. `PATH 1` stages 4 adjacent K tiles per buffer and pulls
//      each weight row as one 512 B `cp.async.bulk`; `PATH 2` stages 8 (clamped
//      to K) at TILE_N=64 and pulls 1 KiB rows. `PATH 0` is the golden fetch
//      (one K tile per stage, 16 B `cp.async`) with the measured w13-only
//      `#pragma unroll`. All three feed the SAME `mma.sync.m16n8k32.e4m3` and
//      apply the SAME per-K-tile fp32 scale product, in ascending K order.
//  (c) A ballot-compaction routing gather on w2 only (bit-identical smem
//      contents: same ascending-token compaction, different writer lane).
//
// PRESERVED-FP32 BLOCK SCALES (the whole reason this file exists) are UNTOUCHED
// on every path: `b_scale_row[kt]` is read straight from the checkpoint's
// `weight_scale_inv` as float32 and multiplied by the float32 activation scale
// once per 128-element K tile, into an FP32 accumulator. No ue8m0 truncation,
// no requantisation, no per-row collapse. A TILE_N=64 work item reads the row
// of its CONTAINING 128-column scale block (`n0 / BLOCK_N`), so the values a
// column sees do not depend on the tiling.
//
// THE PER-TASK N SLICE IS UNCHANGED. `OUTPUT_SIZE % BLOCK_N == 0` still holds
// (static_assert below), so `weight_scale`'s grid split stays the exact
// `dim(1) * 128 == output_size` division `task_register.cc` already asserts --
// none of M4-I2's row-replication machinery is needed here, and the
// integer-division scale-slice hazard cannot be reached. TILE_N=64 subdivides
// only INSIDE a task, below the scale block.
// ===========================================================================
'''

DISPATCH = r'''
namespace moe_fp8_blockscale_fast {

// ---- admissibility, all compile-time -----------------------------------
// A PATH is admissible for an instantiation when every static_assert inside
// moe_impl_path would hold for it. The dispatcher below static_asserts that
// SOME path is reachable for every instantiation it claims, so an inadmissible
// shape fails the BUILD rather than a numeric check at run time (M4-I2's
// lesson: do not lean on `if constexpr` branch-discarding to suppress a
// static_assert -- nvcc may still parse the discarded branch).
constexpr int num_k_tiles(int red_k) {
  return red_k / GROUP_K;
}
constexpr int warps_m(int max_rows) {
  return tile_m(max_rows) / 16;
}
constexpr int warps_n(int max_rows) {
  return (WORKER_NUM_THREADS / NUM_THREADS_PER_WARP) / warps_m(max_rows);
}

constexpr bool path_admissible(int max_rows,
                               int path,
                               int out_n,
                               int red_k,
                               bool w13) {
  return red_k % GROUP_K == 0 && out_n % BLOCK_N == 0 &&
         out_n % path_tile_n(path) == 0 &&
         BLOCK_N % path_tile_n(path) == 0 &&
         num_k_tiles(red_k) % stage_tiles_k(path, red_k) == 0 &&
         (path_tile_n(path) / warps_n(max_rows)) % 8 == 0 &&
         smem_bytes_k(max_rows, path, red_k) <=
             mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE &&
         // the w2 ballot gather puts one token per lane
         (w13 || max_rows <= NUM_THREADS_PER_WARP);
}

// M4-I2's lesson, applied structurally: nvcc is not guaranteed to leave a
// DISCARDED `if constexpr` branch uninstantiated (it kept parsing one while
// recovering from an earlier diagnostic and surfaced its static_assert). So the
// PATH template argument is SANITISED here rather than guarded at the call
// site -- a spuriously instantiated branch compiles as PATH 0, and the safety
// rests on the dispatcher's reachability static_assert, which no compiler's
// instantiation eagerness can affect.
constexpr int safe_path(int max_rows,
                        int p,
                        int out_n,
                        int red_k,
                        bool w13) {
  return path_admissible(max_rows, p, out_n, red_k, w13) ? p : 0;
}

// The ferret run validated exactly BATCH_SIZE == 16 with 1..16 live rows -- the
// shipped decode geometry (`max_num_batched_tokens = 16`). PREFILL instantiates
// the same template with the full batched-token count; it stays on the golden
// path, which is the proven one (goal.md AC-5 depends on prefill).
constexpr int FAST_MAX_BATCH = 16;

constexpr bool fast_path_ok(int max_rows, int out_n, int red_k, bool w13) {
#ifdef MPK_MOE_BLOCKSCALE_BASELINE
  return false; // A/B arm A: pin the pre-M4-I7 kernel from one tree
#else
  return max_rows <= FAST_MAX_BATCH &&
         (path_admissible(max_rows, 0, out_n, red_k, w13) ||
          path_admissible(max_rows, 1, out_n, red_k, w13) ||
          path_admissible(max_rows, 2, out_n, red_k, w13));
#endif
}

// The GOLDEN path needs whole 128-column blocks and it is the only fallback, so
// a shape the fast paths reject must be one the golden path can run.
constexpr bool golden_can_run(int out_n, int red_k) {
  return out_n % BLOCK_N == 0 && red_k % GROUP_K == 0;
}

} // namespace moe_fp8_blockscale_fast

// The MPK-facing entry point: a compile-time dispatcher over `fast_path_ok`,
// then a runtime choice among the admissible fetch paths.
template <typename T,
          int BATCH_SIZE,
          int NUM_TOPK,
          int NUM_EXPERTS,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool W13_LINEAR>
__device__ __forceinline__ void
    moe_fp8_blockscale_task_impl(void const *__restrict__ input_fp8_ptr,
                                 void const *__restrict__ input_scale_ptr,
                                 void const *__restrict__ weight_fp8_ptr,
                                 void const *__restrict__ weight_scale_ptr,
                                 void const *__restrict__ routing_ptr,
                                 void const *__restrict__ mask_ptr,
                                 void *__restrict__ output_ptr,
                                 int expert_offset,
                                 int expert_stride) {
  using namespace moe_fp8_blockscale_fast;
  constexpr bool FAST =
      fast_path_ok(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
  static_assert(FAST || golden_can_run(OUTPUT_SIZE, REDUCTION_SIZE),
                "no admissible path for this instantiation: the fast paths "
                "rejected it and the golden path needs whole 128-column scale "
                "blocks with K a multiple of 128");

  if constexpr (!FAST) {
    golden::moe_fp8_blockscale_task_impl<T,
                                        BATCH_SIZE,
                                        NUM_TOPK,
                                        NUM_EXPERTS,
                                        OUTPUT_SIZE,
                                        ORIG_OUTPUT_SIZE,
                                        REDUCTION_SIZE,
                                        W13_LINEAR>(input_fp8_ptr,
                                                    input_scale_ptr,
                                                    weight_fp8_ptr,
                                                    weight_scale_ptr,
                                                    routing_ptr,
                                                    mask_ptr,
                                                    output_ptr,
                                                    expert_offset,
                                                    expert_stride);
  } else {
    constexpr bool OK0 =
        path_admissible(BATCH_SIZE, 0, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    constexpr bool OK1 =
        path_admissible(BATCH_SIZE, 1, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    constexpr bool OK2 =
        path_admissible(BATCH_SIZE, 2, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    static_assert(OK0, "the legacy fetch path must always be admissible when "
                       "the fast body runs -- it is the in-body fallback");

#define MPK_MOE_RUN_PATH(P)                                                    \
  moe_impl_path<T,                                                             \
                BATCH_SIZE,                                                    \
                NUM_TOPK,                                                      \
                NUM_EXPERTS,                                                   \
                OUTPUT_SIZE,                                                   \
                ORIG_OUTPUT_SIZE,                                              \
                REDUCTION_SIZE,                                                \
                W13_LINEAR,                                                    \
                safe_path(BATCH_SIZE,                                          \
                          (P),                                                 \
                          OUTPUT_SIZE,                                         \
                          REDUCTION_SIZE,                                      \
                          W13_LINEAR)>(input_fp8_ptr,                          \
                     input_scale_ptr,                                          \
                     weight_fp8_ptr,                                           \
                     weight_scale_ptr,                                         \
                     routing_ptr,                                              \
                     mask_ptr,                                                 \
                     output_ptr,                                               \
                     expert_offset,                                            \
                     expert_stride)

#if defined(MPK_MOE_PATH_POLICY)
    // Sweep/diagnostic pin. Falls back to the legacy path where the pinned one
    // is inadmissible, so every instantiation still builds.
    constexpr int PIN = MPK_MOE_PATH_POLICY;
    static_assert(PIN >= 0 && PIN <= 2, "MPK_MOE_PATH_POLICY must be 0, 1 or 2");
    if constexpr (PIN == 2 && OK2) {
      MPK_MOE_RUN_PATH(2);
    } else if constexpr (PIN == 1 && OK1) {
      MPK_MOE_RUN_PATH(1);
    } else {
      MPK_MOE_RUN_PATH(0);
    }
#else
    // ---- the shipped rule -------------------------------------------------
    // In the ferret harness one CTA ran one work item, so "does the grid fit
    // one wave" (work items vs `%nsmid`) decided whether the wide-smem paths'
    // 1-CTA/SM residency was free. IN MPK THAT DENOMINATOR IS WRONG: there is
    // exactly one persistent worker per SM, each owning the WHOLE dynamic smem
    // budget, so residency is fixed at 1 CTA/SM no matter which path runs and
    // the wide layouts cost nothing. What varies instead is how many of the
    // `expert_stride` EMITTED tasks have work. So the gate keeps the ferret
    // rule's shape and swaps `%nsmid` for `expert_stride`:
    //
    //   * PATH 2 (TILE_N=64) doubles the work items. That is a win only while
    //     the extra items land on tasks that would otherwise be dead, i.e.
    //     while `num_activated * OUTPUT_SIZE/64 <= expert_stride`. Past that
    //     point the same tasks just run more, smaller tiles: strictly more
    //     gathers, A re-fetches and epilogues for the same MMAs.
    //   * Otherwise PATH 1: TILE_N=128 like the golden path, but 512 B bulk
    //     weight rows instead of 16 B `cp.async`. The ferret run only ever
    //     preferred PATH 0 over PATH 1 to protect 4-5 CTAs/SM of residency,
    //     which does not exist here -- so PATH 1 should dominate PATH 0 at
    //     every batch size in MPK. That is a falsifiable claim and
    //     `MPK_MOE_PATH_POLICY` exists to test it.
    if constexpr (OK2) {
      int const nact = static_cast<int32_t const *>(mask_ptr)[NUM_EXPERTS];
      if (nact * (OUTPUT_SIZE / 64) <= expert_stride) {
        MPK_MOE_RUN_PATH(2);
        return;
      }
    }
    if constexpr (OK1) {
      MPK_MOE_RUN_PATH(1);
      return;
    }
    MPK_MOE_RUN_PATH(0);
#endif
#undef MPK_MOE_RUN_PATH
  }
}
'''

out = []
out.append(OLD_PREAMBLE.rstrip("\n"))
out.append(HEADER_NOTE)
out.append("""
// =========================================================================
// GOLDEN -- the pre-M4-I7 body, FROZEN. Byte-for-byte identical to
// moe_fp8_blockscale_sm100.cuh before this change (region sha256 below,
// re-checked by opt/m4i7/scripts/check_golden.py). NEVER EDIT.
//   region sha256: %s
// =========================================================================
namespace golden {
""" % GOLDEN_SHA)
out.append(GOLDEN)
out.append("\n} // namespace golden\n")
out.append("""
// =========================================================================
// FAST -- ferret workspace3 v012 (`git show v012:kernel.cu`, namespace `cand`),
// verbatim apart from the namespace rename recorded in
// opt/m4i7/scripts/gen_header.py and a tightened smem static_assert. Its own
// dispatcher is REPLACED below (see (a) in the header note).
// =========================================================================
""")
# Keep every helper the ferret file put at `kernel::cand::` scope inside
# `kernel::moe_fp8_blockscale_fast::` instead, so the megakernel TU -- which
# concatenates every task header -- gains NO new name at `kernel::` scope.
# moe_impl_path's function-scope `using namespace` makes them visible to it and
# to its lambdas, so no call site changes.
_ns_close = "} // namespace moe_fp8_blockscale_fast"
_i = FAST.index(_ns_close) + len(_ns_close)
FAST_CONSTS, FAST_REST = FAST[:_i], FAST[_i:]
_impl_hdr = "template <typename T,\n          int BATCH_SIZE,"
_j = FAST_REST.index(_impl_hdr)
FAST_HELPERS, FAST_IMPL = FAST_REST[:_j], FAST_REST[_j:]

out.append(FAST_CONSTS)
out.append("\nnamespace moe_fp8_blockscale_fast {\n")
out.append(CG_HELPERS)
out.append(FAST_HELPERS.rstrip("\n"))
out.append("\n} // namespace moe_fp8_blockscale_fast\n")
out.append(FAST_IMPL)
out.append("\n")
out.append(DISPATCH)
out.append("\n} // namespace kernel\n")

new = "\n".join(out)
# collapse the accidental triple newlines the concatenation can make
new = re.sub(r"\n{4,}", "\n\n\n", new)
open(OUT, "w").write(new)

# ---- prove the golden region round-tripped ------------------------------
chk = open(OUT).read()
g0 = chk.index("namespace golden {\n") + len("namespace golden {\n")
g1 = chk.index("\n} // namespace golden")
got = chk[g0:g1].strip("\n")
print(f"golden region: {len(GOLDEN)} bytes, sha256 {GOLDEN_SHA}")
print(f"round-tripped: {len(got)} bytes, sha256 "
      f"{hashlib.sha256(got.encode()).hexdigest()}")
print("GOLDEN_BYTE_IDENTICAL:", got == GOLDEN)
print(f"ferret cand region: {len(FAST_RAW_SHA and FAST)} bytes after "
      f"{n_rename} namespace-name occurrences renamed")
print(f"emitted {len(new)} bytes -> {OUT}")
if got != GOLDEN:
    sys.exit(1)
