"""
Multi-GPU EP MoE unit test (WORLD_SIZE=2, NVLink, NVSHMEM).

Replaces raw peer-access (cudaDeviceEnablePeerAccess) with NVSHMEM:
  - recv_buf: symmetric memory allocated via nvshmem_malloc (same logical
    address on all PEs; NVSHMEM routes puts to the correct GPU via NVLink)
  - sync_sigs: symmetric uint64_t[WS] signal array; dispatch sets
    sync_sigs[rank]=1 via nvshmem_uint64_p; combine waits via
    nvshmem_signal_wait_until (hardware-accelerated on B200 NVLink)

Pipeline stages tested:
  1. Routing  — each GPU: TopK + softmax, verified vs torch.topk
  2. Dispatch — each GPU: NVSHMEM puts to remote recv_buf; sync_sigs set
  3. Combine  — each GPU: weighted sum + residual add; verified vs torch ref

Each stage runs in a separate worker process (mp.spawn, one per GPU).

Usage:
    conda run -n mpk python test_ep_moe_multigpu.py
"""

import sys
import time
import torch
import torch.multiprocessing as mp

# ── Config ────────────────────────────────────────────────────────────────────
WORLD_SIZE       = 2
BATCH_SIZE       = 8          # tokens per GPU
NUM_EXPERTS      = 4          # total experts; 2 per GPU
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE
TOPK             = 2
HIDDEN_DIM       = 64

# recv_buf: each GPU must hold all tokens that may be sent to it.
# In the worst case that's WORLD_SIZE * BATCH_SIZE * TOPK tokens.
RECV_BUF_TOKENS = WORLD_SIZE * BATCH_SIZE * TOPK


# ── Worker: Test 1 — Routing ─────────────────────────────────────────────────

def _worker_routing(rank, world_size, uid):
    """Test that each GPU's routing kernel matches torch.topk + softmax."""
    import runtime_kernel as rk

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"
    logits  = torch.randn(BATCH_SIZE, NUM_EXPERTS, dtype=torch.bfloat16, device=dev)
    indices = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.int32, device=dev)
    weights = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.bfloat16, device=dev)
    dcounts = torch.zeros(WORLD_SIZE,  dtype=torch.int32, device=dev)

    rk.moe_routing_dev(rank, logits, indices, weights, dcounts,
                       EXPERTS_PER_RANK, rank, 0.0)

    # PyTorch reference: topk + softmax
    scores, ref_idx = torch.topk(logits.float(), TOPK, dim=1)
    ref_weights     = torch.softmax(scores, dim=1).bfloat16()

    # Sort by expert index before comparing (topk ordering may differ on ties).
    ord_k  = torch.argsort(indices,  dim=1)
    ord_r  = torch.argsort(ref_idx,  dim=1)
    s_idx  = torch.gather(indices,       1, ord_k)
    s_wts  = torch.gather(weights,       1, ord_k)
    s_ridx = torch.gather(ref_idx.int(), 1, ord_r)
    s_rwts = torch.gather(ref_weights,   1, ord_r)

    assert torch.all(s_idx == s_ridx), \
        f"GPU {rank}: routing indices mismatch\n  kernel={s_idx}\n  ref={s_ridx}"
    torch.testing.assert_close(s_wts, s_rwts, rtol=1e-2, atol=1e-2)

    total = dcounts.sum().item()
    assert total == BATCH_SIZE * TOPK, \
        f"GPU {rank}: dispatch_counts sum {total} != {BATCH_SIZE * TOPK}"

    print(f"  GPU {rank}: routing OK  dispatch_counts={dcounts.tolist()}", flush=True)
    rk.nvshmem_finalize()


# ── Worker: Test 2 — Dispatch ─────────────────────────────────────────────────

def _worker_dispatch(rank, world_size, uid):
    """
    Verify the NVSHMEM dispatch kernel:
      * send_counts[r] == number of (token, expert) pairs sent to rank r
      * sync_sigs[rank] == 1 after dispatch completes (set on ALL PEs)
      * recv_buf contains tokens from other ranks (NVLink NVSHMEM put)

    Fixed routing so results are predictable:
      GPU 0  tokens 0-3  → experts {2,3}  (rank 1)
      GPU 0  tokens 4-7  → experts {0,1}  (rank 0, local)
      GPU 1  tokens 0-3  → experts {0,1}  (rank 0)
      GPU 1  tokens 4-7  → experts {2,3}  (rank 1, local)
    """
    import runtime_kernel as rk

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"

    # Allocate symmetric NVSHMEM memory (same logical address on all PEs).
    recv_buf_bytes  = RECV_BUF_TOKENS * HIDDEN_DIM * 2   # bfloat16
    sync_sigs_bytes = WORLD_SIZE * 8                      # uint64_t

    recv_buf_ptr  = rk.nvshmem_malloc(recv_buf_bytes)
    sync_sigs_ptr = rk.nvshmem_malloc(sync_sigs_bytes)

    # Zero-initialise both buffers on this PE.
    rk.nvshmem_memset_zero(recv_buf_ptr,  recv_buf_bytes)
    rk.nvshmem_memset_zero(sync_sigs_ptr, sync_sigs_bytes)

    # Barrier: ensure all PEs have initialised before dispatch.
    rk.nvshmem_barrier_all()

    # Known token values: GPU r, token t → all hidden dims = float(r*1000+t)
    tok = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    for t in range(BATCH_SIZE):
        tok[t] = float(rank * 1000 + t)

    # Fixed routing: first 4 tokens → remote rank, last 4 → local rank.
    idx = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.int32, device=dev)
    for t in range(BATCH_SIZE):
        goes_remote = (rank == 0 and t < 4) or (rank == 1 and t >= 4)
        if goes_remote:
            idx[t, 0] = 2;  idx[t, 1] = 3   # rank 1's experts
        else:
            idx[t, 0] = 0;  idx[t, 1] = 1   # rank 0's experts

    send_counts  = torch.zeros(WORLD_SIZE, dtype=torch.int32, device=dev)
    send_offsets = torch.zeros(WORLD_SIZE, dtype=torch.int32, device=dev)

    rk.moe_dispatch_dev(
        rank, tok, idx,
        send_counts, send_offsets,
        recv_buf_ptr, sync_sigs_ptr,
        NUM_EXPERTS, EXPERTS_PER_RANK, rank)

    # ── Verify send_counts ────────────────────────────────────────────────────
    expected_sc = [4 * TOPK, 4 * TOPK]   # [8, 8] for the fixed routing above
    got_sc = send_counts.tolist()
    assert got_sc == expected_sc, \
        f"GPU {rank}: send_counts {got_sc} != {expected_sc}"
    print(f"  GPU {rank}: send_counts={got_sc}  OK", flush=True)

    # ── Verify sync_sigs[rank] was set to 1 on this PE ───────────────────────
    # nvshmem_uint64_p writes sync_sigs[rank]=1 on EVERY PE.
    # Use cudaMemcpy helper (sync_sigs_ptr is a GPU device pointer).
    sigs = rk.nvshmem_read_uint64s(sync_sigs_ptr, WORLD_SIZE)
    local_sig = sigs[rank]
    assert local_sig == 1, \
        f"GPU {rank}: sync_sigs[{rank}]={local_sig} expected 1"
    print(f"  GPU {rank}: sync_sigs[{rank}]={local_sig}  OK", flush=True)

    # ── Spot-check recv_buf ────────────────────────────────────────────────────
    # After dispatch, remote GPU should have written some tokens here.
    # Use cudaMemcpy helper (recv_buf_ptr is a GPU device pointer).
    nonzero = rk.nvshmem_has_nonzero_bf16(recv_buf_ptr, recv_buf_bytes)
    print(f"  GPU {rank}: recv_buf has nonzero data: {nonzero}  "
          f"(may be False if no remote tokens arrived at this PE)", flush=True)

    rk.nvshmem_free(recv_buf_ptr)
    rk.nvshmem_free(sync_sigs_ptr)
    rk.nvshmem_finalize()


# ── Worker: Test 3 — Combine ─────────────────────────────────────────────────

def _worker_combine(rank, world_size, uid):
    """
    Verify combine output against PyTorch reference.
    sync_sigs is pre-set to 1 (simulates completed dispatch) so the wait is
    a no-op.
    """
    import runtime_kernel as rk

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"

    # Allocate symmetric sync_sigs and pre-set to 1 (dispatch already done).
    sync_sigs_bytes = WORLD_SIZE * 8
    sync_sigs_ptr = rk.nvshmem_malloc(sync_sigs_bytes)

    # Fill all entries with 1 so combine doesn't wait.
    # (sync_sigs_ptr is a GPU device pointer; use H2D helper instead of ctypes.)
    rk.nvshmem_fill_uint64(sync_sigs_ptr, WORLD_SIZE, 1)

    rk.nvshmem_barrier_all()   # ensure all PEs are ready

    eo  = torch.randn(BATCH_SIZE, TOPK, HIDDEN_DIM,
                      dtype=torch.bfloat16, device=dev)
    idx = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK),
                        dtype=torch.int32, device=dev)
    wf  = torch.rand(BATCH_SIZE, TOPK, dtype=torch.float32, device=dev)
    wts = (wf / wf.sum(dim=1, keepdim=True)).bfloat16()
    res = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    rc  = torch.tensor([BATCH_SIZE // 2, BATCH_SIZE // 2],
                       dtype=torch.int32, device=dev)
    ro  = torch.tensor([0, BATCH_SIZE // 2], dtype=torch.int32, device=dev)

    rk.moe_combine_dev(
        rank, eo, idx, wts, res, out, rc, ro,
        sync_sigs_ptr, NUM_EXPERTS, EXPERTS_PER_RANK, rank)

    # PyTorch reference
    ref = (eo.float() * wts.unsqueeze(-1).float()).sum(dim=1).bfloat16() + res
    max_err = (out.float() - ref.float()).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print(f"  GPU {rank}: combine OK  max_err={max_err:.4f}", flush=True)

    rk.nvshmem_free(sync_sigs_ptr)
    rk.nvshmem_finalize()


# ── Worker: Benchmark ─────────────────────────────────────────────────────────

def _worker_benchmark_routing(rank, world_size, uid, n_iters=200):
    import runtime_kernel as rk

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"
    logits  = torch.randn(BATCH_SIZE, NUM_EXPERTS, dtype=torch.bfloat16, device=dev)
    indices = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.int32, device=dev)
    weights = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.bfloat16, device=dev)
    dcounts = torch.zeros(WORLD_SIZE, dtype=torch.int32, device=dev)

    def run():
        rk.moe_routing_dev(rank, logits, indices, weights, dcounts,
                           EXPERTS_PER_RANK, rank, 0.0)

    for _ in range(10): run()
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(n_iters): run()
    torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / n_iters * 1e3
    if rank == 0:
        print(f"  routing: {ms:.3f} ms / iter  ({n_iters} iters)", flush=True)

    rk.nvshmem_finalize()


def _worker_benchmark_dispatch(rank, world_size, uid, n_iters=50):
    import runtime_kernel as rk
    import ctypes

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"

    recv_buf_bytes  = RECV_BUF_TOKENS * HIDDEN_DIM * 2
    sync_sigs_bytes = WORLD_SIZE * 8

    recv_buf_ptr  = rk.nvshmem_malloc(recv_buf_bytes)
    sync_sigs_ptr = rk.nvshmem_malloc(sync_sigs_bytes)

    tok = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    idx = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK),
                        dtype=torch.int32, device=dev)
    sc  = torch.zeros(WORLD_SIZE, dtype=torch.int32, device=dev)
    so  = torch.zeros(WORLD_SIZE, dtype=torch.int32, device=dev)

    rk.nvshmem_barrier_all()

    def run():
        rk.nvshmem_memset_zero(sync_sigs_ptr, sync_sigs_bytes)
        rk.moe_dispatch_dev(rank, tok, idx, sc, so,
                            recv_buf_ptr, sync_sigs_ptr,
                            NUM_EXPERTS, EXPERTS_PER_RANK, rank)

    for _ in range(5): run()
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(n_iters): run()
    torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / n_iters * 1e3
    if rank == 0:
        print(f"  dispatch: {ms:.3f} ms / iter  ({n_iters} iters)", flush=True)

    rk.nvshmem_free(recv_buf_ptr)
    rk.nvshmem_free(sync_sigs_ptr)
    rk.nvshmem_finalize()


def _worker_benchmark_combine(rank, world_size, uid, n_iters=200):
    import runtime_kernel as rk

    torch.cuda.set_device(rank)
    rk.nvshmem_init_uid(rank, world_size, uid)

    dev = f"cuda:{rank}"

    sync_sigs_bytes = WORLD_SIZE * 8
    sync_sigs_ptr = rk.nvshmem_malloc(sync_sigs_bytes)
    rk.nvshmem_fill_uint64(sync_sigs_ptr, WORLD_SIZE, 1)

    rk.nvshmem_barrier_all()

    eo  = torch.randn(BATCH_SIZE, TOPK, HIDDEN_DIM,
                      dtype=torch.bfloat16, device=dev)
    idx = torch.zeros(BATCH_SIZE, TOPK, dtype=torch.int32, device=dev)
    wts = torch.full((BATCH_SIZE, TOPK), 0.5, dtype=torch.bfloat16, device=dev)
    res = torch.randn(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(BATCH_SIZE, HIDDEN_DIM, dtype=torch.bfloat16, device=dev)
    rc  = torch.tensor([BATCH_SIZE // 2, BATCH_SIZE // 2],
                       dtype=torch.int32, device=dev)
    ro  = torch.tensor([0, BATCH_SIZE // 2], dtype=torch.int32, device=dev)

    def run():
        rk.nvshmem_fill_uint64(sync_sigs_ptr, WORLD_SIZE, 1)
        rk.moe_combine_dev(rank, eo, idx, wts, res, out, rc, ro,
                           sync_sigs_ptr, NUM_EXPERTS, EXPERTS_PER_RANK, rank)

    for _ in range(10): run()
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(n_iters): run()
    torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / n_iters * 1e3
    if rank == 0:
        print(f"  combine:  {ms:.3f} ms / iter  ({n_iters} iters)", flush=True)

    rk.nvshmem_free(sync_sigs_ptr)
    rk.nvshmem_finalize()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import runtime_kernel as rk

    n_gpu = torch.cuda.device_count()
    if n_gpu < WORLD_SIZE:
        print(f"Need {WORLD_SIZE} GPUs, found {n_gpu}. Skipping.")
        sys.exit(0)

    print(f"EP MoE NVSHMEM multi-GPU test  |  "
          f"WORLD_SIZE={WORLD_SIZE}  BATCH={BATCH_SIZE}  "
          f"EXPERTS={NUM_EXPERTS}  TOPK={TOPK}  HIDDEN={HIDDEN_DIM}")
    for r in range(WORLD_SIZE):
        prop = torch.cuda.get_device_properties(r)
        print(f"  GPU {r}: {prop.name}  ({prop.total_memory // 2**30} GiB)")
    print()

    # Get NVSHMEM unique ID once in main process; share with all workers.
    uid = rk.nvshmem_get_uniqueid()

    # ── Correctness tests ─────────────────────────────────────────────────────
    print("=== Test 1: Multi-GPU Routing (NVSHMEM init) ===")
    mp.spawn(_worker_routing,  args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)
    print()

    print("=== Test 2: Multi-GPU Dispatch (NVSHMEM puts + signals) ===")
    uid = rk.nvshmem_get_uniqueid()   # fresh uid for each spawn
    mp.spawn(_worker_dispatch, args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)
    print()

    print("=== Test 3: Multi-GPU Combine (NVSHMEM signal wait) ===")
    uid = rk.nvshmem_get_uniqueid()
    mp.spawn(_worker_combine,  args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)
    print()

    print("All correctness tests passed!\n")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print("=== Benchmark: Routing ===")
    uid = rk.nvshmem_get_uniqueid()
    mp.spawn(_worker_benchmark_routing,  args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)
    print()

    print("=== Benchmark: Dispatch ===")
    uid = rk.nvshmem_get_uniqueid()
    mp.spawn(_worker_benchmark_dispatch, args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)
    print()

    print("=== Benchmark: Combine ===")
    uid = rk.nvshmem_get_uniqueid()
    mp.spawn(_worker_benchmark_combine,  args=(WORLD_SIZE, uid), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
