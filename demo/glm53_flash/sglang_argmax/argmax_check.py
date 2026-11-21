"""lm_argmax.cu against torch.argmax on N GPUs (no SGLang): one process per GPU, a NCCL group, the exchange and launch of
flash_argmax_hook.py. Every rank makes the same full logits [rows][N x W] bf16 (seeded) and passes its shard [rows][W]
(columns rank x W ..); every rank's ids must equal torch.argmax of the full logits (fp32 of the bf16 values; columns >= V
dropped first) in every case:
  random rows 64 / 1 / 3 / 17 / 2 / 64 ... (the launch counts and the slot parities of rows 0..63 advance unevenly), ties
  of the row maximum across ranks (-> the lowest index), all-equal rows, NaN in two ranks' shards (-> the first NaN), -inf
  rows, -0 / +0 ties, vocab padding (V < N x W: larger values past V ignored), the logit scale (bf16(x * scale)), and a CUDA
  graph of one launch replayed with new data each time (the decode graphs' use).
Then a rough kernel time at rows 64: per launch, L2 flushed (a 512 MB write) and the ranks aligned (barrier), CUDA events
around the launch, median of 30.

python argmax_check.py [--gpus 4] [--width 38720] [--vocab 154880]   -> last line ARGMAX-CHECK PASS / FAIL
"""
import argparse
import os
import socket
import statistics
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flash_argmax_hook as fa  # noqa: E402


def _cases(world, W, V):
    """(name, rows, vocab, scale, make(gen) -> full [rows][world * W] fp32 holding bf16-exact values)."""
    N = world * W

    def rnd(rows):
        return lambda g: (torch.randn(rows, N, generator=g) * 3).bfloat16().float()

    def ties(g):   # the maximum at a column of the last rank and of rank 1 (and of rank 0 in row 1): the lowest wins
        x = rnd(8)(g)
        x[:, (world - 1) * W + 5] = 100.0
        x[:, 1 * W + 7] = 100.0
        x[1, 3] = 100.0
        return x

    def equal(g):
        return torch.ones(4, N)

    def nan(g):   # NaN in the last rank's shard and in rank 1's: the first NaN (rank 1's) wins; row 1: NaN at column 0
        x = rnd(4)(g)
        x[:, (world - 1) * W + 11] = float("nan")
        x[:, 1 * W + 13] = float("nan")
        x[:, 2 * W] = 1e4
        x[1, 0] = float("nan")
        return x

    def neginf(g):
        x = torch.full((3, N), float("-inf"))
        x[1, 2 * W + 9] = -1e30   # bf16 -1e30 > -inf
        return x

    def zeros(g):   # -0 at the lower column, +0 at the higher: equal values, the lower index
        x = torch.full((2, N), -5.0)
        x[:, 1 * W + 3] = -0.0
        x[:, 0 * W + 2] = 0.0
        x[1, 0 * W + 1] = -0.0
        return x

    def padded(g):   # vocab V2 < N: columns >= V2 hold larger values, ignored
        x = rnd(16)(g)
        x[:, V - 37:] = 1e4
        return x

    out = [("random", r, V, None, rnd(r)) for r in (64, 1, 3, 64, 17, 2, 64, 64, 33, 64)]
    out += [("ties", 8, V, None, ties), ("equal", 4, V, None, equal), ("nan", 4, V, None, nan),
            ("neginf", 3, V, None, neginf), ("zeros", 2, V, None, zeros), ("padded", 16, V - 37, None, padded),
            ("scale", 64, V, 0.37, rnd(64))]
    return out


def _ref(full, vocab, scale, dev):
    """torch.argmax on the GPU (as SGLang's sampler) of the fp32 logits SGLang would sample; on the CPU as int64."""
    x = full[:, :vocab].to(dev)
    if scale is not None:   # logits.mul_(scale) on bf16: fp32 product rounded to bf16
        x = (x * scale).bfloat16().float()
    return torch.argmax(x, -1).cpu()


def _worker(rank, world, port, W, V, fails):
    try:
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, device_id=dev)
        group = dist.group.WORLD
        lib = fa.load(dev)
        maxr = lib.flash_lm_argmax_max_rows()
        ex = fa.make_exchange(int(lib.flash_lm_argmax_slot_bytes()), group, dev)
        cnt = torch.zeros(maxr, dtype=torch.int32, device=dev)
        bad = 0

        def check(name, ids, ref):
            nonlocal bad
            ok = torch.equal(ids.cpu(), ref)
            if not ok:
                bad += 1
                d = (ids.cpu() != ref).nonzero().flatten()[:4].tolist()
                print(f"rank {rank} {name}: MISMATCH rows {d}: ids {ids.cpu()[d].tolist()} ref {ref[d].tolist()}", flush=True)
            elif rank == 0:
                print(f"rank 0 {name}: OK ({ids.numel()} rows)", flush=True)

        for i, (name, rows, vocab, scale, make) in enumerate(_cases(world, W, V)):
            g = torch.Generator().manual_seed(1234 + i)
            full = make(g).bfloat16().float()   # the values the kernel sees (bf16), as fp32
            x = full[:, rank * W:(rank + 1) * W].bfloat16().contiguous().to(dev)
            ids = torch.full((rows,), -1, dtype=torch.int64, device=dev)
            fa.launch(lib, x, rows, rank * W, vocab, scale, ex, rank, world, cnt, ids)
            torch.cuda.synchronize(dev)
            check(f"case {i} {name} rows {rows}", ids, _ref(full, vocab, scale, dev))

        # a CUDA graph of one launch (rows 64), replayed with new shard data each time
        xs = torch.zeros(maxr, W, dtype=torch.bfloat16, device=dev)
        gids = torch.full((maxr,), -1, dtype=torch.int64, device=dev)
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
            fa.launch(lib, xs, maxr, rank * W, V, None, ex, rank, world, cnt, gids)
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        dist.barrier(group=group)
        for rep in range(6):
            g = torch.Generator().manual_seed(777 + rep)
            full = (torch.randn(maxr, world * W, generator=g) * 3).bfloat16().float()
            xs.copy_(full[:, rank * W:(rank + 1) * W].bfloat16())
            graph.replay()
            torch.cuda.synchronize(dev)
            check(f"graph replay {rep}", gids, _ref(full, V, None, dev))
            if rep % 2:   # an eager launch of another row count between replays (the decode steps of other batch sizes)
                ids = torch.full((5,), -1, dtype=torch.int64, device=dev)
                fa.launch(lib, xs, 5, rank * W, V, None, ex, rank, world, cnt, ids)
                torch.cuda.synchronize(dev)
                check(f"eager rows 5 after replay {rep}", ids, _ref(full[:5], V, None, dev))

        # rough time of one launch at rows 64 (L2 flushed, ranks aligned)
        flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        times = []
        for _ in range(30):
            flush.fill_(1)
            torch.cuda.synchronize(dev)
            dist.barrier(group=group)
            torch.cuda.synchronize(dev)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fa.launch(lib, xs, maxr, rank * W, V, None, ex, rank, world, cnt, gids)
            b.record()
            torch.cuda.synchronize(dev)
            times.append(a.elapsed_time(b) * 1000.0)
        print(f"rank {rank} time rows {maxr} W {W}: median {statistics.median(times):.1f} us "
              f"(min {min(times):.1f}, max {max(times):.1f}; L2 flushed, after a barrier)", flush=True)
        dist.barrier(group=group)
        dist.destroy_process_group()
        if bad:
            fails[rank] = bad
    except Exception:
        traceback.print_exc()
        fails[rank] = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--width", type=int, default=38720, help="columns per rank (GLM-5.3-Flash TP4: 154880 / 4)")
    ap.add_argument("--vocab", type=int, default=154880)
    args = ap.parse_args()
    if torch.cuda.device_count() < args.gpus:
        print(f"ARGMAX-CHECK FAIL: {torch.cuda.device_count()} GPUs visible, {args.gpus} needed")
        sys.exit(1)
    print("lib", fa.build(0, verbose=True), flush=True)   # once, before the ranks start
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    fails = mp.Manager().dict()
    mp.start_processes(_worker, args=(args.gpus, port, args.width, args.vocab, fails), nprocs=args.gpus, join=True,
                       start_method="spawn")
    if fails:
        print(f"ARGMAX-CHECK FAIL: {dict(fails)} (rank: mismatches, -1 = exception)")
        sys.exit(1)
    print(f"ARGMAX-CHECK PASS ({args.gpus} GPUs, W {args.width}, vocab {args.vocab})")


if __name__ == "__main__":
    main()
