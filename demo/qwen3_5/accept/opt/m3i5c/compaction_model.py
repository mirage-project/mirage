#!/usr/bin/env python3
"""M3-I5c -- host model of the SM100 routers' Phase-7 active-expert compaction.

CPU-only. No CUDA, no torch, no GPU. Run: `python3 compaction_model.py`.

Why this exists: the fix lives in two `.cuh` files whose only unit tests need a
device, and the defect is a *schedule-dependent* one, so a single GPU run proves
little either way. This models the algorithm at the level the bug lives at --
individual global-memory accesses of individual threads, interleaved by an
adversarial scheduler, with `__syncthreads()` as the only ordering primitive --
and checks two things:

  1. a barrier-interval RACE DETECTOR: inside any interval between two
     block-wide barriers, no address may be written by one thread and touched
     by another. This is the exact happens-before model CUDA gives a block that
     synchronises only with `__syncthreads()`, so "no conflict reported" is a
     proof over ALL interleavings, not a sample of them;
  2. the OUTPUT under many concrete schedules (random, sequential, reverse,
     round-robin): the compacted list must equal the sorted active set and the
     count must equal its size.

The OLD algorithm is included as a positive control. If the detector did not
fire on it, the detector would be worthless.

Both thread bodies below are transliterations of the CUDA, statement for
statement -- `yield ('r', a)` is a load of `mpk_active_expert_ids[a]`,
`yield ('w', a, v)` a store, `yield ('bar',)` a `__syncthreads()`.
"""

import random
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# Block simulator: threads are generators of memory ops; the scheduler picks
# which thread advances. Only `__syncthreads()` orders anything.
# --------------------------------------------------------------------------

RUNNABLE, AT_BARRIER, DONE = 0, 1, 2


class Conflict(object):
    def __init__(self, interval, addr, w_tids, other_tids, kind):
        self.interval, self.addr = interval, addr
        self.w_tids, self.other_tids, self.kind = w_tids, other_tids, kind

    def __str__(self):
        return "interval %d: %s on slot %d (writers %s, others %s)" % (
            self.interval, self.kind, self.addr,
            sorted(self.w_tids), sorted(self.other_tids))


class Block(object):
    """Memory + per-barrier-interval access log."""

    def __init__(self, mem):
        self.mem = list(mem)
        self.n_slots = len(mem)
        self.oob = []
        self.interval = 0
        self._reads = defaultdict(set)    # addr -> tids that loaded it
        self._writes = defaultdict(set)   # addr -> tids that stored to it
        self._atomics = defaultdict(set)  # addr -> tids that did an RMW
        self.conflicts = []

    # ---- memory ops -------------------------------------------------------
    def load(self, tid, addr):
        if not 0 <= addr < self.n_slots:
            self.oob.append(("load", tid, addr))
            return 0
        self._reads[addr].add(tid)
        return self.mem[addr]

    def store(self, tid, addr, val):
        if not 0 <= addr < self.n_slots:
            self.oob.append(("store", tid, addr))
            return
        self._writes[addr].add(tid)
        self.mem[addr] = val

    def atomic_add(self, tid, addr, delta):
        if not 0 <= addr < self.n_slots:
            self.oob.append(("atomic", tid, addr))
            return 0
        self._atomics[addr].add(tid)
        old = self.mem[addr]
        self.mem[addr] = old + delta
        return old

    # ---- barrier ----------------------------------------------------------
    def close_interval(self):
        """Flag every unsynchronised conflicting pair in the interval that just
        ended. Two accesses conflict when they are by DIFFERENT threads, at
        least one is a plain store, and nothing ordered them. An atomic RMW is
        exempt against other atomics on the same address but not against plain
        accesses."""
        addrs = set(self._writes) | set(self._reads) | set(self._atomics)
        for a in addrs:
            w = self._writes[a]
            r = self._reads[a]
            x = self._atomics[a]
            if w:
                others = (r | x | w) - w
                if others:
                    self.conflicts.append(
                        Conflict(self.interval, a, w, others,
                                 "unsynchronised read/write" if (r | x) - w
                                 else "write/write"))
                elif len(w) > 1:
                    self.conflicts.append(
                        Conflict(self.interval, a, w, w, "write/write"))
            if x and r - x:
                self.conflicts.append(
                    Conflict(self.interval, a, x, r - x,
                             "unsynchronised atomic/read"))
        self._reads.clear()
        self._writes.clear()
        self._atomics.clear()
        self.interval += 1


def run_block(body, nthreads, mem, order="random", rng=None):
    """Execute `nthreads` copies of the generator `body(tid, nthreads)` against
    `mem` under the named schedule. Returns the Block."""
    blk = Block(mem)
    gens = [body(t, nthreads) for t in range(nthreads)]
    state = [RUNNABLE] * nthreads
    send = [None] * nthreads

    def step(t):
        try:
            op = gens[t].send(send[t])
        except StopIteration:
            state[t] = DONE
            return
        send[t] = None
        if op[0] == "r":
            send[t] = blk.load(t, op[1])
        elif op[0] == "w":
            blk.store(t, op[1], op[2])
        elif op[0] == "aadd":
            send[t] = blk.atomic_add(t, op[1], op[2])
        elif op[0] == "bar":
            state[t] = AT_BARRIER
        else:
            raise AssertionError("bad op %r" % (op,))

    rr = 0
    guard = 0
    while any(s != DONE for s in state):
        guard += 1
        if guard > 40_000_000:
            raise AssertionError("scheduler did not converge")
        live = [t for t in range(nthreads) if state[t] == RUNNABLE]
        if live:
            if order == "random":
                t = rng.choice(live)
            elif order == "sequential":
                t = live[0]
            elif order == "reverse":
                t = live[-1]
            elif order == "roundrobin":
                rr = (rr + 1) % len(live)
                t = live[rr]
            else:
                raise AssertionError("bad order %r" % order)
            step(t)
            continue
        # nobody runnable: everyone is either done or waiting on the barrier
        waiting = [t for t in range(nthreads) if state[t] == AT_BARRIER]
        if not waiting:
            break
        blk.close_interval()
        for t in waiting:
            state[t] = RUNNABLE
    blk.close_interval()
    return blk


# --------------------------------------------------------------------------
# The two algorithms
# --------------------------------------------------------------------------

def make_old(n_local, start_expert, num_experts):
    """PRE-M3-I5c. topk_softmax_sm100.cuh:372-382 / topk_sigmoid_sm100.cuh:382-392

        for (int expert = start_expert + threadIdx.x; expert < end_expert;
             expert += blockDim.x) {
          int const local_expert = expert - start_expert;
          int const mark = mpk_active_expert_ids[local_expert];
          if (mark >= 0) {
            int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
            mpk_active_expert_ids[pos] = expert;
          }
        }
    """
    end_expert = start_expert + n_local

    def body(tid, nthreads):
        expert = start_expert + tid
        while expert < end_expert:
            local_expert = expert - start_expert
            mark = yield ("r", local_expert)
            if mark >= 0:
                pos = yield ("aadd", num_experts, 1)
                yield ("w", pos, expert)
            expert += nthreads

    return body


def make_new(n_local, start_expert, num_experts, with_barrier=True):
    """POST-M3-I5c. The per-tile prefix-count compaction, transliterated.

    `with_barrier=False` is the ABLATION: the same prefix-count code with the
    one `__syncthreads()` deleted. It must race, otherwise the barrier is not
    the thing carrying the correctness argument and the detector is blind."""

    def body(tid, nthreads):
        base = 0
        tile_base = 0
        while tile_base < n_local:
            tile_end = min(tile_base + nthreads, n_local)
            local_expert = tile_base + tid
            is_active = False
            rank_in_tile = 0
            tile_count = 0
            for j in range(tile_base, tile_end):
                if (yield ("r", j)) >= 0:
                    tile_count += 1
                    if j < local_expert:
                        rank_in_tile += 1
                    if j == local_expert:
                        is_active = True
            if with_barrier:
                yield ("bar",)
            if is_active:
                yield ("w", base + rank_in_tile, start_expert + local_expert)
            base += tile_count
            tile_base += nthreads
        if tid == 0:
            yield ("w", num_experts, base)

    return body


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def initial_mem(active_locals, n_local, num_experts):
    """Phase 0 leaves marks[-1] everywhere and marks[e]==e for active experts;
    slot num_experts is the counter, zeroed by thread 0."""
    mem = [-1] * num_experts + [0]
    for e in active_locals:
        mem[e] = e
    return mem


def check(blk, active_locals, start_expert, num_experts):
    """Returns (ok, reason). Requires the exact ascending list AND the count."""
    if blk.oob:
        return False, "out-of-bounds access %r" % (blk.oob[:3],)
    want = sorted(start_expert + e for e in active_locals)
    got_n = blk.mem[num_experts]
    if got_n != len(want):
        return False, "count %d != %d" % (got_n, len(want))
    got = blk.mem[:got_n]
    if got != want:
        if sorted(got) != want:
            return False, "SET differs (got %r)" % (got[:12],)
        return False, "order not ascending (got %r)" % (got[:12],)
    return True, ""


def masks_for(n_local, rng):
    """Edge cases first, then randomised densities."""
    out = [
        ("none", []),
        ("all", list(range(n_local))),
        ("first", [0]),
        ("last", [n_local - 1]),
        ("low-half", list(range(n_local // 2))),
        ("high-half", list(range(n_local // 2, n_local))),
        ("alternating", list(range(0, n_local, 2))),
        ("all-but-first", list(range(1, n_local))),
    ]
    for density in (0.03, 0.25, 0.75):
        for r in range(2):
            k = max(0, int(round(n_local * density)))
            out.append(("rand-d%.2f-%d" % (density, r),
                        sorted(rng.sample(range(n_local), k))))
    return out


CONFIGS = [
    # (num_experts, n_local, start_expert, [blockDim.x ...])
    # the two shipped shapes come first
    (256, 256, 0, [256]),                      # Qwen3.5 / DeepSeek-V3 as shipped
    (128, 128, 0, [256, 128]),                 # Qwen3 30B-A3B shape
    # blockDim.x < NUM_EXPERTS -> multi-tile, the M3-I9b aspect
    (256, 256, 0, [128, 64, 32, 33, 7, 1]),
    (64, 64, 0, [64, 32, 16, 5, 3, 1]),
    (16, 16, 0, [16, 8, 5, 4, 3, 2, 1]),
    (8, 8, 0, [8, 4, 3, 2, 1]),
    # blockDim.x > NUM_EXPERTS (idle threads) and expert-parallel slices
    (16, 16, 0, [32, 17]),
    (64, 32, 16, [64, 32, 8, 3]),              # start_expert != 0, n_local < N
]

SCHEDULES = ["sequential", "reverse", "roundrobin", "random", "random"]


def main():
    rng = random.Random(20260727)
    n_case = 0
    new_fail = []
    old_stats = defaultdict(lambda: [0, 0])   # sched -> [wrong, total]
    old_raced = 0
    new_raced = 0
    abl_raced = 0
    abl_stats = [0, 0]                        # [wrong, total]
    new_by_case = {}

    for num_experts, n_local, start_expert, block_dims in CONFIGS:
        for mname, active in masks_for(n_local, rng):
            for nthreads in block_dims:
                outs = set()
                for sched in SCHEDULES:
                    n_case += 1
                    tag = "N=%d n_local=%d start=%d B=%d %s/%s" % (
                        num_experts, n_local, start_expert, nthreads,
                        mname, sched)

                    # ---- the fix ------------------------------------------
                    blk = run_block(make_new(n_local, start_expert, num_experts),
                                    nthreads,
                                    initial_mem(active, n_local, num_experts),
                                    order=sched, rng=rng)
                    ok, why = check(blk, active, start_expert, num_experts)
                    if not ok:
                        new_fail.append((tag, why))
                    if blk.conflicts:
                        new_raced += 1
                        new_fail.append((tag, "RACE: %s" % blk.conflicts[0]))
                    outs.add(tuple(blk.mem[:blk.mem[num_experts]] +
                                   [blk.mem[num_experts]]))

                    # ---- the pre-fix control ------------------------------
                    oblk = run_block(make_old(n_local, start_expert, num_experts),
                                     nthreads,
                                     initial_mem(active, n_local, num_experts),
                                     order=sched, rng=rng)
                    ook, _ = check(oblk, active, start_expert, num_experts)
                    # the old code never claimed ascending order; judge it on
                    # the SET + count only, which is all its consumers need
                    owant = sorted(start_expert + e for e in active)
                    on = oblk.mem[num_experts]
                    oset_ok = (not oblk.oob and on == len(owant) and
                               sorted(oblk.mem[:on]) == owant)
                    old_stats[sched][1] += 1
                    if not oset_ok:
                        old_stats[sched][0] += 1
                    if oblk.conflicts:
                        old_raced += 1

                    # ---- ablation: the fix with the barrier deleted --------
                    ablk = run_block(
                        make_new(n_local, start_expert, num_experts,
                                 with_barrier=False),
                        nthreads,
                        initial_mem(active, n_local, num_experts),
                        order=sched, rng=rng)
                    aok, _ = check(ablk, active, start_expert, num_experts)
                    abl_stats[0] += 0 if aok else 1
                    abl_stats[1] += 1
                    if ablk.conflicts:
                        abl_raced += 1

                if len(outs) != 1:
                    new_fail.append(("N=%d B=%d %s" % (num_experts, nthreads,
                                                       mname),
                                     "NON-DETERMINISTIC across schedules"))
                new_by_case[(num_experts, n_local, start_expert, nthreads,
                             mname)] = outs

    total = n_case
    print("M3-I5c compaction model")
    print("=" * 72)
    print("cases: %d  (%d configs x masks x blockDims x %d schedules)"
          % (total, len(CONFIGS), len(SCHEDULES)))
    print()
    print("POST-FIX (per-tile prefix count + barrier)")
    print("  barrier-interval races detected : %d / %d" % (new_raced, total))
    print("  wrong set / count / order       : %d / %d"
          % (len([f for f in new_fail if not f[1].startswith("RACE")]), total))
    print("  identical output across all %d schedules per case: %s"
          % (len(SCHEDULES),
             "yes" if all(len(v) == 1 for v in new_by_case.values()) else "NO"))
    print("  output order                    : strictly ascending by expert id")
    print()
    print("PRE-FIX control (in-place atomicAdd scatter) -- SET+count only")
    print("  barrier-interval races detected : %d / %d" % (old_raced, total))
    for s in SCHEDULES[:4]:
        bad, tot = old_stats[s]
        print("  wrong under %-11s        : %d / %d  (%.0f%%)"
              % (s, bad, tot, 100.0 * bad / max(tot, 1)))
    print()
    print("ABLATION: the fix with its __syncthreads() deleted")
    print("  barrier-interval races detected : %d / %d" % (abl_raced, total))
    print("  wrong set / count / order       : %d / %d"
          % (abl_stats[0], abl_stats[1]))
    print()

    if new_fail:
        print("FAIL -- %d finding(s):" % len(new_fail))
        for tag, why in new_fail[:25]:
            print("   %-46s %s" % (tag, why))
        return 1

    if old_raced == 0:
        print("FAIL -- the detector found no race in the PRE-fix code, so it")
        print("        cannot be trusted to have cleared the POST-fix code.")
        return 1
    if sum(v[0] for v in old_stats.values()) == 0:
        print("FAIL -- the pre-fix control never produced a wrong answer;")
        print("        the adversarial schedules are not adversarial.")
        return 1
    if abl_raced == 0 or abl_stats[0] == 0:
        print("FAIL -- deleting the barrier from the fix changed nothing, so")
        print("        the barrier is not what makes the fix correct.")
        return 1

    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
