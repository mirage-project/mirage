"""Analysis tools for MPK v2 profiler dumps (--profiling --prof-dump x.npy).

Usage:
    python -m mirage.mpk.prof check    dump.npy   # structural invariants
    python -m mirage.mpk.prof summary  dump.npy   # per-task-type attribution
    python -m mirage.mpk.prof pagewait dump.npy   # page-protocol serialization
    python -m mirage.mpk.prof diff     a.npy b.npy

The dump is the raw profiler buffer: entry = {u32 tag, u32 globaltimer_ns}.
Geometry (nblocks, ngroups) comes from the header entry; the reserved tail
(accumulators + emitter cursors) is V2_PROF_TAIL entries at the end (must
match runtime_v2.cuh).
"""
from __future__ import annotations

import sys
import numpy as np
from collections import defaultdict

# must match runtime_v2.cuh: misc(256) + cursors(1024) + spin(7*256) + suffix(256)
V2_PROF_TAIL = (1048576 + 1) + 256 + 1024 + 7 * 256 + 256
V2_PROF_NUM_BUCKETS = 7
WINDOW_ITERS = 25

GROUP_NAMES = ["consumer", "loader", "launcher", "storer", "controller",
               "consumer-phase", "loader-phase", "launcher-phase"]

LINEAR_TYPES = {244, 245}
EVENT_NAMES = {
    244: "LINEAR_V2", 245: "LINEAR_RES_V2",
    281: "RMSNORM", 282: "SILU_MUL", 283: "EMBEDDING",
    284: "ATTENTION", 285: "ARGMAX_P", 286: "ARGMAX_R",
    204: "PREPARE_BATCH", 205: "ITER_SYNC", 206: "GO_WAIT",
    207: "DEP_WAIT", 208: "PAGE_WAIT",
    209: "W_TMA_WAIT", 210: "MMA_EMPTY_WAIT", 211: "TMEM_READY_WAIT",
    212: "MAINLOOP_WAIT", 213: "EPILOGUE_WAIT", 214: "CONSUMER_DONE_WAIT",
}
BUCKET_NAMES = {0: "linear", 1: "attn", 2: "rmsnorm", 3: "silu",
                4: "argmax", 5: "embed", 6: "other"}


def dur_us(a, b):
    return ((b - a) & 0xFFFFFFFF) / 1e3


class Dump:
    """Parsed profiler dump: per-(sm, group) ordered windows."""

    def __init__(self, path_or_array):
        buf = (np.load(path_or_array) if isinstance(path_or_array, str)
               else np.asarray(path_or_array))
        self.buf = buf
        hdr = int(buf[0])
        self.nblocks = hdr & 0xFFFFFFFF
        self.ngroups = hdr >> 32
        self.ntracks = self.nblocks * self.ngroups
        self.windows = defaultdict(list)  # (sm, group) -> [(start, end, ev)]
        self.stray = 0
        ents = buf[1:len(buf) - V2_PROF_TAIL]
        nz = np.nonzero(ents)[0]
        self.n_entries = len(nz)
        st = {}
        for i in nz:
            v = int(ents[i])
            tag, ts = v & 0xFFFFFFFF, v >> 32
            ev = (tag >> 2) & 0x1FF
            et = tag & 3
            tr = i % self.ntracks
            if et == 0:
                if tr in st:
                    self.stray += 1
                st[tr] = (ts, ev)
            elif et == 1:
                if tr not in st:
                    self.stray += 1
                    continue
                s, ev0 = st.pop(tr)
                self.windows[(tr // self.ngroups, tr % self.ngroups)].append(
                    (s, ts, ev0))

    # accumulator regions ---------------------------------------------------
    def spin_bucket(self, b):
        base = len(self.buf) - 7 * 256 - 256 + 256 * b
        ns = self.buf[base:base + 128].astype(float)
        n = self.buf[base + 128:base + 256].astype(float)
        return ns, n

    def suffix(self):
        base = len(self.buf) - 256
        return (self.buf[base:base + 128].astype(float),
                self.buf[base + 128:base + 256].astype(float))

    def dropped(self):
        """Events dropped by the emitter's capacity guard, per SM."""
        base = len(self.buf) - V2_PROF_TAIL
        return self.buf[base:base + 128].astype(float)


def cmd_check(d: Dump) -> int:
    ok = True
    print(f"header: nblocks={d.nblocks} ngroups={d.ngroups} "
          f"entries={d.n_entries} stray={d.stray}")
    if d.stray > d.nblocks * d.ngroups:
        print("FAIL: begin/end alternation broken")
        ok = False
    # durations sane
    bad = sum(1 for w in d.windows.values() for (s, e, _) in w
              if dur_us(s, e) > 50_000)
    if bad:
        print(f"FAIL: {bad} windows exceed 50ms")
        ok = False
    # role counts consistent
    mism = 0
    for sm in range(d.nblocks):
        c = [len(d.windows.get((sm, g), [])) for g in range(4)]
        if max(c) - min(c) > 2:
            mism += 1
    if mism:
        print(f"FAIL: {mism} SMs with mismatched role window counts")
        ok = False
    ntask = len(d.windows.get((0, 0), []))
    print(f"role windows/SM: {ntask}")
    # phase containment: every phase slice must lie inside SOME window of its
    # role track (interval check — phase tracks interleave wait kinds, so
    # index pairing no longer applies).
    if d.ngroups >= 6:
        import bisect
        viol = checked = 0
        for pg, rg in ((5, 0), (6, 1), (7, 2)):
            if pg >= d.ngroups:
                continue
            for sm in range(d.nblocks):
                role = d.windows.get((sm, rg), [])
                ph = d.windows.get((sm, pg), [])
                if not role or not ph:
                    continue
                t0 = role[0][0]
                ivals = sorted((((s - t0) & 0xFFFFFFFF),
                                ((e - t0) & 0xFFFFFFFF)) for s, e, _ in role)
                starts = [iv[0] for iv in ivals]
                for s, e, _ in ph:
                    checked += 1
                    su = (s - t0) & 0xFFFFFFFF
                    eu = (e - t0) & 0xFFFFFFFF
                    i = bisect.bisect_right(starts, su) - 1
                    if i < 0 or eu > ivals[i][1] + 2000:  # 2us slop
                        viol += 1
        print(f"phase containment: {checked} slices, {viol} violations")
        if viol > checked * 0.01 + d.nblocks:
            print("FAIL: phase slices do not nest")
            ok = False
    drops = d.dropped().sum()
    print(f"emitter dropped events: {int(drops)}")
    if drops > 0:
        print("FAIL: trace truncated (enlarge buffer or shrink window)")
        ok = False
    print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
    return 0 if ok else 1


def cmd_summary(d: Dump, iters=WINDOW_ITERS) -> int:
    norm = d.nblocks * iters
    suf_ns, suf_n = d.suffix()
    suf_per = suf_ns.sum() / max(suf_n.sum(), 1) / 1e3
    win_by_bucket = defaultdict(list)
    b_of = {244: 0, 245: 0, 246: 0, 247: 0, 284: 1, 281: 2, 282: 3,
            285: 4, 286: 4, 283: 5}
    for sm in range(d.nblocks):
        for s, e, ev in d.windows.get((sm, 0), []):
            win_by_bucket[b_of.get(ev, 6)].append(dur_us(s, e))
    print("type     n/SM/it  dep-wait   suffix  body+disp   win-mean   win-p50")
    for b in range(V2_PROF_NUM_BUCKETS):
        ns, n = d.spin_bucket(b)
        if n.sum() == 0:
            continue
        dep = ns.sum() / n.sum() / 1e3
        w = np.array(win_by_bucket.get(b, [0.0]))
        sfx = 0.0 if b == 0 else suf_per
        print("%-8s %7.1f %8.2fu %7.2fu %9.2fu %9.2fu %9.2fu" %
              (BUCKET_NAMES[b], n.sum() / norm, dep, sfx,
               w.mean() - dep - sfx, w.mean(), np.percentile(w, 50)))
    busy = np.array([sum(dur_us(s, e) for s, e, _ in d.windows.get((sm, 0), []))
                     for sm in range(d.nblocks)])
    print(f"\nconsumer busy: {busy.mean()/iters/1e3:.2f} ms/SM/step "
          f"(min {busy.min()/iters/1e3:.2f}, max {busy.max()/iters/1e3:.2f})")
    return 0


def cmd_pagewait(d: Dump, iters=WINDOW_ITERS) -> int:
    norm = d.nblocks * iters
    tot = after_b = after_l = dead = 0.0
    n_b = n_l = 0
    for sm in range(d.nblocks):
        L = d.windows.get((sm, 1), [])
        # loader-phase track interleaves wait kinds; the page prefix (208)
        # is one slice per task, so it index-aligns with loader windows.
        P = [w for w in d.windows.get((sm, 6), []) if w[2] == 208]
        C = d.windows.get((sm, 0), [])
        n = min(len(L), len(P), len(C))
        for k in range(n):
            dwait = dur_us(P[k][0], P[k][1])
            tot += dwait
            if L[k][2] in LINEAR_TYPES and k > 0:
                if L[k - 1][2] in LINEAR_TYPES:
                    after_l += dwait
                    n_l += 1
                else:
                    after_b += dwait
                    n_b += 1
                    if dur_us(P[k][0], C[k - 1][1]) < 1e6:
                        dead += min(dur_us(P[k][0], C[k - 1][1]), dwait)
    print(f"loader PAGE_WAIT total:      {tot/norm:7.0f} us/SM/step")
    print(f"  linear after bodiless:     {after_b/norm:7.0f} us/SM/step "
          f"({n_b/norm:.1f} tasks, mean {after_b/max(n_b,1):.1f}us)")
    print(f"    dead prefetch (overlap): {dead/norm:7.0f} us/SM/step")
    print(f"  linear after linear:       {after_l/norm:7.0f} us/SM/step "
          f"({n_l/norm:.1f} tasks, mean {after_l/max(n_l,1):.1f}us)")
    return 0


def cmd_diff(a: Dump, b: Dump, iters=WINDOW_ITERS) -> int:
    def busy_by_type(d):
        out = defaultdict(float)
        for sm in range(d.nblocks):
            for s, e, ev in d.windows.get((sm, 0), []):
                out[ev] += dur_us(s, e)
        return out
    ba, bb = busy_by_type(a), busy_by_type(b)
    norm_a, norm_b = a.nblocks * iters, b.nblocks * iters
    print("type             A us/SM/it   B us/SM/it      delta")
    for ev in sorted(set(ba) | set(bb), key=lambda e: -(bb.get(e, 0) - ba.get(e, 0))):
        va, vb = ba.get(ev, 0) / norm_a, bb.get(ev, 0) / norm_b
        nm = EVENT_NAMES.get(ev, f"ev{ev}")
        print(f"{nm:16} {va:10.1f}  {vb:10.1f}  {vb-va:+10.1f}")
    return 0


def print_run_summary(profiler_tensor) -> None:
    """Called automatically at the end of a profiled v2 run."""
    try:
        d = Dump(profiler_tensor.cpu().numpy())
        if d.ngroups < 5:
            return  # v1 buffer — no v2 summary
        print("\n=== MPK v2 profiling summary (last "
              f"{WINDOW_ITERS} decode steps) ===")
        cmd_summary(d)
        cmd_pagewait(d)
        print("=== (details: --prof-dump x.npy + python -m mirage.mpk.prof) ===")
    except Exception as e:  # noqa: BLE001 — never break the run for a summary
        print(f"[prof] summary failed: {e}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "diff":
        return cmd_diff(Dump(argv[2]), Dump(argv[3]))
    d = Dump(argv[2])
    return {"check": cmd_check, "summary": cmd_summary,
            "pagewait": cmd_pagewait}[cmd](d)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
