"""CPU model of STATIC_MOE64_W13C (tasks/scaled_moe_experts.cuh): W13 over a slot's own tokens in groups of 16.

Part A, the data (per randomized step, T = 64 tokens, the GLM-5.3 TP8 bank or the GLM-5.3-Flash TP4 bank; routing and
build_table from moe64_w2c_model.py; half of the runs with idle rows, live drawn from 1..63: a token several times in a
shared slot. rows_once must be false exactly then, and the step runs the old code (SlotsOld, no W13C groups: checked in
moe64_w2c_model.py); the checks below are of the steps with every row live):
  1. three SMs build their tables (TOKEN_ORDER lists), order_slots (natural place order: shared, many-token, others;
     the W2_TABLE rotation r0 / rot), w13c_groups (the extra groups 1 .. ng - 1 of each slot, place order)
  2. entry_of on every SM: queue index idx < n13 -> (slot, group q, first item); the same on every SM; every
     (slot, group, item) exactly once
  3. the loader's job (w13c_job): the group's n = min(ntok - 16 q, 16) tokens, nq = ceil(n / 4) gather4 a K tile, the
     row words (rows past n: the group's first token); the expect bytes vs the TMA bytes (32 KB + 2 nq 512 B)
  4. the stage's z_q image: gather4 qq's 4 rows at + 512 qq of the K tile (1024-B aligned), the 128-B swizzle by
     address (16-B chunk k of a row at chunk k ^ ((address >> 7) & 7)); the MMA's B row c (c < 16) through the swizzle
     (chunk k at k ^ (c & 7)): for c < n it must be token tk[c]'s z_q bytes
  5. drain_promoted_c: column c's activation scale index tok[c] (c < n), its token
  6. the epilogue (another SM: the entry's SM) writes h_q row slot T + 16 q + c (c < n) of token tk[c] (hq_out c0);
     W2 on a third SM reads row slot T + (i - start) for list entry i (W2C): it must be token tok[i]'s
  7. CHQ: each group adds 2 per 128 features (MT13 a group); W2 waits for MT13 ceil(ntok / 16): equal to the adds
  8. the extra groups' table fits Smem OFF_XG (NXG ints)
Negative controls (each must fail): the old atomic list order; W2 waiting MT13 (not per group); nq = floor(n / 4);
h_q rows without the group's c0; the gather rows without the group's 16 q; the scales indexed by place, not token; the
W13C code for a step with idle rows (no fallback).

Part B, the W13C ring protocol (randomized interleavings of one SM's roles, the bank's KT_LAT / 2 stages an item):
  loaders warp 0 (s even) / warp 7 (s odd): per own stage the `empty` wait of slot g % 4 (g >= 4), expect + TMAs
  (async: they land later and complete the tx); warp 7 publishes its stage count after each item;
  issuers 0 / 1: every stage's `full` in order, the owner (s % 2) issues the MMAs + commits `empty`, the other arrives;
  each round's acc_full (commit or arrive), the acc_empty wait before a round of a reused tick (ACC_STAGES 2);
  the tensor core: one FIFO of the issued MMAs (reading the stage) and commits (arriving when reached);
  the epilogue: per round the acc_full wait, then the acc_empty arrive;
  warp 0's switch after its last item: warp 7's count >= its own, then each slot's last `empty` phase; then W2 writes
  the ring's area.
  Checked: every parity wait passes only on its intended phase (the completed count exactly one past it); an MMA reads
  the stage of its own g, landed; a TMA never writes a stage an MMA of an older use still reads; W2 writes the area only
  after every MMA read and TMA write of it; no deadlock (every role ends).
  Roles and engines run at random speeds that change in bursts (a lagging warp 7, a slow tensor core, ...).
  Also checked (must pass): the same protocol with 4 accumulator ticks.
Negative controls (each must fail): 3 slots (odd depth with the two alternating loaders); empty count 1 (only the
owner's commit); the switch without warp 7's count with 4 accumulator ticks; the switch waiting only warp 0's slots.
Printed, not required to fail: the switch without warp 7's count at ACC_STAGES 2 (T = 64): issuer 0 starts a round only
after the round two ticks back was drained, whose acc_full commit follows issuer 1's last MMAs of that round in the
tensor core's order, so the slots' previous uses are consumed when the switch looks; the count wait stays in the code.
Run: python3 moe64_w13c_model.py [runs]   (exit 0: both parts pass every run, every control fails)
"""
import random
import sys

from moe64_w2c_model import CFGS, build_table, hq_swz, route, rows_once

T, N16, NSM = 64, 16, 148
MT13 = 4          # IR 256 / 64
W13_ROUND = 2 * NSM // MT13
PER13, NPE = MT13 // 2, 2


# ---------------- Part A ----------------
def order_slots(tab, shared):
    """order_slots: natural place p -> slot; fill (the table, place w2_table_pos(p)), s_ints[3..5]"""
    ntok, n_live = tab["ntok"], len(tab["ntok"])
    n0 = n_live - shared
    nat = list(range(n0, n_live)) + [i for i in range(n0) if ntok[i] > 8] + [i for i in range(n0) if ntok[i] <= 8]
    tot = sum(1 for i in range(n0) if ntok[i] > 8)
    r0 = rot = 0
    if n_live * (MT13 // 2) <= 2 * NSM + NSM // 10:
        r0 = min(W13_ROUND, n_live)
        rot = min(shared + tot, r0)

    def pos(p):
        return (p + r0 - rot if p < rot else p - rot) if p < r0 else p
    fill = [None] * n_live
    for p, s in enumerate(nat):
        fill[pos(p)] = s
    return dict(fill=fill, M=shared + tot, r0=r0, rot=rot, pos=pos)


def w13c_groups(tab, od):
    xg = []
    for p in range(len(tab["ntok"])):
        s = od["fill"][od["pos"](p)]
        ng = (tab["ntok"][s] + N16 - 1) // N16
        xg += [(s, q) for q in range(1, ng)]
    return xg


def entries(tab, od, xg):
    """entry_of for the W13 entries: idx -> (slot, q, first item)"""
    n_live, M, nx = len(tab["ntok"]), od["M"], len(xg)
    out = []
    for idx in range((n_live + nx) * PER13):
        p13 = idx // PER13
        if M <= p13 < M + nx:
            s, q = xg[p13 - M]
        else:
            s, q = od["fill"][od["pos"](p13 - nx if p13 >= M else p13)], 0
        out.append((s, q, idx % PER13 * NPE))
    return out


def zbyte(t, kt, f):
    return ("z", t, kt, f)


def run_data(rng, cfg, ctl, live=T):
    IR, NSLOT, shared = cfg["IR"], cfg["NSLOT"], cfg["SHARED"]
    pairs = route(rng, cfg, live)
    fast = rows_once(pairs) or ctl == "no_fallback"
    if ctl is None and fast != (live == T):
        return f"rows_once {rows_once(pairs)} with live {live}"
    if not fast:   # the old code (SlotsOld): moe64_w2c_model.py checks it
        return None
    tabs = [build_table(pairs, rng, ctl != "atomic_order") for _ in range(3)]   # W13 loader+epilogue SM, another, W2 SM
    if None in tabs:
        return "two pairs on one entry (a token twice in a slot)"
    ods = [order_slots(tb, shared) for tb in tabs]
    xgs = [w13c_groups(tb, od) for tb, od in zip(tabs, ods)]
    ents = [entries(tb, od, xg) for tb, od, xg in zip(tabs, ods, xgs)]
    npair = len(pairs) * len(pairs[0])
    nxg = npair // N16 + 1
    if len(xgs[0]) > nxg - 1:
        return f"extra groups {len(xgs[0])} > NXG - 1 = {nxg - 1}"
    if 2304 + 64 + 4 * nxg > 3072:
        return f"OFF_XG table {nxg} ints past the OFF_W2BAR area"
    if not (ents[0] == ents[1] == ents[2]):
        return "entry_of differs between SMs"
    tw, tr = tabs[0], tabs[2]
    nslot = len(tw["ntok"])
    want = {(s, q, m) for s in range(nslot) for q in range((tw["ntok"][s] + N16 - 1) // N16) for m in range(MT13)}
    got = [(s, q, m0 + i) for s, q, m0 in ents[0] for i in range(NPE)]
    if sorted(got) != sorted(want):
        return "the W13 entries do not cover every (slot, group, item) once"
    planes, chq = {}, [0] * nslot
    FEATS = (0, 15, 16, 77, 127, 128, 200, 255)
    for s, q, m0 in ents[0]:
        ntk = tw["ntok"][s]
        n = min(ntk - N16 * q, N16)
        base = tw["start"][s] + (0 if ctl == "gather_no_q" else N16 * q)
        tk = [tw["tok"][base + c] for c in range(n)]
        nq = n // 4 if ctl == "nq_floor" else (n + 3) // 4
        rows = [tk[c if c < n else 0] for c in range(N16)]
        expect = 2 * 16384 + ((n + 3) // 4) * 1024
        tx = 2 * 16384 + nq * 2 * 512
        if tx != expect:
            return f"tx {tx} != expect {expect} (slot {s} group {q} n {n})"
        for kt in range(2):   # the stage's K tile image: address -> byte (1024-aligned base 0)
            img = {}
            for qq in range(nq):
                for r in range(4):
                    a0 = 512 * qq + 128 * r
                    for f in range(0, 128, 16):
                        img[a0 + ((((f >> 4) ^ ((a0 >> 7) & 7)) << 4))] = zbyte(rows[4 * qq + r], kt, f)
            for c in range(n):   # the MMA's B row c, chunk k at k ^ (c & 7)
                for f in range(0, 128, 16):
                    got_b = img.get(128 * c + ((((f >> 4) ^ (c & 7)) << 4)), "stale")
                    want_b = zbyte(tw["tok"][tw["start"][s] + N16 * q + c], kt, f)
                    if got_b != want_b:
                        return f"slot {s} group {q} column {c} K tile {kt}: B row {got_b} != {want_b}"
        for c in range(n):   # drain_promoted_c's scale index, the epilogue's h_q rows
            xs_idx = c if ctl == "xs_by_place" else tk[c]
            if xs_idx != tw["tok"][tw["start"][s] + N16 * q + c]:
                return f"slot {s} group {q} column {c}: scale of token {xs_idx}"
            r = s * T + (0 if ctl == "no_c0" else N16 * q) + c
            for f in FEATS:
                if f < IR:
                    planes[(f // 128, r, hq_swz(f % 128, r))] = ("h", s, tk[c], f)
        chq[s] += 2 * NPE // 2   # thread 64 adds 2 per entry (two items, 128 features)
    for s in range(nslot):   # W2 on the third SM
        ntk = tr["ntok"][s]
        ng = (ntk + N16 - 1) // N16
        need = MT13 if ctl == "chq_mt13" else MT13 * ng
        if need != chq[s]:
            return f"slot {s}: W2 waits CHQ {need}, the groups add {chq[s]}"
        st = tr["start"][s]
        for i in range(st, st + ntk):
            c = i - st
            r = s * T + c
            for f in FEATS:
                if f < IR:
                    got_h = planes.get((f // 128, r, hq_swz(f % 128, r)))
                    if got_h != ("h", s, tr["tok"][i], f):
                        return f"slot {s} W2 column {c}: token {tr['tok'][i]} feature {f} reads {got_h}"
    return None


# ---------------- Part B ----------------
class Bar:
    def __init__(self, count):
        self.count, self.pend, self.tx, self.done = count, count, 0, 0

    def arrive(self, tx=0):
        self.pend -= 1
        self.tx += tx
        self._chk()

    def complete_tx(self, b):
        self.tx -= b
        self._chk()

    def _chk(self):
        if self.pend == 0 and self.tx == 0:
            self.done += 1
            self.pend = self.count


class Fail(Exception):
    pass


def wait(bar, phase, who):
    """parity wait for the completion of `phase` (a role's generator: yields its wake-up condition while blocked):
    passes when (done & 1) != (phase & 1); a pass must be exact (done == phase + 1)"""
    while (bar.done & 1) == (phase & 1):
        yield lambda: (bar.done & 1) != (phase & 1)
    if bar.done != phase + 1:
        raise Fail(f"{who}: parity wait for phase {phase} passed at completed count {bar.done}")


def run_ring(rng, nst, ctl):
    ST = 3 if ctl == "odd_depth" else 4
    ACC = 4 if ctl in ("acc4", "switch_no_w7_acc4") else 2   # T = 64: ACC_STAGES 2
    n_items = rng.choice([1, 2, 2, 4, 4, 6, 8])
    G = n_items * nst
    rounds = (nst + 1) // 2
    full = [Bar(1) for _ in range(ST)]
    empty = [Bar(1 if ctl == "empty1" else 2) for _ in range(ST)]
    acc_full = [Bar(2) for _ in range(ACC)]
    acc_empty = [Bar(1) for _ in range(ACC)]
    slot_tag = [None] * ST          # the use g whose bytes are in the slot (landed)
    slot_read = [0] * ST            # MMA reads queued, not done
    slot_wpend = [0] * ST           # TMA writes issued, not landed
    tma_q, tc_q = [], []            # async TMA landings; the tensor core FIFO
    w7pub = [0]
    w2_written = [False]

    def loader(par):
        g = 0
        for it in range(n_items):
            for s in range(nst):
                if s % 2 == par:
                    st = g % ST
                    if g >= ST:
                        yield from wait(empty[st], g // ST - 1, f"loader {par} g {g}")
                    if slot_read[st]:
                        raise Fail(f"loader {par} g {g}: slot {st} still read by an MMA")
                    full[st].arrive(tx=36864)
                    slot_wpend[st] += 1
                    tma_q.append((st, g))
                    yield
                g += 1
            if par == 1:
                w7pub[0] = g
                yield
        if par == 0:   # the switch to W2
            if ctl not in ("switch_no_w7", "switch_no_w7_acc4"):
                while w7pub[0] < g:
                    yield lambda: w7pub[0] >= g
            for st0 in range(ST):
                if ctl == "switch_own_slots" and st0 % 2 == 1:
                    continue
                if g > st0:
                    gq = st0 + ST * ((g - 1 - st0) // ST)
                    yield from wait(empty[st0], gq // ST, f"switch slot {st0}")
            for st0 in range(ST):
                if slot_read[st0] or slot_wpend[st0]:
                    raise Fail(f"W2 writes slot {st0} with {slot_read[st0]} MMA reads / {slot_wpend[st0]} TMA writes pending")
            w2_written[0] = True

    def issuer(iss):
        g = 0
        wtt = 0
        for it in range(n_items):
            for s in range(nst):
                tick = wtt + s // 2
                a = tick % ACC
                if s % 2 == 0 and tick >= ACC:
                    yield from wait(acc_empty[a], tick // ACC - 1, f"issuer {iss} acc_empty")
                st = g % ST
                yield from wait(full[st], g // ST, f"issuer {iss} g {g} full")
                if slot_tag[st] != g:
                    raise Fail(f"issuer {iss} g {g}: slot {st} holds use {slot_tag[st]}")
                if s % 2 == iss:
                    slot_read[st] += 1
                    tc_q.append(("mma", st, g))
                    tc_q.append(("arrive", empty[st]))
                elif ctl != "empty1":
                    empty[st].arrive()
                if s % 2 == 1 or s == nst - 1:
                    if 2 * (s // 2) + iss < nst:
                        tc_q.append(("arrive", acc_full[a]))
                    else:
                        acc_full[a].arrive()
                yield
                g += 1
            wtt += rounds

    def epilogue():
        wtt = 0
        for it in range(n_items):
            for r in range(rounds):
                tick = wtt + r
                a = tick % ACC
                yield from wait(acc_full[a], tick // ACC, "epilogue acc_full")
                yield
                acc_empty[a].arrive()
            wtt += rounds

    roles = [loader(0), loader(1), issuer(0), issuer(1), epilogue()]
    # each role / engine its own speed, changed in bursts (a lagging warp 7, a slow tensor core, ...)
    speed = [rng.choice([0.02, 0.2, 1.0, 1.0, 5.0]) for _ in range(len(roles) + 2)]
    alive = list(range(len(roles)))
    block = [None] * len(roles)   # a blocked role's wake-up condition
    while alive or tma_q or tc_q:
        acts, wts = [], []
        for i in alive:
            if block[i] is None or block[i]():
                acts.append(("r", i))
                wts.append(speed[i])
        if tma_q:
            acts.append(("tma", None))
            wts.append(speed[-2])
        if tc_q:
            acts.append(("tc", None))
            wts.append(speed[-1])
        if not acts:
            raise Fail("deadlock: every role waits, nothing in flight")
        kind, i = rng.choices(acts, weights=wts)[0]
        if rng.random() < 0.02:   # bursts: one role / engine changes its speed
            speed[rng.randrange(len(speed))] = rng.choice([0.02, 0.2, 1.0, 1.0, 5.0])
        if kind == "tma":
            st, g = tma_q.pop(rng.randrange(len(tma_q)))   # landings in any order
            if slot_read[st]:
                raise Fail(f"TMA of g {g} lands in slot {st} while an MMA reads it")
            slot_tag[st] = g
            slot_wpend[st] -= 1
            full[st].complete_tx(36864)
            continue
        if kind == "tc":
            op = tc_q.pop(0)
            if op[0] == "mma":
                _, st, g = op
                if slot_tag[st] != g:
                    raise Fail(f"MMA of g {g} reads slot {st} holding use {slot_tag[st]}")
                slot_read[st] -= 1
            else:
                op[1].arrive()
            continue
        try:
            block[i] = next(roles[i])
        except StopIteration:
            alive.remove(i)
    if not w2_written[0]:
        raise Fail("the switch never ran")


def ring_once(rng, nst, ctl):
    try:
        run_ring(rng, nst, ctl)
    except Fail as e:
        return str(e)
    return None


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(1122)
    ok = True
    nst_of = {"glm53_tp8": 6144 // 128 // 2, "flash_tp4": 4096 // 128 // 2}   # KT_LAT / 2 stages an item
    for name, cfg in CFGS.items():
        bad, nidle = 0, 0
        for _ in range(runs):
            lv = T if rng.random() < 0.5 else rng.randint(1, T - 1)
            nidle += lv < T
            bad += run_data(rng, cfg, None, lv) is not None
        print(f"{name} W13C data: {runs - bad} / {runs} runs pass ({nidle} with idle rows: the old code)")
        ok &= bad == 0
        for ctl in ("atomic_order", "chq_mt13", "nq_floor", "no_c0", "gather_no_q", "xs_by_place", "no_fallback"):
            n = max(runs // 100, 200)
            fails, first = 0, None
            for _ in range(n):
                r = run_data(rng, cfg, ctl, rng.randint(1, T - 1) if ctl == "no_fallback" else T)
                if r is not None:
                    fails += 1
                    first = first or r
            print(f"  data control {ctl}: {fails} / {n} runs fail (must be > 0){'' if first is None else ': ' + first}")
            ok &= fails > 0
        nst = nst_of[name]
        bad, first = 0, None
        for _ in range(runs):
            r = ring_once(rng, nst, None)
            if r is not None:
                bad += 1
                first = first or r
        print(f"{name} W13C ring ({nst} stages an item): {runs - bad} / {runs} runs pass{'' if first is None else ': ' + first}")
        ok &= bad == 0
        bad, first = 0, None
        for _ in range(max(runs // 10, 500)):
            r = ring_once(rng, nst, "acc4")
            if r is not None:
                bad += 1
                first = first or r
        print(f"  ring with 4 accumulator ticks: {bad} / {max(runs // 10, 500)} runs fail (must be 0){'' if first is None else ': ' + first}")
        ok &= bad == 0
        n = max(runs // 20, 500)
        fails = sum(ring_once(rng, nst, "switch_no_w7") is not None for _ in range(n))
        print(f"  (the switch without warp 7's count, ACC_STAGES 2: {fails} / {n} runs fail; not required)")
        for ctl in ("odd_depth", "empty1", "switch_no_w7_acc4", "switch_own_slots"):
            n = max(runs // 20, 500) if ctl != "switch_no_w7_acc4" else max(runs // 4, 3000)
            fails, first = 0, None
            for _ in range(n):
                r = ring_once(rng, nst, ctl)
                if r is not None:
                    fails += 1
                    first = first or r
            print(f"  ring control {ctl}: {fails} / {n} runs fail (must be > 0){'' if first is None else ': ' + first}")
            ok &= fails > 0
    print("MODEL OK" if ok else "MODEL FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
