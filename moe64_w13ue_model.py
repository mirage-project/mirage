"""CPU model of STATIC_MOE64_W13UE's protocol (tasks/scaled_moe_experts.cuh issue_w13_ue / drain_w13_ue): one SM's W13
items on the layer ring at T = 64.

What is modelled (randomized interleavings of the roles; the engines run their operations at random times):
  loader     every stage g of the items in turn (load_job): the `empty` of slot g % SMAX (g >= SMAX: both issuers' arrivals
             of use g - SMAX, the owner's a commit), then the stage's TMA (lands later, then `full`)
  issuer i   per item (tick wtt = the item's number, slot a = wtt % ACC): acc_empty of slot a (wtt >= ACC); per stage s:
             the stage's `full`; the other issuer's stage (s % 2 != i): an arrive on `empty`; its own: per piece at the two
             chunk stores into the scale slot k = 2 (g % SMAX) + at (generic stores, made visible to the async proxy by
             the fence), then in its tcgen05 order: cp of slot k's chunks to TMEM slot k, the MMA (reads TMEM slot k and the
             stage, writes issuer i's accumulator of slot a; the item's first does not accumulate); then commit `empty`;
             after the item's stages, commit acc_full of slot a
  epilogue   per item: acc_full of slot a, reads both issuers' accumulators, arrives on acc_empty
  tensor core  each issuer's operations in issue order, with the tcgen05 ordering rules: an MMA after every earlier copy
             and MMA; a commit after every earlier operation; a copy after every earlier copy and commit only
             (tcgen05.mma -> tcgen05.cp is not an ordered pair: a copy may run before an earlier MMA)
  TMA        the stages' loads land in any order
Checked: a chunk store never overwrites bytes a pending copy still reads; a copy never overwrites TMEM chunk columns a
pending MMA still reads; every MMA reads its own piece's chunks (through the copy) and its own stage; an item's first MMA
writes an accumulator the epilogue has read; the epilogue reads both issuers' sums of its own item, complete; a TMA never
overwrites a stage a pending MMA reads; no deadlock.
Negative controls (each must fail): store_before_full (the chunk stores before the stage's `full` wait: the other
issuer's copy of use g - SMAX may still read the slot); one_slot (one scale slot a piece for every stage: the other
issuer's copy or MMA of stage g - 1 may still read it); no_acc_empty (the issuer starts an item without waiting for
acc_empty).
Printed, not required to fail: slots_per_issuer (scale slots by the issuer's own stage count, two per issuer). It passes:
before its stage g an issuer waited the `full` of g - 1, whose load followed the `empty` of g - 1 - SMAX = g - 4 (SMAX 3),
its own previous stage but one, so that stage's copies and MMAs are done.
Run: python3 moe64_w13ue_model.py [runs]   (exit 0: the protocol passes every run and every control fails)
"""
import random
import sys

SMAX, ACC = 3, 2


class Violation(Exception):
    pass


class Bar:
    def __init__(self, count):
        self.count, self.arr, self.n = count, 0, 0   # n: completed phases

    def arrive(self):
        self.arr += 1
        if self.arr == self.count:
            self.arr, self.n = 0, self.n + 1


def run(rng, ctl=None, nst=4):
    items = rng.randint(1, 6)
    total = items * nst
    full = [Bar(1) for _ in range(SMAX)]
    empty = [Bar(2) for _ in range(SMAX)]
    acc_full = [Bar(2) for _ in range(ACC)]
    acc_empty = [Bar(1) for _ in range(ACC)]
    stage = [None] * SMAX                  # the stage's data: its g
    smem, tmem = {}, {}                    # scale slot k -> the tag (g, at) of the chunks there
    acc = {}                               # (issuer, a) -> [item, MMAs added]
    drained = {(i, a): True for i in range(2) for a in range(ACC)}
    queues = [[], []]                      # each issuer's tcgen05 operations not yet done
    tma = []                               # loads in flight: (st, g)
    own = [0, 0]                           # (slots_per_issuer) each issuer's own stage count

    def pending_mma(pred):
        return any(op[0] == "mma" and pred(op) for q in queues for op in q)

    def loader():
        for g in range(total):
            st = g % SMAX
            if g >= SMAX:
                yield lambda: empty[st].n >= g // SMAX
            tma.append((st, g))
            yield None

    def slot(iss, g, at):
        if ctl == "slots_per_issuer":
            return ("p", iss, own[iss] % 2, at)
        if ctl == "one_slot":
            return (0, at)
        return (g % SMAX, at)

    def store(k, tag):
        if any(op[0] == "cp" and op[1] == k for q in queues for op in q):
            raise Violation(f"chunk store into slot {k} while a copy still reads it")
        smem[k] = tag

    def issuer(iss):
        g = 0
        for it in range(items):
            a = it % ACC
            if it >= ACC and ctl != "no_acc_empty":
                yield lambda: acc_empty[a].n >= it // ACC
            for s in range(nst):
                st = g % SMAX
                mine = s % 2 == iss
                ks = [slot(iss, g, at) for at in range(2)] if mine else []
                if mine and ctl == "store_before_full":
                    for at in range(2):
                        store(ks[at], (g, at))
                    yield None
                yield lambda: full[st].n > g // SMAX
                if not mine:
                    empty[st].arrive()
                    g += 1
                    yield None
                    continue
                if ctl != "store_before_full":
                    for at in range(2):
                        store(ks[at], (g, at))
                    yield None
                for at in range(2):
                    queues[iss].append(("cp", ks[at], (g, at)))
                    queues[iss].append(("mma", ks[at], (g, at), st, g, (iss, a), it, s == iss and at == 0))
                queues[iss].append(("commit", empty[st]))
                own[iss] += 1
                g += 1
                yield None
            queues[iss].append(("commit", acc_full[a]))
            yield None

    def epilogue():
        for it in range(items):
            a = it % ACC
            yield lambda: acc_full[a].n > it // ACC
            for iss in range(2):
                v = acc.get((iss, a))
                if v is None or v[0] != it or v[1] != nst:   # each issuer: nst / 2 stages x 2 pieces
                    raise Violation(f"epilogue item {it}: issuer {iss}'s accumulator holds {v}")
                drained[(iss, a)] = True
            acc_empty[a].arrive()
            yield None

    def ready(q, i):
        op = q[i]
        if op[0] == "cp":
            return all(o[0] == "mma" for o in q[:i])
        return i == 0

    def execute(q, i):
        op = q.pop(i)
        if op[0] == "commit":
            op[1].arrive()
        elif op[0] == "cp":
            k, tag = op[1], op[2]
            if smem.get(k) != tag:
                raise Violation(f"copy of slot {k} reads {smem.get(k)}, wants {tag}")
            if pending_mma(lambda o: o[1] == k and o[2] != tag):
                raise Violation(f"copy into TMEM slot {k} while an MMA of another piece still reads it")
            tmem[k] = tag
        else:
            _, k, tag, st, g, key, it, first = op
            if tmem.get(k) != tag:
                raise Violation(f"MMA of {tag} reads TMEM chunks {tmem.get(k)}")
            if stage[st] != g:
                raise Violation(f"MMA of stage {g} reads stage data {stage[st]}")
            if first:
                if not drained[key]:
                    raise Violation(f"item {it}'s first MMA overwrites accumulator {key} before the epilogue read it")
                drained[key] = False
                acc[key] = [it, 1]
            else:
                if acc.get(key, [None])[0] != it:
                    raise Violation(f"item {it}'s MMA adds into accumulator {key} of {acc.get(key)}")
                acc[key][1] += 1

    actors = [loader(), issuer(0), issuer(1), epilogue()]
    waits = [None] * len(actors)
    alive = list(range(len(actors)))
    while True:
        choices = []
        for i in alive:
            if waits[i] is None or waits[i]():
                choices.append(("actor", i))
        for qi, q in enumerate(queues):
            for i in range(len(q)):
                if ready(q, i):
                    choices.append(("op", qi, i))
        for i, (st, g) in enumerate(tma):
            choices.append(("tma", i))
        if not choices:
            if alive or any(queues) or tma:
                raise Violation("deadlock")
            return
        c = rng.choice(choices)
        if c[0] == "actor":
            i = c[1]
            try:
                waits[i] = next(actors[i])
            except StopIteration:
                alive.remove(i)
                waits[i] = None
        elif c[0] == "op":
            execute(queues[c[1]], c[2])
        else:
            st, g = tma.pop(c[1])
            if pending_mma(lambda o: o[3] == st and o[4] != g):
                raise Violation(f"TMA of stage {g} overwrites stage data a pending MMA reads")
            stage[st] = g
            full[st].arrive()


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(1)
    ok = True
    for ctl in (None, "store_before_full", "one_slot", "no_acc_empty", "slots_per_issuer"):
        fails, first = 0, None
        n = runs if ctl is None else max(runs // 10, 200)
        for _ in range(n):
            try:
                run(rng, ctl, nst=rng.choice((2, 4, 6)))
            except Violation as e:
                fails += 1
                first = first or str(e)
        name = ctl or "protocol"
        print(f"{name}: {fails} of {n} runs fail" + (f" (first: {first})" if first else ""))
        if ctl != "slots_per_issuer":   # (printed only)
            ok &= (fails == 0) if ctl is None else (fails > 0)
    print("MODEL OK" if ok else "MODEL FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
