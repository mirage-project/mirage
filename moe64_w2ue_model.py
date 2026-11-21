"""CPU model of STATIC_MOE64_W2UE's protocol (tasks/scaled_moe_experts.cuh): one SM's W2 phase at T = 64.

What is modelled (randomized interleavings of the roles, engines at random speeds that change in bursts):
  warp 0     per entry qn: publish (the qempty wait of position qn % 4 for qn >= 4), then its tiles (w2k even): the
             `empty` wait of stage k % 3 (k >= 3), expect + the weights' TMA (lands later); for an entry whose first
             tile is its own: the h_q buffer (e2 & 1; its `empty` wait for e2 >= 2) and the entry's scales into buffer
             qn % 4 (w2_stage_scales: a bulk copy completing w2s_bar(qn % 4))
  warp 7     the same for its tiles (w2k odd), after the entry is published
  issuer 0   per entry: the h_q buffer's `full`; (W2UE) w2s_bar(qn % 4) with its own parity bit, then reads the scales;
             per tile: acc_empty of tick a (wtt >= 2), the stage's `full`, then the scale chunks into the stage's free
             bytes (generic stores; the fence makes them visible to the async proxy) and at its first tile the constant
             chunk; then in its tcgen05 order: cp of the constant chunk to tick a's column, cp of the stage's chunk to
             tick a's column + 4, the MMAs (read the stage, the h_q buffer, tick a's chunk columns; write tick a's
             accumulator), commit `empty` of the stage, commit acc_full of tick a; after the entry's tiles, commit the h_q
             buffer's `empty`
  issuer 1   per tile: acc_empty of tick a (wtt >= 2), arrive on acc_full
  epilogue   per tile: acc_full, reads the accumulator, arrives on acc_empty; per entry: qempty (retired)
  tensor core  issuer 0's operations in issue order, with the tcgen05 ordering rules: an MMA after every earlier copy and
             MMA; a commit after every earlier operation; a copy after every earlier copy only (tcgen05.mma ->
             tcgen05.cp is not an ordered pair: a copy may run before an earlier MMA)
Checked: every parity wait passes only on its intended phase; the issuer reads its entry's scales (landed, not
overwritten before it read them); the stage holds the tile's weights; a chunk store never overwrites bytes a pending
copy still reads; every MMA reads its own tile's chunk (through the copy) and the constant chunk, its own h_q, and writes
an accumulator the epilogue has read; the epilogue reads its tile's result; no deadlock.
Negative controls (each must fail): no_fence (the chunk stores without fence.proxy.async: the copy reads the bytes the
async proxy saw before); cols_fixed (the chunk columns the same for both ticks: a copy runs before the previous tile's
MMA); no_w2s_wait (issuer 0 reads the scales without waiting w2s_bar); no_acc_empty (issuer 0 without the acc_empty
wait: its copies and MMA overwrite tick a's columns before the previous tile there is done or read).
Printed, not required to fail: chunk_before_full (the chunk stores before the stage's `full` wait). The acc_empty wait
before them already orders them: tile k - 2's MMA is done, so tile k - 3's copy of the same stage (before it in the
copy -> MMA order) is too.
Run: python3 moe64_w2ue_model.py [runs]   (exit 0: the protocol passes every run and every control fails)
"""
import random
import sys

ACC, W2_ST, W2QD = 2, 3, 4


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
    """parity wait for the completion of `phase`: passes when (done & 1) != (phase & 1); a pass must be exact"""
    while (bar.done & 1) == (phase & 1):
        yield lambda: (bar.done & 1) != (phase & 1)
    if bar.done != phase + 1:
        raise Fail(f"{who}: parity wait for phase {phase} passed at completed count {bar.done}")


def run(rng, ctl):
    n_ent = rng.randint(3, 30)
    ents = [rng.choice([1, 2, 2, 2]) for _ in range(n_ent)]   # tiles an entry
    full = [Bar(1) for _ in range(W2_ST)]
    empty = [Bar(1) for _ in range(W2_ST)]
    hqfull, hqempty = [Bar(1) for _ in range(2)], [Bar(1) for _ in range(2)]
    w2s = [Bar(1) for _ in range(W2QD)]
    qempty = [Bar(1) for _ in range(W2QD)]
    acc_full, acc_empty = [Bar(2) for _ in range(ACC)], [Bar(1) for _ in range(ACC)]
    pub = [0]
    st_w = [None] * W2_ST            # the tile whose weights landed in the stage
    st_rd = [0] * W2_ST              # MMAs queued that read the stage
    chunk_gen = [None] * W2_ST       # the stage's chunk bytes as the generic proxy wrote them (the tile)
    chunk_vis = [None] * W2_ST       # ... as the async proxy sees them (after the fence)
    chunk_rd = [0] * W2_ST           # copies queued that read the stage's chunk
    csf_vis = [False]                # the constant chunk visible to the async proxy
    hq_tag, hq_rd = [None, None], [0, 0]
    sbuf_tag = [None] * W2QD         # the entry whose scales landed in the buffer
    sbuf_unread = [False] * W2QD     # landed, issuer 0 has not read them yet
    tm_csf, tm_sfb = {}, {}          # TMEM chunk columns: what they hold
    acc_tag, acc_unread = [None] * ACC, [False] * ACC
    async_q, pipe = [], []

    tile_of = []                     # per tile k: (entry, e2 = the entry's index, its first tile)
    for qn, b in enumerate(ents):
        for i in range(b):
            tile_of.append((qn, i))
    first_tile = []
    k = 0
    for b in ents:
        first_tile.append(k)
        k += b

    def col(a):
        return 0 if ctl == "cols_fixed" else 64 * a

    def loader(par):
        for qn, b in enumerate(ents):
            if par == 0:
                if qn >= W2QD:
                    yield from wait(qempty[qn % W2QD], qn // W2QD - 1, f"publish {qn}")
                pub[0] = qn + 1
                yield
            else:
                while pub[0] <= qn:
                    yield lambda: pub[0] > qn
            k0 = first_tile[qn]
            for kk in range(k0, k0 + b):
                if kk % 2 != par:
                    continue
                s = kk % W2_ST
                if kk >= W2_ST:
                    yield from wait(empty[s], kk // W2_ST - 1, f"loader {par} tile {kk}")
                if st_rd[s]:
                    raise Fail(f"loader {par}: tile {kk}'s TMA into stage {s} while an MMA reads it")
                full[s].arrive(tx=32768)
                async_q.append(("w", s, kk))
                yield
            if k0 % 2 == par:   # the entry's first tile is this loader's: h_q, then the scales
                e2, hb = qn, qn & 1
                if e2 >= 2:
                    yield from wait(hqempty[hb], (e2 >> 1) - 1, f"loader {par} hq {e2}")
                if hq_rd[hb]:
                    raise Fail(f"loader {par}: h_q of entry {e2} into buffer {hb} while an MMA reads it")
                hqfull[hb].arrive(tx=4096)
                async_q.append(("hq", hb, e2))
                q = qn % W2QD
                if sbuf_unread[q]:
                    raise Fail(f"loader {par}: entry {qn}'s scales into buffer {q} before issuer 0 read entry "
                               f"{sbuf_tag[q]}'s")
                w2s[q].arrive(tx=528)
                async_q.append(("s", q, qn))
                yield

    def issuer0():
        w2s_par, ue_csf, wtt = 0, False, 0
        for qn, b in enumerate(ents):
            while pub[0] <= qn:
                yield lambda: pub[0] > qn
            e2, hb, q = qn, qn & 1, qn % W2QD
            yield from wait(hqfull[hb], e2 >> 1, f"issuer hq {e2}")
            if ctl != "no_w2s_wait":
                par = (w2s_par >> q) & 1   # the issuer's own parity bit of position q
                while (w2s[q].done & 1) == par:
                    yield lambda q=q, par=par: (w2s[q].done & 1) != par
                if w2s[q].done != qn // W2QD + 1:
                    raise Fail(f"issuer w2s {q}: passed at completed count {w2s[q].done} for entry {qn}")
                w2s_par ^= 1 << q
            if sbuf_tag[q] != qn:
                raise Fail(f"issuer: entry {qn} reads scale buffer {q} holding entry {sbuf_tag[q]}")
            sbuf_unread[q] = False
            k0 = first_tile[qn]
            for i in range(b):
                kk, a, s = k0 + i, wtt % ACC, (k0 + i) % W2_ST
                if wtt >= ACC and ctl != "no_acc_empty":
                    yield from wait(acc_empty[a], wtt // ACC - 1, f"issuer acc_empty {wtt}")

                def store_chunk():
                    if chunk_rd[s]:
                        raise Fail(f"issuer: tile {kk}'s chunk stores into stage {s} while a copy reads it")
                    chunk_gen[s] = kk

                if ctl == "chunk_before_full":
                    store_chunk()
                yield from wait(full[s], kk // W2_ST, f"issuer full tile {kk}")
                if st_w[s] != kk:
                    raise Fail(f"issuer: stage {s} holds tile {st_w[s]}, not {kk}")
                if ctl != "chunk_before_full":
                    store_chunk()
                first = not ue_csf
                ue_csf = True
                yield
                if ctl != "no_fence":   # fence.proxy.async.shared::cta (+ __syncwarp)
                    chunk_vis[s] = chunk_gen[s]
                    if first:
                        csf_vis[0] = True
                chunk_rd[s] += 1
                st_rd[s] += 1
                hq_rd[hb] += 1
                pipe.extend([["cpc", col(a)], ["cps", col(a), s, kk], ["mma", kk, a, s, hb, e2],
                             ["commit", empty[s]], ["commit", acc_full[a]]])
                yield
                wtt += 1
            pipe.append(["commit", hqempty[hb]])
            yield

    def issuer1():
        wtt = 0
        for qn, b in enumerate(ents):
            while pub[0] <= qn:
                yield lambda: pub[0] > qn
            for i in range(b):
                a = wtt % ACC
                if wtt >= ACC:
                    yield from wait(acc_empty[a], wtt // ACC - 1, f"issuer 1 acc_empty {wtt}")
                acc_full[a].arrive()
                yield
                wtt += 1

    def epilogue():
        wtt = 0
        for qn, b in enumerate(ents):
            while pub[0] <= qn:
                yield lambda: pub[0] > qn
            k0 = first_tile[qn]
            for i in range(b):
                a = wtt % ACC
                yield from wait(acc_full[a], wtt // ACC, f"epilogue acc_full {wtt}")
                if acc_tag[a] != k0 + i:
                    raise Fail(f"epilogue: tick {a} holds tile {acc_tag[a]}'s result, not {k0 + i}")
                acc_unread[a] = False
                yield
                acc_empty[a].arrive()
                wtt += 1
            qempty[qn % W2QD].arrive()
            yield

    def pipe_step():
        """one executable tensor-core operation (the ordering rules of the module docstring), at random"""
        ready = []
        for j, op in enumerate(pipe):
            before = pipe[:j]
            if op[0] in ("cpc", "cps"):
                if all(o[0] not in ("cpc", "cps") for o in before):
                    ready.append(j)
            elif not before:
                ready.append(j)
        j = rng.choice(ready)
        op = pipe.pop(j)
        if op[0] == "cpc":
            tm_csf[op[1]] = "one" if csf_vis[0] else "stale"
        elif op[0] == "cps":
            _, c, s, kk = op
            tm_sfb[c] = chunk_vis[s]
            chunk_rd[s] -= 1
        elif op[0] == "mma":
            _, kk, a, s, hb, e2 = op
            if tm_csf.get(col(a)) != "one":
                raise Fail(f"MMA of tile {kk}: the constant chunk column holds {tm_csf.get(col(a))}")
            if tm_sfb.get(col(a)) != kk:
                raise Fail(f"MMA of tile {kk}: its chunk column holds tile {tm_sfb.get(col(a))}'s scales")
            if st_w[s] != kk or hq_tag[hb] != e2:
                raise Fail(f"MMA of tile {kk}: stage {s} holds {st_w[s]}, h_q buffer {hb} entry {hq_tag[hb]}")
            if acc_unread[a]:
                raise Fail(f"MMA of tile {kk}: tick {a}'s accumulator not read yet (tile {acc_tag[a]})")
            acc_tag[a], acc_unread[a] = kk, True
            st_rd[s] -= 1
            hq_rd[hb] -= 1
        else:
            op[1].arrive()

    roles = [loader(0), loader(1), issuer0(), issuer1(), epilogue()]
    speed = [rng.choice([0.02, 0.2, 1.0, 1.0, 5.0]) for _ in range(len(roles) + 2)]
    alive = list(range(len(roles)))
    block = [None] * len(roles)
    while alive or async_q or pipe:
        acts, wts = [], []
        for i in alive:
            if block[i] is None or block[i]():
                acts.append(("r", i))
                wts.append(speed[i])
        if async_q:
            acts.append(("dma", None))
            wts.append(speed[-2])
        if pipe:
            acts.append(("tc", None))
            wts.append(speed[-1])
        if not acts:
            raise Fail("deadlock: every role waits, nothing in flight")
        kind, i = rng.choices(acts, weights=wts)[0]
        if rng.random() < 0.02:
            speed[rng.randrange(len(speed))] = rng.choice([0.02, 0.2, 1.0, 1.0, 5.0])
        if kind == "dma":
            what, x, tag = async_q.pop(rng.randrange(len(async_q)))
            if what == "w":
                if st_rd[x]:
                    raise Fail(f"TMA of tile {tag} lands in stage {x} while an MMA reads it")
                st_w[x] = tag
                full[x].complete_tx(32768)
            elif what == "hq":
                hq_tag[x] = tag
                hqfull[x].complete_tx(4096)
            else:
                sbuf_tag[x], sbuf_unread[x] = tag, True
                w2s[x].complete_tx(528)
            continue
        if kind == "tc":
            pipe_step()
            continue
        try:
            block[i] = next(roles[i])
        except StopIteration:
            alive.remove(i)


def once(rng, ctl):
    try:
        run(rng, ctl)
    except Fail as e:
        return str(e)
    return None


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(2111)
    ok = True
    bad, first = 0, None
    for _ in range(runs):
        r = once(rng, None)
        if r is not None:
            bad += 1
            first = first or r
    print(f"W2UE protocol: {runs - bad} / {runs} runs pass{'' if first is None else ': ' + first}")
    ok &= bad == 0
    n = max(runs // 10, 500)
    fails = sum(once(rng, "chunk_before_full") is not None for _ in range(n))
    print(f"  (chunk_before_full: {fails} / {n} runs fail; not required)")
    for ctl in ("no_fence", "cols_fixed", "no_w2s_wait", "no_acc_empty"):
        n = max(runs // 10, 500)
        fails, first = 0, None
        for _ in range(n):
            r = once(rng, ctl)
            if r is not None:
                fails += 1
                first = first or r
        print(f"  control {ctl}: {fails} / {n} runs fail (must be > 0){'' if first is None else ': ' + first}")
        ok &= fails > 0
    print("MODEL OK" if ok else "MODEL FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
