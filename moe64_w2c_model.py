"""CPU model of STATIC_MOE64_W2C (tasks/scaled_moe_experts.cuh): the h_q rows one SM writes and another SM's W2 reads.

What is modelled, per randomized step (T = 64 tokens, the GLM-5.3 TP8 bank or the GLM-5.3-Flash TP4 bank):
  1. routing: each token picks TOPK distinct routed experts (skewed, as real routing), each split into DUP sub-experts,
     plus the SHARED shared sub-experts (every token)
  2. build_table on two SMs (writer W: a W13 entry's SM; reader R: a W2 entry's SM): slots = live experts in id order,
     start = prefix sum of the counts, the slot's list order: TOKEN_ORDER (popcount of the slot's token mask below the
     token) or the old shared-atomic order (a random permutation per SM)
  3. writer: hq_out of every (slot, list place j, feature f in 0..IR) into the planes [KT2][NSLOT T][128 B], row
     slot T + j, byte hq_swz(f, row) (f >= 128: plane 1)
  4. reader: w2_load_hq: expect = KT2 x ceil(nt / 16) x 16 x 128 B, one bulk copy a plane of that many bytes from row
     slot T into the buffer's K tile (a 1024-B aligned base); the tx count must reach the expect exactly
  5. the MMA's B operand: row c of K tile kt read through the 128-B swizzle (chunk k at (k ^ (c & 7)) of row c), for
     c < N = 16 ceil(nt / 16); accumulator column c = B row c
  6. the W2 epilogue: list entry i of the reader's table reads column i - start (W2C) and stores to row (tok[i], kk[i]):
     the bytes must be token tok[i]'s h_q (the value the writer produced for that token)
Negative controls (each must fail in most runs): the old atomic list order with W2C rows; the writer without the
swizzle; plane rows from start[slot] (not a multiple of 8) with the swizzle by global row; ng = floor(nt / 16); the
old constant expect (16 KB) with the W2C copies.
Run: python3 moe64_w2c_model.py [runs]   (exit 0: the W2C model passes every run and every control fails)
"""
import itertools
import random
import sys

T, W2C_N = 64, 16


def route(rng, cfg):
    """the pairs (token, k) -> bank expert, k = the pair's place in the token's K"""
    e_routed, topk, dup, shared = cfg["E"], cfg["TOPK"], cfg["DUP"], cfg["SHARED"]
    # skewed popularity (real routing: ~3.7-4.5 tokens an expert at bs 64 on GLM-5.3), a fresh draw each step
    cw = list(itertools.accumulate(rng.paretovariate(1.2) for _ in range(e_routed)))
    pairs = []
    for t in range(T):
        chosen = set()
        while len(chosen) < topk:
            for e in rng.choices(range(e_routed), cum_weights=cw, k=2 * topk):
                if len(chosen) < topk:
                    chosen.add(e)
        ks = []
        for e in sorted(chosen, key=lambda _: rng.random()):   # topk's order of a token's experts: arbitrary
            ks += [e * dup + d for d in range(dup)]
        ks += [e_routed * dup + s for s in range(shared)]
        pairs.append(ks)
    return pairs   # pairs[t][k] = bank expert


def build_table(pairs, rng, token_order):
    K = len(pairs[0])
    live = sorted({e for row in pairs for e in row})
    slot_of = {e: s for s, e in enumerate(live)}
    ntok = [0] * len(live)
    for row in pairs:
        for e in row:
            ntok[slot_of[e]] += 1
    start, run = [], 0
    for c in ntok:
        start.append(run)
        run += c
    tok, kk = [None] * run, [None] * run
    if token_order:   # popcount of the slot's mask below the token (a token at most once in a slot)
        mask = [0] * len(live)
        for t, row in enumerate(pairs):
            for e in row:
                mask[slot_of[e]] |= 1 << t
        for t, row in enumerate(pairs):
            for k, e in enumerate(row):
                s = slot_of[e]
                pos = start[s] + bin(mask[s] & ((1 << t) - 1)).count("1")
                assert tok[pos] is None
                tok[pos], kk[pos] = t, k
    else:   # the old order: each slot's pairs in the order the SM's threads won the shared atomic
        fill = [0] * len(live)
        order = [(t, k) for t in range(T) for k in range(K)]
        rng.shuffle(order)
        for t, k in order:
            s = slot_of[pairs[t][k]]
            pos = start[s] + fill[s]
            fill[s] += 1
            tok[pos], kk[pos] = t, k
    return dict(live=live, ntok=ntok, start=start, tok=tok, kk=kk)


def hq_swz(f, r):
    return (((f >> 4) ^ (r & 7)) << 4) | (f & 15)


def value(slot, t, f):
    """the h_q byte of (slot, token t, feature f) as the W13 epilogue computes it: any injective stand-in"""
    return (slot, t, f)


# the features checked: every 16-B chunk of both K tiles, first and last byte of a chunk, both halves (f, f + 64)
FEATS = sorted({c * 16 + b for c in range(16) for b in (0, 15)} | {5, 77, 130, 201})


def run_once(rng, cfg, ctl):
    IR = cfg["IR"]
    KT2 = IR // 128
    pairs = route(rng, cfg)
    tw = build_table(pairs, rng, ctl != "atomic_order")   # the writer SM's table
    tr = build_table(pairs, rng, ctl != "atomic_order")   # the reader SM's table (its own atomics)
    nslot = len(tw["live"])
    plane_rows = cfg["NSLOT"] * T
    assert nslot <= cfg["NSLOT"]
    planes = {}   # (kt, row, byte) -> value

    def row0(tab, s):
        return tab["start"][s] if ctl == "start_rows" else s * T

    # 3. the writer's hq_out (every slot: each slot's W13 entries run on some SM, all with this SM's view here)
    for s in range(nslot):
        nt = tw["ntok"][s]
        for j in range(nt):
            t = tw["tok"][tw["start"][s] + j]
            r = row0(tw, s) + j
            for f in FEATS:
                if f >= IR:
                    continue
                kt, fb = f // 128, f % 128
                o = fb if ctl == "no_swizzle" else hq_swz(fb, r)
                assert r < plane_rows
                planes[(kt, r, o)] = value(s, t, f)
    # 4.-6. the reader: every slot's W2 entry
    for s in range(nslot):
        nt = tr["ntok"][s]
        ng = nt // W2C_N if ctl == "ng_floor" else (nt + W2C_N - 1) // W2C_N
        by = ng * W2C_N * 128
        expect = KT2 * (16384 // 2 if ctl == "old_expect" else by)   # the old expect: W2_HQB / KT2 a plane x KT2
        tx = KT2 * by   # the bulk copies: by bytes of each plane from row row0 on, into K tile kt (b -> plane byte b)

        def smem(kt, b):
            return planes.get((kt, row0(tr, s) + b // 128, b % 128)) if b < by else "stale"
        if tx != expect:
            return f"tx {tx} != expect {expect} (slot {s}, nt {nt})"
        n_mma = W2C_N * ng
        for i in range(tr["start"][s], tr["start"][s] + nt):
            c = i - tr["start"][s]   # the epilogue's column (W2C)
            t = tr["tok"][i]
            if c >= n_mma:
                return f"column {c} past the MMA's N {n_mma} (slot {s}, nt {nt})"
            for f in FEATS:   # B row c through the 128-B swizzle (rows from a 1024-B aligned base)
                if f >= IR:
                    continue
                kt, fb = f // 128, f % 128
                got = smem(kt, c * 128 + hq_swz(fb, c))
                if got != value(s, t, f):
                    return f"slot {s} column {c}: token {t} feature {f} reads {got}"
    return None


CFGS = {
    # GLM-5.3 TP8: 256 experts, top 8, the shared expert in the bank (K 9), IR 256; NSLOT = min(64 x 9, 288)
    "glm53_tp8": dict(E=256, TOPK=8, DUP=1, SHARED=1, IR=256, NSLOT=288),
    # GLM-5.3-Flash TP4: 288 experts as 2 sub-experts of IR 256, top 8, 2 shared sub-experts (K 18); NSLOT = min(1152, 608)
    "flash_tp4": dict(E=288, TOPK=8, DUP=2, SHARED=2, IR=256, NSLOT=608),
}


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(1121)
    ok = True
    for name, cfg in CFGS.items():
        bad = sum(run_once(rng, cfg, None) is not None for _ in range(runs))
        print(f"{name} W2C: {runs - bad} / {runs} runs pass")
        ok &= bad == 0
        for ctl in ("atomic_order", "no_swizzle", "start_rows", "ng_floor", "old_expect"):
            n = max(runs // 100, 200)
            fails = 0
            first = None
            for _ in range(n):
                r = run_once(rng, cfg, ctl)
                if r is not None:
                    fails += 1
                    first = first or r
            print(f"  control {ctl}: {fails} / {n} runs fail (must be > 0){'' if first is None else ': ' + first}")
            ok &= fails > 0
    print("MODEL OK" if ok else "MODEL FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
