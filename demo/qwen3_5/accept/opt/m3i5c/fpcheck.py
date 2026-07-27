#!/usr/bin/env python3
"""M3-I5c per-function SASS check.

M3-I5b could claim byte-identical SASS. M3-I5c cannot: the fix intends to
change code. The claim here is narrower and is what bit-exactness for the
consumers actually rests on:

  A. every VALUE-PRODUCING floating-point instruction of the routing
     arithmetic is unchanged in count, and
  B. every REDUCTION shuffle (SHFL.BFLY) is unchanged in count,

because the compaction is pure integer/memory work downstream of a
__syncthreads(). If one FADD/FMUL/FFMA/FMNMX/FSEL/FSETP/MUFU.EX2 or one
SHFL.BFLY moved, the routing arithmetic was perturbed and the claim is void.

Three SASS idioms are deliberately NOT counted as floating-point arithmetic,
each verified by reading the operands in this build (see prep.md 3.3):

  * `FLO.U32`               - integer find-leading-one, part of the
                              warp-aggregated-atomic leader election.
  * `HFMA2 Rn, -RZ, RZ, i`  - ptxas's 32-bit immediate-materialisation MOV
                              idiom. Neither router has any fp16 arithmetic.
  * `I2F.U32.RP -> MUFU.RCP -> F2I.FTZ.U32.TRUNC.NTZ` - the unsigned
                              integer-division-by-a-runtime-value idiom. Each
                              occurrence consumes exactly one I2F.U32.RP and
                              one MUFU.RCP, so GENUINE reciprocals (the
                              kernels' `1.f/row_sum`) are counted as
                              #MUFU.RCP - #I2F.U32.RP.

Structural assertions the fix makes falsifiable:
  * POST must contain ZERO atomics (ATOM*/RED*, excluding the REDUX.SYNC
    warp reduction) -- the only atomicAdd in either router was the
    compaction's, and nvcc had warp-aggregated it into
    VOTEU.ANY + POPC + FLO.U32 + SHFL.IDX + ATOMG;
  * POST must contain exactly ONE more BAR.SYNC than PRE.
"""
import re
import sys
from collections import Counter

FUNC = re.compile(r"^\s*Function : (.+?)\s*$")
ADDR = re.compile(r"/\*[0-9a-f]{4}\*/")
ENC = re.compile(r"/\* 0x[0-9a-f]+ \*/")

# value-producing float arithmetic; MUFU.RCP and HFMA2 are handled separately
VALUE = re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FMNMX3|FSEL|FSETP|FCHK"
                   r"|MUFU\.(EX2|LG2|SIN|COS|RSQ|SQRT|RCP64H)"
                   r"|F2F|F2FP|HADD2|HMUL2|DADD|DMUL|DFMA)")
REDUCE = re.compile(r"^SHFL\.(BFLY|UP|DOWN)")
ATOMIC = re.compile(r"^(ATOM|ATOMG|ATOMS|RED)(?!UX)")
MOVIDIOM = re.compile(r"^HFMA2\s+\S+\s*,\s*-?RZ\s*,\s*RZ\s*,")


def split(path):
    out, cur, name = {}, [], None
    for ln in open(path):
        m = FUNC.match(ln)
        if m:
            if name:
                out[name] = cur
            name, cur = m.group(1), []
            continue
        s = ENC.sub("", ADDR.sub("", ln)).strip()
        if s:
            cur.append(s)
    if name:
        out[name] = cur
    return out


def analyse(lines):
    """-> (value_counter, reduce_counter, n_atomic, n_bar, n_hfma2_real)"""
    val, red = Counter(), Counter()
    n_atom = n_bar = 0
    n_rcp = n_i2f = 0
    for s in lines:
        s = s.split(";")[0].strip()
        if not s:
            continue
        t = s.split()
        if t and t[0].startswith("@"):
            t = t[1:]
            s = " ".join(t)
        if not t:
            continue
        op = t[0]
        if op.startswith("HFMA2"):
            if not MOVIDIOM.match(s):
                val["HFMA2(real)"] += 1
            continue
        if op.startswith("MUFU.RCP") and not op.startswith("MUFU.RCP64H"):
            n_rcp += 1
            continue
        if op.startswith("I2F.U32.RP"):
            n_i2f += 1
            continue
        if op.startswith("I2F") or op.startswith("F2I"):
            continue          # remaining int<->float conversions: div idiom
        if VALUE.match(op):
            val[op] += 1
        elif REDUCE.match(op):
            red[op] += 1
        elif ATOMIC.match(op):
            n_atom += 1
        elif op.startswith("BAR"):
            n_bar += 1
    val["MUFU.RCP(genuine)"] = n_rcp - n_i2f
    return val, red, n_atom, n_bar


def main():
    a, b = split(sys.argv[1]), split(sys.argv[2])
    names = sorted(n for n in set(a) & set(b) if "EmptyKernel" not in n)
    extra = sorted(n for n in set(a) ^ set(b) if "EmptyKernel" not in n)
    bad = 0
    for n in names:
        va, ra, aa, ba = analyse(a[n])
        vb, rb, ab, bb = analyse(b[n])
        vok, rok = va == vb, ra == rb
        aok, bok = ab == 0, bb == ba + 1
        ok = vok and rok and aok and bok
        bad += 0 if ok else 1
        print("  %s  %-56s  fp=%s(%d ops)  shfl=%s(%d)  atomics %d->%d  "
              "BAR %d->%d"
              % ("PASS" if ok else "FAIL", n,
                 "SAME" if vok else "DIFF", sum(va.values()),
                 "SAME" if rok else "DIFF", sum(ra.values()),
                 aa, ab, ba, bb))
        if not vok:
            print("        fp delta pre->post: %s"
                  % ", ".join("%s %d->%d" % (k, va[k], vb[k])
                              for k in sorted(set(va) | set(vb))
                              if va[k] != vb[k]))
        if not rok:
            print("        shfl delta pre->post: %s"
                  % ", ".join("%s %d->%d" % (k, ra[k], rb[k])
                              for k in sorted(set(ra) | set(rb))
                              if ra[k] != rb[k]))
        if not aok:
            print("        POST still contains %d atomic(s)" % ab)
        if not bok:
            print("        BAR delta %+d, expected +1" % (bb - ba))
    for n in extra:
        print("  FAIL  function present in only one tree: %s" % n)
        bad += 1
    print("\n  %d/%d functions PASS" % (len(names) - bad, len(names)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
