#!/usr/bin/env python3
"""Is the extra FP instruction in POST a DUPLICATE of one already in PRE, or a
NEW computation?

Canonicalise register names away and compare the multiset of FP instruction
TEXTS. If every post-only text already occurs in pre, the compiler duplicated
an existing guarded block (tail duplication / if-conversion); no new
arithmetic was introduced. If a post-only text has no pre counterpart, the
routing arithmetic genuinely changed and the bit-exactness claim is void.
"""
import re
import sys
from collections import Counter

FUNC = re.compile(r"^\s*Function : (.+?)\s*$")
ADDR = re.compile(r"/\*[0-9a-f]{4}\*/")
ENC = re.compile(r"/\* 0x[0-9a-f]+ \*/")
REG = re.compile(r"\bR\d+\b")
PRED = re.compile(r"\bP\d+\b")
UREG = re.compile(r"\bUR\d+\b")
FP = re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FMNMX3|FSEL|FSETP|MUFU)")


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


def fp_texts(lines):
    out = Counter()
    for s in lines:
        s = s.split(";")[0].strip()
        t = s.split()
        if t and t[0].startswith("@"):
            t = t[1:]
        if not t or not FP.match(t[0]):
            continue
        c = UREG.sub("UR#", REG.sub("R#", PRED.sub("P#", " ".join(t))))
        out[c] += 1
    return out


a, b = split(sys.argv[1]), split(sys.argv[2])
fn = sys.argv[3]
ca, cb = fp_texts(a[fn]), fp_texts(b[fn])
post_only = cb - ca
pre_only = ca - cb
print("=== %s" % fn)
print("distinct FP forms: pre=%d post=%d" % (len(ca), len(cb)))
novel = [k for k in post_only if k not in ca]
print("\nPOST-only surplus (count) -- 'also in PRE' means pure duplication:")
for k, v in sorted(post_only.items()):
    print("  +%d  %-52s  also in PRE: %s (pre count %d)"
          % (v, k, "YES" if k in ca else "NO", ca.get(k, 0)))
print("\nPRE-only surplus:")
for k, v in sorted(pre_only.items()):
    print("  -%d  %-52s  also in POST: %s" % (v, k, "YES" if k in cb else "NO"))
print("\nVERDICT: %s"
      % ("every surplus POST instruction is a duplicate of a form already "
         "present in PRE -- no new arithmetic"
         if not novel else
         "NEW arithmetic forms in POST: %r" % (novel,)))
sys.exit(1 if novel else 0)
