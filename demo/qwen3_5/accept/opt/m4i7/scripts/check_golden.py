#!/usr/bin/env python3
"""Assert that M4-I7's frozen `golden` region is byte-for-byte the pre-M4-I7
moe_fp8_blockscale_sm100.cuh, and that the header advertises the same sha256.

No GPU, no build. Run from anywhere:

    python3 check_golden.py [--base <git-rev>]

`--base` defaults to the commit that introduced M4-I7's parent, i.e. the file as
of `git show <base>:<path>`. The check is what licenses calling the fallback a
"preserved" path rather than a rewritten one.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *([os.pardir] * 6)))
REL = "include/mirage/persistent_kernel/tasks/blackwell/moe_fp8_blockscale_sm100.cuh"

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="5e48eaab",
                help="git rev whose version of the file is the golden source")
a = ap.parse_args()

old = subprocess.run(["git", "-C", REPO, "show", f"{a.base}:{REL}"],
                     capture_output=True, text=True, check=True).stdout
new = open(os.path.join(REPO, REL)).read()

# the pre-M4-I7 frozen region: the constants namespace through the end of the
# task impl, i.e. everything between them and the closing `namespace kernel`.
o0 = old.index("namespace moe_fp8_blockscale {")
o1 = old.index("} // namespace kernel")
want = old[o0:o1].rstrip("\n")

n0 = new.index("namespace golden {\n") + len("namespace golden {\n")
n1 = new.index("\n} // namespace golden")
got = new[n0:n1].strip("\n")

wsha = hashlib.sha256(want.encode()).hexdigest()
gsha = hashlib.sha256(got.encode()).hexdigest()
adv = re.search(r"region sha256:\s*([0-9a-f]{64})", new)

fails = []
print(f"base {a.base}: frozen region {len(want)} bytes sha256 {wsha}")
print(f"HEAD:      golden region {len(got)} bytes sha256 {gsha}")
if got != want:
    fails.append("golden region is NOT byte-identical to the pre-M4-I7 body")
if not adv:
    fails.append("header does not advertise a region sha256")
elif adv.group(1) != gsha:
    fails.append(f"header advertises {adv.group(1)} but the region hashes {gsha}")
else:
    print(f"header advertises: {adv.group(1)}  [matches]")

# the golden body must not have grown a fast-path escape hatch
for banned in ("moe_impl_path", "cp_async_bulk", "mbarrier_", "load_smem_cg"):
    if banned in got:
        fails.append(f"golden region references {banned!r}")

for f in fails:
    print("FAIL:", f)
print("CHECK_GOLDEN:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
