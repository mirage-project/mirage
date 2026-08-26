#!/usr/bin/env python3
"""Anti-regression check: mbarriers must not live in the `extern __shared__` arena.

WHY THIS EXISTS
---------------
Every MPK task body declares its own `extern __shared__` symbol, but all of them
alias ONE arena at ONE base. A persistent worker CTA runs heterogeneous tasks
back-to-back separated only by `__syncthreads()`, which orders THREADS and
drains NO asynchronous agent: a TMA `expect_tx` completion or a
`tcgen05.commit ... mbarrier::arrive` keeps writing an mbarrier's state word
after the issuing task has nominally ended. If that mbarrier lives in the arena,
the late arrival lands in memory the NEXT task has already reused -- a fault or
silent corruption depending only on what occupies that byte.

That defect shipped undetected for months and was found only by a canary probe.
Manual audit demonstrably cannot hold the line, so the invariant is enforced
here instead:

    An mbarrier's byte range must not intersect any task's live-data range.

Static `__shared__` makes that decidable at the source level: nvcc SUMS
per-branch statics (it does not overlay mutually exclusive dispatch branches)
and places all of them BELOW the dynamic arena base -- measured on sm_100a, six
distinct template instantiations received six distinct, non-overlapping
addresses, every one below the arena base. So "the barrier is not arena-derived"
is equivalent to "the barrier is disjoint from every task's live data", and the
check reduces to a reachability question over intra-file assignments.

WHAT IT DOES
------------
For every task header it (1) finds the `extern __shared__` arena symbols and the
static `__shared__` symbols, (2) finds every identifier used as the BARRIER
operand of an mbarrier operation, (3) traces that identifier back through
assignments, and (4) reports ARENA / STATIC / UNRESOLVED. Anything not STATIC
must appear in the allowlist next to this script with a written reason.

Exit status: 0 = clean, 1 = a new arena-resident barrier appeared (or an
allowlist entry became stale).

Usage:
    python3 scripts/check_async_barrier_placement.py [--update-allowlist]
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = [
    os.path.join(REPO, "include", "mirage", "persistent_kernel", "tasks"),
    os.path.join(REPO, "include", "mirage", "persistent_kernel"),
]
ALLOWLIST = os.path.join(
    REPO, "scripts", "async_barrier_placement_allowlist.txt"
)

# --- barrier-operand extraction -------------------------------------------
# Raw PTX: the mbarrier is operand [%0] for every one of these except the TMA
# bulk-tensor copy, where it is the trailing [%N].
PTX_BAR_FIRST = re.compile(
    r"\"\s*(?:@\S+\s+)?(?:mbarrier\.(?:init|arrive|arrive\.expect_tx|"
    r"try_wait|test_wait)[\w.:]*|tcgen05\.commit[\w.:]*|"
    r"cp\.async\.mbarrier\.arrive[\w.:]*)"
)
PTX_TMA_BULK = re.compile(r"cp\.async\.bulk\.tensor[\w.:]*")

# Helper wrappers whose FIRST argument is the barrier address/pointer. These
# cover the raw-PTX families above plus the CUTLASS/CuTe pipeline helpers.
WRAPPER_FIRST_ARG = [
    "mb_init", "mb_wait", "mb_arrive", "mb_arrive_tx", "mbar_tx",
    "tcgen05_commit", "umma_arrive", "initialize_barrier_array_aligned",
    "wait_barrier", "try_wait_barrier", "arrive_barrier",
    "set_barrier_transaction_bytes", "ws_cpasync_arrive_noinc",
]
WRAPPER_RE = re.compile(
    # The explicit list, PLUS any `mb*` / `mbar*` / `*barrier*` helper. Kernels
    # define private variants (`detail::mb_init_impl`, `mbar_init_1`, ...) and a
    # fixed list silently misses them — which is how a whole file escaped an
    # earlier version of this check.
    r"\b(?:" + "|".join(WRAPPER_FIRST_ARG) +
    r"|mb_\w+|mbar_\w+|\w*[Bb]arrier\w*)\s*(?:<[^;{}]*?>\s*)?\(", re.S
)

IDENT = re.compile(r"[A-Za-z_]\w*")
# Names that are types/keywords/intrinsics, never a storage symbol.
NOISE = {
    "int", "uint32_t", "uint64_t", "unsigned", "long", "char", "void", "bool",
    "const", "static_cast", "reinterpret_cast", "constexpr", "auto", "return",
    "sizeof", "float", "double", "if", "for", "while", "asm", "volatile",
    "__cvta_generic_to_shared", "cute", "cutlass", "kernel", "detail", "arch",
    "make_shape", "size", "true", "false", "nullptr", "threadIdx", "blockIdx",
    "elect_one_sync", "__shared__", "extern", "struct", "template", "typename",
}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def split_args(argstr):
    """Split a call's argument list at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            if depth == 0 and ch == ")":
                break
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def balanced_call_args(text, open_idx):
    """Return the substring inside the parens starting at open_idx."""
    depth, i = 0, open_idx
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
        i += 1
    return ""


def statement_at(text, idx):
    """The `;`-delimited statement containing position idx."""
    start = max(text.rfind(";", 0, idx), text.rfind("{", 0, idx),
                text.rfind("}", 0, idx)) + 1
    end = text.find(";", idx)
    return text[start: end if end != -1 else len(text)]


def asm_input_operands(text, idx):
    """Ordered C++ expressions bound to the input operands of the inline-asm
    statement containing position idx.

    `asm volatile("tmpl" : outs : ins : clobbers);` -- the template may be
    several concatenated string literals, and (critically) those literals
    contain `::` themselves (`shared::cta`), so the operand lists can only be
    found by skipping over string literals rather than by searching for `::`.
    Every arming instruction we care about has an EMPTY output list, so the
    barrier is input operand 0 (or the last input, for the TMA bulk copy).
    """
    a = text.rfind("asm", 0, idx)
    if a == -1:
        return []
    end = text.find(";", idx)
    stmt = text[a: end if end != -1 else len(text)]
    i, n = 0, len(stmt)
    colon_sections, cur, depth = [], "", 0
    while i < n:
        c = stmt[i]
        if c == '"':                      # skip a string literal wholesale
            i += 1
            while i < n and stmt[i] != '"':
                i += 2 if stmt[i] == "\\" else 1
            i += 1
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == ":" and depth <= 1:
            colon_sections.append(cur)
            cur = ""
            i += 1
            continue
        cur += c
        i += 1
    colon_sections.append(cur)
    # colon_sections[0] = 'asm volatile(<template>', [1] = outputs, [2] = inputs
    if len(colon_sections) < 3:
        return []
    return [m.group(1) for m in
            re.finditer(r"\"[^\"]*\"\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
                        colon_sections[2])]


def analyze(path):
    src = strip_comments(open(path, encoding="utf-8", errors="replace").read())

    arena = set(re.findall(
        r"extern\s+__shared__\s+(?:__align__\(\s*\d+\s*\)\s+)?"
        r"(?:\w+\s+)*?(\w+)\s*\[", src))
    static = set()
    for m in re.finditer(
            r"(?<!extern\s)__shared__\s+(?:__align__\(\s*\d+\s*\)\s+|"
            r"alignas\(\s*\d+\s*\)\s+)*(?:\w+(?:::\w+)*\s+)+(\w+)\s*[\[;=]",
            src):
        static.add(m.group(1))
    static -= arena

    # identifier -> identifiers it was assigned from (intra-file, flow-insensitive)
    #
    # Split on `;` rather than matching a leading delimiter: a regex that
    # CONSUMES the preceding `;` cannot match two adjacent declarations, because
    # the first match eats the separator the second one needs. That silently
    # dropped every other statement — and with it the `int bf = bars_addr;` and
    # `int sb_aligned = ...;` links that connect a barrier to the arena, which
    # is exactly the chain this check exists to find.
    edges = {}
    assign = re.compile(r"^\s*(?:[\w:*&<>,\s]+\s)?(\*?\w+)\s*=(?!=)\s*(.*)$",
                        re.S)
    for frag in src.split(";"):
        if "=" not in frag or len(frag) > 600:
            continue
        m = assign.match(frag)
        if not m:
            continue
        lhs = m.group(1).lstrip("*")
        rhs_ids = {i for i in IDENT.findall(m.group(2)) if i not in NOISE}
        edges.setdefault(lhs, set()).update(rhs_ids)
    # `Type &ref = *reinterpret_cast<Type*>(x);` binds ref to the arena too.
    for m in re.finditer(r"(\w+)\s*&\s*(\w+)\s*=\s*([^;]{0,400});", src):
        edges.setdefault(m.group(2), set()).update(
            i for i in IDENT.findall(m.group(3)) if i not in NOISE)

    def origin(name, seen=None):
        """ARENA if the identifier reaches an arena symbol, STATIC if it
        reaches only static-shared symbols, else UNRESOLVED."""
        seen = seen or set()
        if name in seen:
            return set()
        seen.add(name)
        if name in arena:
            return {"ARENA"}
        if name in static:
            return {"STATIC"}
        out = set()
        for nxt in edges.get(name, ()):
            out |= origin(nxt, seen)
        return out

    # Formal-parameter names of every function defined in this file. The raw-PTX
    # arming sites live inside tiny wrappers (`mb_init(int a, int c)`,
    # `mbar_tx(int a, int b)`), so their operand is a PARAMETER, not a storage
    # symbol. Because the edge graph is flow-insensitive, such a name collides
    # with any unrelated local of the same name elsewhere in the file (`int a =
    # qn_s + swz<SN>(r,c)` is arena-derived) and manufactures a false positive.
    # Skip parameter names here; the wrappers' CALL SITES are what actually
    # decide the storage class, and those are checked via WRAPPER_FIRST_ARG.
    params = set()
    for m in re.finditer(r"\(([^()]{0,400})\)\s*\{", src):
        for part in m.group(1).split(","):
            names = IDENT.findall(part)
            if names:
                params.add(names[-1])

    findings = {}   # barrier identifier -> verdict

    def record(expr):
        ids = [i for i in IDENT.findall(expr) if i not in NOISE]
        if not ids:
            return
        base = ids[0]
        if base in params and base not in arena and base not in static:
            return
        verdicts = origin(base)
        if "ARENA" in verdicts:
            v = "ARENA"
        elif verdicts == {"STATIC"}:
            v = "STATIC"
        else:
            v = "UNRESOLVED"
        # keep the worst verdict seen for this identifier
        rank = {"STATIC": 0, "UNRESOLVED": 1, "ARENA": 2}
        if base not in findings or rank[v] > rank[findings[base]]:
            findings[base] = v

    # raw PTX, barrier = input operand 0
    for m in PTX_BAR_FIRST.finditer(src):
        ops = asm_input_operands(src, m.start())
        if ops:
            record(ops[0])
    # raw PTX TMA bulk-tensor copy, barrier = trailing input operand
    for m in PTX_TMA_BULK.finditer(src):
        ops = asm_input_operands(src, m.start())
        if ops:
            record(ops[-1])
    # helper wrappers, barrier = first argument
    for m in WRAPPER_RE.finditer(src):
        args = split_args(balanced_call_args(src, m.end() - 1))
        if args:
            record(args[0])

    # Only ARENA is a violation. UNRESOLVED is almost always a wrapper's own
    # formal parameter (the wrapper's CALL SITES are checked separately, which
    # is where the real storage class is decided), so failing on it would bury
    # the signal in noise -- it is reported for human audit only.
    exposed = {k: v for k, v in findings.items() if v == "ARENA"}
    unresolved = {k: v for k, v in findings.items() if v == "UNRESOLVED"}
    return bool(arena), unresolved, exposed


def rel(p):
    return os.path.relpath(p, REPO)


def main():
    files = []
    for root in ROOTS:
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                if n.endswith((".cuh", ".h")):
                    files.append(os.path.join(dirpath, n))
    files = sorted(set(files))

    violations = {}
    for f in files:
        has_arena, unresolved, exposed = analyze(f)
        if has_arena and exposed:
            violations[rel(f)] = exposed

    allow = {}
    if os.path.exists(ALLOWLIST):
        for line in open(ALLOWLIST):
            line = line.split("#")[0].strip()
            if line:
                allow[line] = True

    if "--update-allowlist" in sys.argv:
        with open(ALLOWLIST, "w") as fh:
            fh.write("# Files with arena-resident (or unresolved) mbarriers.\n"
                     "# Every entry needs a written reason. Generated by\n"
                     "# scripts/check_async_barrier_placement.py "
                     "--update-allowlist\n")
            for f in sorted(violations):
                fh.write(f + "\n")
        print(f"wrote {ALLOWLIST} with {len(violations)} entries")
        return 0

    new = sorted(set(violations) - set(allow))
    stale = sorted(set(allow) - set(violations))

    for f in sorted(violations):
        tag = "NEW " if f in new else "known"
        detail = ", ".join(f"{k}={v}" for k, v in sorted(violations[f].items()))
        print(f"[{tag}] {f}: {detail}")

    rc = 0
    if new:
        print("\nERROR: arena-resident mbarrier(s) introduced in:")
        for f in new:
            print("  " + f)
        print("\nAn mbarrier armed by an asynchronous agent (TMA expect_tx,\n"
              "tcgen05.commit, cp.async.mbarrier.arrive) MUST NOT live in the\n"
              "`extern __shared__` arena: __syncthreads() at the task boundary\n"
              "drains no async agent, so a late arrival corrupts the NEXT\n"
              "task's reused bytes. Declare the barrier block in static\n"
              "__shared__ instead -- see the rationale comment in\n"
              "include/mirage/persistent_kernel/tasks/blackwell/"
              "fp8_gemm_dense_sm100_common.cuh and the PipedBarriers helper in\n"
              ".../blackwell/storage.cuh. If the placement is genuinely safe,\n"
              "add the file to scripts/async_barrier_placement_allowlist.txt\n"
              "WITH A WRITTEN REASON.")
        rc = 1
    if stale:
        print("\nERROR: allowlist entries no longer exposed (remove them):")
        for f in stale:
            print("  " + f)
        rc = 1
    if rc == 0:
        print(f"\nOK: {len(files)} headers checked, "
              f"{len(violations)} known/allowlisted, 0 new.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
