"""The loop's automatic checks (docs/megakernel_dev_loop.md section 6). Each returns a list of Findings; a "fail" drops the
candidate (or fails the round for the tree checks), a "flag" goes to review. The rule ids are docs/megakernel_loop_rules.md's.
A rule whose enforcement there is only `auto` fails; one that is `auto` + `review` flags.

  lint_contract(pk)                         every node's declaration (static_megakernel._add_node) is complete and usable, its variants too
  lint_source(paths)                        task templates and generator files: C6, C11, C18, H7, B14
  lint_build(cu_path, so_path, flags)       the generated layer and its build: smem assert, C11 kernel parameters, C13, C24,
                                            registers / static smem / local memory (cuobjdump --dump-resource-usage)
  lint_sass(so_path, parent_summary)        the built kernel's SASS: C14, C15, C11 (CALL), C7 (R2UR), growth vs the parent;
                                            sass_summary / sass_identical (C23)
  lint_plan(pk, plan, costs)                compile_plan's checks (deadlock, per-SM cap, lists), B2 (grids), D6, F3
  lint_records(path, code)                  D2, D8, D9 on a cost file

python -m mirage.mpk.loop.lints tree [--root INCLUDE_DIR] [--baseline F | --write-baseline]   the templates and generator
                                                              files of this tree; findings in the baseline only flag
python -m mirage.mpk.loop.lints build RUN_DIR [--parent SASS_SUMMARY_JSON]   a run's build (run.py --out); writes sass_summary.json
python -m mirage.mpk.loop.lints records FILE [--code TAG]
Exit 1 when any check fails. Imports nothing compiled at import time.
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
MPK_DIR = os.path.dirname(HERE)
REPO_TEMPLATES = os.path.abspath(os.path.join(MPK_DIR, "..", "..", "..", "include", "mirage", "static_megakernel"))
GENERATOR_FILES = ("compiler.py", "static_schedule.py", "search.py", "solver.py", "cost_records.py", "static_harness.py")
# B14: the K3 MoE layer's sizes (hidden, latent, experts, routed / shared intermediate per GPU at TP8); none belongs in the
# generator, whose inputs are only the graph and the schedule
PART_CONSTANTS = (7168, 3584, 896, 384, 768)
# what static_megakernel._add_node keeps per node (compiler.py and the code generator read these)
NODE_KEYS = ("name", "input_maps", "changeable_grid_dims", "rows_split_over_gpus", "gpu_split_blocks", "one_task_per_sm",
             "concurrent_along_axis", "cluster_pair_params", "buf", "out_buffer", "exchange", "counter", "buf_slots",
             "map_slots", "dry", "cost_inputs", "variants", "variant_defaults", "stage0_inputs")
# gemm_tile's combine modes: static_megakernel.COMBINE's values (copied, so this module imports without the extension)
COMBINE_MODES = (0, 1, 2)
RECORD_KEYS = ("task_type", "key", "context", "median_us", "n", "run", "machine", "code")


@dataclass
class Finding:
    rule: str      # e.g. "C6"
    level: str     # "fail" or "flag"
    where: str     # file:line, node, record, ...
    msg: str

    def __str__(self):
        return f"{self.level.upper():4} {self.rule:4} {self.where}: {self.msg}"


def failed(findings: List[Finding]) -> bool:
    return any(f.level == "fail" for f in findings)


# ---------------- contract ----------------
def lint_contract(pk) -> List[Finding]:
    """Every node of pk (a StaticMegakernel, or anything compiler.graph_nodes reads): its declaration has every field
    _add_node writes (B1-B5); it has at least one candidate grid (B1: the cut dimensions divide into whole units); every
    changeable grid axis cuts some tensor; cost_inputs name inputs; concurrent_along_axis is a grid axis; each declared variant
    is (a params index, a list of int values) with a default; a gemm_tile's combine mode is one of the three (B4)."""
    from .. import compiler
    out = []
    for idx, info in sorted(pk._static_nodes.items()):
        missing = [k for k in NODE_KEYS if k not in info]
        if missing:
            out.append(Finding("B1", "fail", f"node {idx} ({info.get('name')})", f"declaration misses {missing}"))
    if failed(out):
        return out
    for n in compiler.graph_nodes(pk):
        where = f"node {n.graph_idx} ({n.name})"
        if len(n.info["input_maps"]) != len(n.inputs) + len(n.outputs):
            out.append(Finding("B1", "fail", where, f"{len(n.info['input_maps'])} input maps for {len(n.inputs) + len(n.outputs)} "
                                                    f"tensors"))
            continue
        for axis in n.info["changeable_grid_dims"]:
            if axis not in (0, 1, 2) or not any(m[axis] >= 0 for m in n.info["input_maps"]):
                out.append(Finding("B1", "fail", where, f"changeable grid axis {axis} cuts no tensor"))
        try:
            if not compiler.candidate_grids(n):
                out.append(Finding("B1", "fail", where, "no candidate grid (no size gives whole units of every cut dimension)"))
        except ValueError as e:
            out.append(Finding("B1", "fail", where, str(e)))
        bad = [i for i in n.info["cost_inputs"] if not 0 <= i < len(n.inputs)]
        if bad:
            out.append(Finding("B1", "fail", where, f"cost_inputs {bad} are not inputs (it has {len(n.inputs)})"))
        a = n.info["concurrent_along_axis"]
        if a is not None and a not in (0, 1, 2):
            out.append(Finding("B1", "fail", where, f"concurrent_along_axis {a} is not a grid axis"))
        if n.info["cluster_pair_params"] is not None and a is None:
            out.append(Finding("F3", "fail", where, "cluster_pair_params without concurrent_along_axis"))
        for name, spec in n.info["variants"].items():   # {name: (params index, [values])}
            ok = (isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], int) and 0 <= spec[0] < len(n.params)
                  and spec[1] and all(isinstance(v, int) for v in spec[1]))
            if not ok:
                out.append(Finding("B1", "fail", where, f"variant {name} {spec}: not (a params index below {len(n.params)}, "
                                                        f"a list of int values)"))
            elif name not in n.info["variant_defaults"]:
                out.append(Finding("B1", "fail", where, f"variant {name} has no default"))
        if n.name == "gemm_tile" and (not n.params or n.params[0] not in COMBINE_MODES):
            out.append(Finding("B4", "fail", where, f"combine mode {n.params[:1]} is not one of {COMBINE_MODES}"))
        if n.info["one_task_per_sm"] and n.grid[0] != pk._num_sms():
            out.append(Finding("B1", "fail", where, f"one task per SM, grid x {n.grid[0]} for {pk._num_sms()} SMs"))
    return out


# ---------------- source ----------------
def strip_comments(text: str) -> str:
    """C/C++ text with // and /* */ comments replaced by spaces (newlines kept, so line numbers stay); literals untouched."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def loop_depths(code: str) -> List[bool]:
    """Per character of comment-free code: whether it is inside the body of a for / while / do loop (by braces: a { whose
    text since the last ; { or } starts a loop opens a loop body)."""
    inside = [False] * len(code)
    stack, last, parens = [], 0, 0
    for i, c in enumerate(code):
        parens += (c == "(") - (c == ")")
        if c == "{":
            head = code[last:i]
            stack.append(bool(re.search(r"\b(for|while|do)\b", head)) or (bool(stack) and stack[-1]))
            last = i + 1
        elif c == "}":
            if stack:
                stack.pop()
            last = i + 1
        elif c == ";" and parens == 0:       # not the ; of a for header
            last = i + 1
        inside[i] = bool(stack) and stack[-1]
    return inside


def lint_cuda_source(path: str, text: Optional[str] = None) -> List[Finding]:
    """One template file (.cuh):
      C6   a static __shared__ declaration (not extern): fail in tasks/ (a task template), flag in the shared runtime files
      C11  a __device__ function without __forceinline__: fail; one declared __noinline__ on purpose: flag (check no CALL in a
           hot path); constexpr helpers are not looked at
      C18  __threadfence inside a loop: flag
      H7   an asm statement reading %globaltimer without a "memory" clobber: flag"""
    text = open(path).read() if text is None else text
    code = strip_comments(text)
    name = os.path.basename(path)
    in_tasks = os.path.basename(os.path.dirname(path)) == "tasks"
    out = []
    for m in re.finditer(r"\b__shared__\b", code):
        before = code[max(0, m.start() - 40):m.start()]
        if re.search(r"\bextern\s+$", before):
            continue
        decl = code[m.start():code.find(";", m.start())]
        out.append(Finding("C6", "fail" if in_tasks else "flag", f"{name}:{line_of(code, m.start())}",
                           f"static shared memory: {' '.join(decl.split())[:80]}"))
    for m in re.finditer(r"\b__device__\b", code):
        head_end = min([k for k in (code.find("(", m.end()), code.find(";", m.end()), code.find("=", m.end())) if k >= 0]
                       or [len(code)])
        if head_end >= len(code) or code[head_end] != "(":
            continue                                    # a __device__ variable
        line_start = code.rfind("\n", 0, m.start()) + 1
        head = code[line_start:head_end]
        if re.search(r"\bconstexpr\b", head) or "__forceinline__" in head:
            continue
        fn = re.findall(r"(\w+)\s*$", head)
        where = f"{name}:{line_of(code, m.start())}"
        if "__noinline__" in head:
            out.append(Finding("C11", "flag", where, f"{fn[0] if fn else '?'} is __noinline__ (out of line on purpose? no CALL in "
                                                    f"a hot path: lint_sass)"))
        else:
            out.append(Finding("C11", "fail", where, f"{fn[0] if fn else '?'} is not __forceinline__"))
    inside = loop_depths(code)
    for m in re.finditer(r"\b__threadfence\w*\s*\(", code):
        if inside[m.start()]:
            out.append(Finding("C18", "flag", f"{name}:{line_of(code, m.start())}",
                               "__threadfence inside a loop (one bar.sync + one red.release.gpu instead)"))
    for m in re.finditer(r"\basm\b[^;]*?globaltimer[^;]*;", code, re.S):
        if '"memory"' not in m.group(0):
            out.append(Finding("H7", "flag", f"{name}:{line_of(code, m.start())}", "globaltimer read without a \"memory\" clobber"))
    return out


def lint_python_source(path: str, text: Optional[str] = None) -> List[Finding]:
    """One generator file (.py):
      B14  a part constant (PART_CONSTANTS) as a number in the code (comments and strings are not code): flag
      H7   a generated asm statement reading %globaltimer without a "memory" clobber (in a string of the generator): flag"""
    text = open(path).read() if text is None else text
    name = os.path.basename(path)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NUMBER:
            try:
                v = int(tok.string, 0)
            except ValueError:
                continue
            if v in PART_CONSTANTS:
                out.append(Finding("B14", "flag", f"{name}:{tok.start[0]}", f"part constant {v} in the generator"))
        elif tok.type == tokenize.STRING and "globaltimer" in tok.string and "asm" in tok.string:
            if "memory" not in tok.string:
                out.append(Finding("H7", "flag", f"{name}:{tok.start[0]}", "generated globaltimer read without a \"memory\" clobber"))
    return out


def lint_source(paths: List[str]) -> List[Finding]:
    out = []
    for p in paths:
        out += lint_python_source(p) if p.endswith(".py") else lint_cuda_source(p)
    return out


def tree_paths(root: str = REPO_TEMPLATES) -> List[str]:
    """The templates (root/*.cuh, root/tasks/*.cuh) and the generator files."""
    return (sorted(glob.glob(os.path.join(root, "*.cuh")) + glob.glob(os.path.join(root, "tasks", "*.cuh")))
            + [os.path.join(MPK_DIR, f) for f in GENERATOR_FILES])


# ---------------- build ----------------
def parse_resource_usage(text: str) -> Dict[str, dict]:
    """`cuobjdump --dump-resource-usage` text -> {function: {"REG", "STACK", "SHARED", "LOCAL", ...}} plus "arch" per function:
      arch = sm_100a
      Function _Z12layer_kernelN9static_mk4MapsENS_1GEPK10StaticTask:
        REG:168 STACK:0 SHARED:2312 LOCAL:0 CONSTANT[0]:1424 TEXTURE:0 SURFACE:0 SAMPLER:0"""
    out, arch, fn = {}, None, None
    for line in text.splitlines():
        m = re.match(r"\s*arch\s*=\s*(\S+)", line)
        if m:
            arch = m.group(1)
        m = re.match(r"\s*Function\s+([^:\s]+)\s*:", line)
        if m:
            fn = m.group(1)
            out[fn] = {"arch": arch}
            continue
        if fn and re.search(r"\bREG:\d+", line):
            out[fn].update({k: int(v) for k, v in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", line)})
            fn = None
    return out


def lint_build(cu_path: str, so_path: Optional[str], flags: List[str] = (), resource_text: Optional[str] = None,
               parent: Optional[dict] = None) -> Tuple[List[Finding], dict]:
    """The generated layer.cu and its build. Returns (findings, {"registers", "static_smem", "local", "arch"} of layer_kernel).
      C5/C6  layer.cu asserts STATIC_SMEM_BYTES <= 227 KB (smem_code); layer_kernel's static shared memory (.nv.shared) over
             1 KB flagged (crossing 1 KB lowered the dynamic maximum)
      C11    layer_kernel's Maps and G parameters are __grid_constant__ (G: flag; the generator passes G by value today)
      C13    no -use_fast_math in the flags
      C24    the arch of a tcgen05 build is sm_10Xa (in the flags given, and in the cubin)
      C15    layer_kernel uses no local memory (LOCAL > 0: fail)
      C20    registers per thread reported, compared with the parent build's (parent: a summary returned here before)"""
    out, summary = [], {}
    code = open(cu_path).read() if cu_path and os.path.exists(cu_path) else ""
    where = os.path.basename(cu_path or "layer.cu")
    if code:
        if not re.search(r"static_assert\(\s*STATIC_SMEM_BYTES\s*<=\s*227\s*\*\s*1024", code):
            out.append(Finding("B5", "fail", where, "no static_assert(STATIC_SMEM_BYTES <= 227 * 1024)"))
        sig = re.search(r"__global__[^{]*\blayer_kernel\s*\(([^)]*)\)", code)
        if not sig:
            out.append(Finding("C11", "fail", where, "no layer_kernel"))
        else:
            for typ, level in (("Maps", "fail"), ("G", "flag")):
                p = next((a for a in sig.group(1).split(",") if re.search(rf"\bstatic_mk::{typ}\b", a)), None)
                if p is None or "__grid_constant__" not in p:
                    out.append(Finding("C11", level, where, f"layer_kernel's {typ} parameter is not __grid_constant__: {(p or '').strip()}"))
    joined = " ".join(flags or [])
    if re.search(r"(^|\s)-?-use_fast_math\b", joined):
        out.append(Finding("C13", "fail", "flags", "-use_fast_math"))
    for arch in sorted(set(re.findall(r"(?:arch=compute_|code=sm_|-arch=sm_)(\d+)(a?)", joined))):
        if arch[0].startswith("10") and not arch[1]:
            out.append(Finding("C24", "fail", "flags", f"tcgen05 build for sm_{arch[0]} without the 'a' (sm_{arch[0]}a)"))
    if resource_text is None and so_path and shutil.which("cuobjdump"):
        resource_text = subprocess.run(["cuobjdump", "--dump-resource-usage", so_path], capture_output=True, text=True).stdout
    if resource_text:
        usage = parse_resource_usage(resource_text)
        kernel = next((u for fn, u in usage.items() if "layer_kernel" in fn), None)
        if kernel is None:
            out.append(Finding("C11", "fail", "resource usage", "no layer_kernel in the build"))
        else:
            summary = {"registers": kernel.get("REG"), "static_smem": kernel.get("SHARED"), "local": kernel.get("LOCAL"),
                       "stack": kernel.get("STACK"), "arch": kernel.get("arch")}
            if kernel.get("LOCAL", 0) > 0 or kernel.get("STACK", 0) > 0:
                out.append(Finding("C15", "fail", "layer_kernel", f"local memory {kernel.get('LOCAL')} B, stack {kernel.get('STACK')} B"))
            if kernel.get("SHARED", 0) > 1024:
                out.append(Finding("C6", "flag", "layer_kernel", f"static shared memory {kernel['SHARED']} B (> 1 KB)"))
            a = kernel.get("arch") or ""
            if re.fullmatch(r"sm_10\d", a):
                out.append(Finding("C24", "fail", "layer_kernel", f"built for {a}, not {a}a"))
            if parent and parent.get("registers") and summary["registers"] and summary["registers"] > parent["registers"]:
                out.append(Finding("C20", "flag", "layer_kernel", f"registers {parent['registers']} -> {summary['registers']}"))
    return out, summary


# ---------------- SASS ----------------
SASS_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T\d]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)")
FP64_OPS = re.compile(r"^(DADD|DMUL|DFMA|DSETP|DSET|DMNMX|DMMA)\b|^(F2F|I2F|F2I)\b.*F64")
MMA_OPS = re.compile(r"^UTC\w*MMA|^[HQIO]MMA|MMA\b")


def sass_functions(text: str) -> Dict[str, List[str]]:
    """`cuobjdump -sass` text -> {function name: [opcodes in order]} (opcodes with their modifiers, e.g. LDG.E.128)."""
    out, fn = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if m:
            fn = m.group(1)
            out[fn] = []
            continue
        m = SASS_LINE.search(line)
        if m and fn is not None:
            out[fn].append(m.group(1))
    return out


def sass_summary(text: str, function: str = "layer_kernel") -> dict:
    """Counts over the functions whose name contains `function` (all functions when none does): instructions (NOPs not
    counted), LDL, STL, fp64 ops, CALL, R2UR, MMA sites."""
    fns = sass_functions(text)
    ops = [op for fn, o in fns.items() if function in fn for op in o] or [op for o in fns.values() for op in o]
    base = lambda op: op.split(".")[0]
    return {"functions": [fn for fn in fns if function in fn] or list(fns),
            "instructions": sum(base(op) != "NOP" for op in ops),
            "ldl": sum(base(op) == "LDL" for op in ops), "stl": sum(base(op) == "STL" for op in ops),
            "fp64": sum(bool(FP64_OPS.search(op)) for op in ops),
            "call": sum(base(op).startswith("CALL") for op in ops),
            "r2ur": sum(base(op) == "R2UR" for op in ops),
            "mma_sites": sum(bool(MMA_OPS.search(base(op))) for op in ops)}


def lint_sass(so_path: Optional[str], sass_text: Optional[str] = None, parent_summary: Optional[dict] = None
              ) -> Tuple[List[Finding], dict]:
    """The built kernel's SASS (cuobjdump -sass so_path, or sass_text). Returns (findings, sass_summary):
      C15  LDL / STL (local memory: spills, runtime-indexed register arrays): fail
      C14  fp64 instructions: fail
      C11  CALL (a body not inlined): flag
      C7   R2UR count reported (vs the parent: more flagged)
      growth vs the parent build's summary: instructions or MMA sites up by more than 5%: flag"""
    if sass_text is None:
        if not so_path or not shutil.which("cuobjdump"):
            return [Finding("C15", "flag", "sass", "no cuobjdump: SASS not checked")], {}
        sass_text = subprocess.run(["cuobjdump", "-sass", so_path], capture_output=True, text=True).stdout
    s = sass_summary(sass_text)
    out = []
    if s["ldl"] or s["stl"]:
        out.append(Finding("C15", "fail", "sass", f"local memory: {s['ldl']} LDL, {s['stl']} STL"))
    if s["fp64"]:
        out.append(Finding("C14", "fail", "sass", f"{s['fp64']} fp64 instructions"))
    if s["call"]:
        out.append(Finding("C11", "flag", "sass", f"{s['call']} CALL instructions (a function not inlined)"))
    if parent_summary:
        for k, rule in (("instructions", "C23"), ("mma_sites", "C23"), ("r2ur", "C7")):
            a, b = parent_summary.get(k), s[k]
            if a and b > a * (1.05 if k != "r2ur" else 1.0):
                out.append(Finding(rule, "flag", "sass", f"{k} {a} -> {b} ({100 * (b - a) / a:+.1f}%) vs the parent build"))
    return out, s


def sass_identical(text_a: str, text_b: str, function: str = "layer_kernel") -> bool:
    """C23: the same instructions in the same order, register numbers and addresses aside (a flagged variant's default build
    vs the baseline's)."""
    def norm(text):
        fns = {fn: o for fn, o in sass_functions(text).items() if function in fn}
        return [op for fn in sorted(fns) for op in fns[fn]]
    return norm(text_a) == norm(text_b)


# ---------------- plan ----------------
def lint_plan(pk, plan: dict, costs=None) -> List[Finding]:
    """A plan for pk (compiler.compile_plan's format). Regrids pk's graph to the plan's grids, as compile_plan does.
      B2   every grid is one of its node's candidate grids (they hold the declared minimum work per task)
      F3   the cluster pairs fit (compiler.pair_clusters) when the plan does not ask for the plain launch: flag (the plain
           launch is taken then)
      D6   with costs (an object with task_us / pool_us, e.g. search.Costs): every placed node's chosen grid has a time
      the compile checks: lists (every task once, one-per-SM tasks on their SM, concurrent groups apart), the per-SM cap and
      the deadlock check (static_schedule.check_deadlock_free), in a temporary directory"""
    from .. import compiler
    out = []
    try:
        nodes = compiler.apply_grids(pk, {int(k): v for k, v in plan["grids"].items()})
    except ValueError as e:
        return [Finding("B2", "fail", "grids", str(e))]
    paired = {n.graph_idx for n in nodes if compiler.can_pair(n)}
    if paired and plan.get("cluster_launch", True):
        for gpu in range(pk.num_gpus):
            try:
                compiler.pair_clusters(nodes, paired, plan["lists"][gpu], pk._num_sms(), gpu)
            except (ValueError, IndexError) as e:
                out.append(Finding("F3", "flag", f"GPU {gpu}", f"cluster pairs do not fit, plain launch: {e}"))
                break
    if costs is not None:
        from .. import search
        for n in nodes:
            if n.info["one_task_per_sm"] and costs.pool_us(n) is not None:
                continue
            if costs.task_us(n, n.grid, search.producer_grids(n, nodes)) is None:
                out.append(Finding("D6", "fail", f"node {n.graph_idx} ({n.name})", f"grid {n.grid}: no measured records"))
    with tempfile.TemporaryDirectory() as d:
        try:
            compiler.compile_plan(pk, plan, d)
        except ValueError as e:
            out.append(Finding("B2", "fail", "compile_plan", str(e)))
    return out


# ---------------- records ----------------
def lint_records(path: str, code: Optional[str] = None) -> List[Finding]:
    """A cost file (cost_records.py): every record has RECORD_KEYS (D8); its context has first_on_sm and prev (D2; the pool's
    record has an empty context and is not looked at); its code is `code` when given (D9: records of other kernel code are
    not used). One finding per kind of problem, with the count."""
    with open(path) as f:
        records = json.load(f)["records"]
    out, name = [], os.path.basename(path)
    missing = [r for r in records if any(k not in r for k in RECORD_KEYS)]
    if missing:
        keys = sorted({k for r in missing for k in RECORD_KEYS if k not in r})
        out.append(Finding("D8", "fail", name, f"{len(missing)} of {len(records)} records miss fields {keys}"))
    no_ctx = [r for r in records if r.get("context") and any(k not in r["context"] for k in ("first_on_sm", "prev"))]
    if no_ctx:
        out.append(Finding("D2", "fail", name, f"{len(no_ctx)} records without first_on_sm / prev in their context"))
    if code is not None:
        other = [r for r in records if r.get("code") != code]
        if other:
            codes = sorted({str(r.get("code")) for r in other})
            out.append(Finding("D9", "flag", name, f"{len(other)} of {len(records)} records measured other kernel code {codes} "
                                                   f"(current {code})"))
    return out


# ---------------- baseline ----------------
BASELINE = os.path.join(HERE, "lint_baseline.json")


def finding_key(f: Finding) -> str:
    """A finding without its line number (lines move when code above changes): rule, file, message."""
    return f"{f.rule}|{os.path.basename(f.where.split(':')[0])}|{f.msg}"


def apply_baseline(findings: List[Finding], path: str = BASELINE) -> List[Finding]:
    """Findings already in the baseline file (the tree's findings when the loop started: existing code is not blocked, new code
    is) become flags marked "known"; new ones keep their level. No baseline file: unchanged."""
    if not os.path.exists(path):
        return findings
    with open(path) as f:
        known = set(json.load(f)["findings"])
    return [Finding(x.rule, "flag", x.where, "known at baseline: " + x.msg) if finding_key(x) in known else x for x in findings]


def write_baseline(findings: List[Finding], path: str = BASELINE) -> None:
    with open(path, "w") as f:
        json.dump({"findings": sorted({finding_key(x) for x in findings})}, f, indent=0)


# ---------------- CLI ----------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    t = sub.add_parser("tree")
    t.add_argument("--root", default=REPO_TEMPLATES, help="include/mirage/static_megakernel")
    t.add_argument("--baseline", default=BASELINE, help="known findings: flags, not fails (default lint_baseline.json)")
    t.add_argument("--write-baseline", action="store_true", help="write this tree's findings as the baseline")
    b = sub.add_parser("build")
    b.add_argument("run_dir")
    b.add_argument("--parent", default=None, help="the parent build's sass_summary.json")
    r = sub.add_parser("records")
    r.add_argument("file")
    r.add_argument("--code", default=None, help="the current code tag (static_schedule.code_tag)")
    args = p.parse_args(argv)
    if args.what == "tree":
        findings = lint_source(tree_paths(args.root))
        if args.write_baseline:
            write_baseline(findings, args.baseline)
            print(f"baseline: {len(findings)} findings -> {args.baseline}")
        findings = apply_baseline(findings, args.baseline)
    elif args.what == "build":
        so = (glob.glob(os.path.join(args.run_dir, "build", "layer*.so")) or [None])[0]
        res_path = os.path.join(args.run_dir, "result.json")
        flags = json.load(open(res_path)).get("nvcc_flags", []) if os.path.exists(res_path) else []
        parent = json.load(open(args.parent)) if args.parent else None
        findings, build = lint_build(os.path.join(args.run_dir, "build", "layer.cu"), so, flags, parent=(parent or {}).get("build"))
        sass_findings, sass = lint_sass(so, parent_summary=(parent or {}).get("sass"))
        findings += sass_findings
        with open(os.path.join(args.run_dir, "sass_summary.json"), "w") as f:
            json.dump({"build": build, "sass": sass, "findings": [asdict(x) for x in findings]}, f, indent=1)
    else:
        findings = lint_records(args.file, args.code)
    for f in findings:
        print(f)
    print(f"{len(findings)} findings, {sum(f.level == 'fail' for f in findings)} fail")
    return 1 if failed(findings) else 0


if __name__ == "__main__":
    sys.exit(main())
