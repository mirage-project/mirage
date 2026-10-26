"""Tests of loop/lints.py and static_schedule.code_tag, without a GPU or the built Mirage extension:
  python tests/static_loop/test_lints.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes   # noqa: E402

fakes.import_mpk()
from mirage.mpk import static_schedule   # noqa: E402
from mirage.mpk.loop import explore, lints   # noqa: E402

TEMPLATES = os.path.join(fakes.ROOT, "include", "mirage", "static_megakernel")


def test_code_tag():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "static_megakernel")
        shutil.copytree(TEMPLATES, root)
        tag = static_schedule.code_tag(root)
        assert tag == static_schedule.code_tag(TEMPLATES) and len(tag) == 12
        path = os.path.join(root, "tasks", "sum_gpus.cuh")
        text = open(path).read()
        # whitespace and comments only: the same tag
        open(path, "w").write("// a new comment\n" + text.replace("{\n", "{  /* x */\n\n").replace(" = ", "="))
        assert static_schedule.code_tag(root) == tag, "a formatting / comment change moved the tag"
        # a code change: a new tag
        open(path, "w").write(text + "\nnamespace static_mk { constexpr int lint_test = 1; }\n")
        assert static_schedule.code_tag(root) != tag, "a code change kept the tag"
        # a string literal's spaces are code
        assert static_schedule._code_tokens('char const *s = "a b";') != static_schedule._code_tokens('char const *s = "ab";')
        assert static_schedule._code_tokens('f("a " "b");') == static_schedule._code_tokens('f("a b");')
    print("code_tag:", tag)


def test_source():
    found = lints.lint_source(lints.tree_paths(TEMPLATES))
    for f in found:
        print("  ", f)
    rules = {(f.rule, f.where.split(":")[0]) for f in found}
    assert ("C11", "runtime.cuh") in rules          # drain_acc: no __forceinline__
    assert not any(f.rule == "B14" for f in found)  # no part constants in the generator
    bad = ('__device__ int f(int x) { return x; }\n'
           'template <int N> __device__ __forceinline__ void g() { __shared__ int a[N]; for (int i = 0; i < N; i++) { '
           '__threadfence(); } }\n'
           '__device__ __forceinline__ long long t() { long long v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); return v; }\n'
           '__device__ __forceinline__ void h() { extern __shared__ char s[]; __threadfence(); }\n')
    got = sorted((f.rule, f.level) for f in lints.lint_cuda_source("tasks/x.cuh", bad))
    assert got == [("C11", "fail"), ("C18", "flag"), ("C6", "fail"), ("H7", "flag")], got
    py = "x = 7168  # 3584 in a comment\ns = 'H = 896'\n"
    got = [(f.rule, f.where) for f in lints.lint_python_source("compiler.py", py)]
    assert got == [("B14", "compiler.py:1")], got
    print("source: ok,", len(found), "findings on the tree")


def test_contract_and_plan():
    pk = fakes.small_graph()
    assert lints.lint_contract(pk) == []
    pk._static_nodes[1]["cost_inputs"] = [3]
    assert [f.rule for f in lints.lint_contract(pk)] == ["B1"]
    pk = fakes.small_graph()
    pk._static_nodes[0]["variants"] = {"stages": (9, [3, 5])}
    assert [f.msg.split(":")[0] for f in lints.lint_contract(pk)] == ["variant stages (9, [3, 5])"], lints.lint_contract(pk)
    pk = fakes.small_graph()
    plan = explore.random_plan(pk, pk.num_gpus, pk.num_sms, 1)
    assert lints.lint_plan(pk, plan) == [], lints.lint_plan(pk, plan)
    broken = {"grids": dict(plan["grids"]), "lists": [[list(l) for l in g] for g in plan["lists"]]}
    sm = next(i for i, l in enumerate(broken["lists"][0]) if any(node != 2 for node, _ in l))
    broken["lists"][0][sm] = [e for e in broken["lists"][0][sm] if e[0] == 2]   # a task listed nowhere
    got = lints.lint_plan(pk, broken)
    assert [f.rule for f in got] == ["B2"] and "miss" in got[0].msg, got
    bad_grid = dict(plan, grids={0: (4, 3, 1)})
    assert [f.rule for f in lints.lint_plan(pk, bad_grid)] == ["B2"]

    class NoCosts:
        def task_us(self, n, grid, producers):
            return None if n.name == "tail" else 1.0

        def pool_us(self, n):
            return 100.0 if n.name == "pool" else None
    got = lints.lint_plan(fakes.small_graph(), plan, NoCosts())
    assert [(f.rule, f.where) for f in got] == [("D6", "node 3 (tail)")], got
    print("contract, plan: ok")


SASS = """
        code for sm_100a
                Function : _Z12layer_kernelN9static_mk4MapsENS_1GEPK10StaticTask
        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM100"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                     /* 0x0000df00ff017b82 */
        /*0010*/                   R2UR UR4, R2 ;                              /* 0x00000000020472ca */
        /*0020*/              @!P0 STL.64 [R1], R4 ;                           /* 0x0000000401008387 */
        /*0030*/                   UTCQMMA gdesc[UR4], gdesc[UR6], tmem[UR8] ; /* 0x0000000000007ea7 */
        /*0040*/                   DADD R4, R4, R6 ;                           /* 0x0000000604047229 */
        /*0050*/                   F2F.F64.F32 R4, R2 ;                        /* 0x0000000200047310 */
        /*0060*/                   CALL.REL.NOINC 0x100 ;                      /* 0x0000000000007944 */
        /*0070*/                   NOP ;                                       /* 0x0000000000007918 */
        /*0080*/                   EXIT ;                                      /* 0x000000000000794d */
"""
RES = """
Fatbin elf code:
================
arch = sm_100a
code version = [1,8]

Resource usage:
 Common:
  GLOBAL:0
 Function _Z12layer_kernelN9static_mk4MapsENS_1GEPK10StaticTask:
  REG:168 STACK:0 SHARED:2312 LOCAL:8 CONSTANT[0]:1424 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def test_sass_and_build():
    s = lints.sass_summary(SASS)
    assert (s["instructions"], s["ldl"], s["stl"], s["fp64"], s["call"], s["r2ur"], s["mma_sites"]) == (8, 0, 1, 2, 1, 1, 1), s
    found, _ = lints.lint_sass(None, SASS, parent_summary={"instructions": 7, "mma_sites": 1, "r2ur": 1})
    assert sorted(f.rule for f in found) == ["C11", "C14", "C15", "C23"], found
    assert lints.sass_identical(SASS, SASS.replace("R4", "R9"))
    with tempfile.TemporaryDirectory() as d:
        cu = os.path.join(d, "layer.cu")
        open(cu, "w").write('__global__ void __launch_bounds__(256, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, '
                            'static_mk::G g, StaticTask const *static_tasks) {}\n'
                            'static_assert(STATIC_SMEM_BYTES <= 227 * 1024, "the dynamic shared memory of one SM");\n')
        found, summary = lints.lint_build(cu, None, ["-use_fast_math", "-gencode", "arch=compute_100,code=sm_100"], RES,
                                          parent={"registers": 160})
        got = sorted((f.rule, f.level) for f in found)
        assert got == [("C11", "flag"), ("C13", "fail"), ("C15", "fail"), ("C20", "flag"), ("C24", "fail"), ("C6", "flag")], got
        assert summary["registers"] == 168 and summary["arch"] == "sm_100a"
    print("sass, build: ok")


def test_records():
    import json
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        rec = {"task_type": "gemm_tile", "key": {}, "context": {"first_on_sm": True, "prev": None}, "median_us": 1.0, "n": 3,
               "run": "r", "machine": {}, "code": "abc"}
        json.dump({"records": [rec, dict(rec, code="old"), dict(rec, context={"prev": None}), {"task_type": "x"}]}, open(path, "w"))
        got = sorted(f.rule for f in lints.lint_records(path, "abc"))
        assert got == ["D2", "D8", "D9"], got
    print("records: ok")


if __name__ == "__main__":
    test_code_tag()
    test_source()
    test_contract_and_plan()
    test_sass_and_build()
    test_records()
    print("all lint tests passed")
