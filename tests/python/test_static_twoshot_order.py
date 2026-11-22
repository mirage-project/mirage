"""static_schedule.check_twoshot_deadlock_free: the GPUs' lists together under the cross-GPU waits of STATIC_AR_TWOSHOT
(core.cuh; GLM-5.3-Flash at T 64 on 4 GPUs: allreduce_send -> hc_post_mix / hc_post).

Run: python tests/python/test_static_twoshot_order.py (no GPU, no mirage build: the module is loaded from its file).
Graph of the cases: node 1 allreduce_send (tasks (0..1, 0, 0)), node 2 hc_post_mix (tasks (0..7, 0, 0)), every
hc_post_mix task depends on both send tasks; 2 SMs a GPU. compile_plan's kinds for such a graph: {1: "send", 2: "twoshot"}
(compiler.twoshot_kinds)."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "static_schedule", os.path.join(HERE, "..", "..", "python", "mirage", "mpk", "static_schedule.py"))
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

KINDS = {1: "send", 2: "twoshot"}
GPUS = 4


def tasks(mix_deps=None):
    """send tasks 0, 1; hc_post_mix task x = id 2 + x (deps: both sends, or mix_deps[x])."""
    t = [{"node": 1, "pos": [i, 0, 0], "deps": []} for i in range(2)]
    t += [{"node": 2, "pos": [x, 0, 0], "deps": (mix_deps or {}).get(x, [0, 1])} for x in range(8)]
    return t


def mix(*xs):
    return [2 + x for x in xs]


def raises(gpus, kinds=KINDS) -> bool:
    try:
        ss.check_twoshot_deadlock_free(gpus, kinds)
        return False
    except ValueError:
        return True


def main():
    same = [[0] + mix(0, 1, 2, 3, 4, 5, 6, 7), [1]]
    # 1. the same lists on 4 GPUs: no deadlock (each GPU's lists alone pass check_deadlock_free too)
    g4 = [(tasks(), same, []) for _ in range(GPUS)]
    for all_tasks, queues, groups in g4:
        ss.check_deadlock_free(all_tasks, queues, groups)
    assert not raises(g4), "same lists flagged"
    # 2. tasks 5 and 7 swapped on one GPU: GPU 0 SM 0 = [..., mix 5, mix 7], GPU 1 SM 0 = [..., mix 7, mix 5] (GPUs 2, 3
    #    as GPU 0): each GPU alone passes, together they wait in a circle (GPU 0's 5 waits for GPU 1's 5, behind GPU 1's 7,
    #    which waits for GPU 0's 7, behind GPU 0's 5)
    a = [[0] + mix(0, 1, 2, 3, 4, 6, 5, 7), [1]]
    b = [[0] + mix(0, 1, 2, 3, 4, 6, 7, 5), [1]]
    for q in (a, b):
        ss.check_deadlock_free(tasks(), q)
    assert raises([(tasks(), a, []), (tasks(), b, [])] + [(tasks(), a, [])] * (GPUS - 2)), "5/7 swap not flagged"
    # 2b. negative control: the same swap with the hc_post_mix tasks not two-shot (one-shot: they wait only for the
    #     sends) passes
    assert not raises([(tasks(), a, []), (tasks(), b, [])] + [(tasks(), a, [])] * (GPUS - 2), {1: "send"}), \
        "one-shot flagged"
    # 3. the order swapped on different SMs only (task 5 and 7 each alone at an SM's head on one GPU): no circle
    c = [[0] + mix(0, 1, 2, 3, 4, 6, 5), [1] + mix(7)]
    d = [[0] + mix(0, 1, 2, 3, 4, 6, 7), [1] + mix(5)]
    assert not raises([(tasks(), c, []), (tasks(), d, [])] + [(tasks(), c, [])] * (GPUS - 2)), \
        "different SMs, no circle, flagged"
    # 4. the cross-GPU wait for the sends (one-shot too): GPU 0 SM 0 = [mix 0 (needs send 0 of every GPU), send 1],
    #    GPU 1 SM 0 = [mix 1 (needs send 1 of every GPU), send 0]; each GPU alone passes
    deps = {0: [0], 1: [1]}
    e = [[2, 1] + mix(2, 3, 4, 5, 6, 7), [0, 3]]
    f = [[3, 0] + mix(2, 3, 4, 5, 6, 7), [1, 2]]
    for q in (e, f):
        ss.check_deadlock_free(tasks(deps), q)
    assert raises([(tasks(deps), e, []), (tasks(deps), f, [])], {1: "send"}), "send circle not flagged"
    assert not raises([(tasks(deps), e, []), (tasks(deps), e, [])], {1: "send"}), "same lists (sends) flagged"
    # 5. when the generated layer.cu has the two-shot (ar_twoshot): T 64 on 4 GPUs, allreduce_send, no sum_rmsnorm
    assert ss.ar_twoshot(["allreduce_send", "hc_post_mix", "hc_pre"], 64, 4), "Flash tail at T 64 on 4 GPUs"
    assert not ss.ar_twoshot(["allreduce_send", "sum_rmsnorm", "sum_gpus"], 64, 4), "K3 MoE at T 64 on 4 GPUs"
    assert not ss.ar_twoshot(["allreduce_send", "hc_post_mix"], 64, 8), "8 GPUs"
    assert not ss.ar_twoshot(["allreduce_send", "hc_post_mix"], 32, 4), "T 32"
    assert not ss.ar_twoshot(["hc_post_mix", "hc_pre"], 64, 4), "no allreduce_send"
    print("test_static_twoshot_order: 7 checks OK")


if __name__ == "__main__":
    main()
