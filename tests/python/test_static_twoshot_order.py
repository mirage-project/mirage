"""static_schedule.check_twoshot_deadlock_free: the GPUs' lists together under the cross-GPU waits of STATIC_GLM64_TWOSHOT.

Run: python tests/python/test_static_twoshot_order.py (no GPU, no mirage build: the module is loaded from its file).
Graph of the cases: node 1 allreduce_send (tasks (0..1, 0, 0)), node 2 residual_add (tasks (0..7, 0, 0)), every
residual_add task depends on both send tasks; 2 SMs a GPU."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "static_schedule", os.path.join(HERE, "..", "..", "python", "mirage", "mpk", "static_schedule.py"))
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

KINDS = {1: "send", 2: "twoshot"}


def tasks(ra_deps=None):
    """send tasks 0, 1; residual_add task x = id 2 + x (deps: both sends, or ra_deps[x])."""
    t = [{"node": 1, "pos": [i, 0, 0], "deps": []} for i in range(2)]
    t += [{"node": 2, "pos": [x, 0, 0], "deps": (ra_deps or {}).get(x, [0, 1])} for x in range(8)]
    return t


def ra(*xs):
    return [2 + x for x in xs]


def raises(gpus, kinds=KINDS) -> bool:
    try:
        ss.check_twoshot_deadlock_free(gpus, kinds)
        return False
    except ValueError:
        return True


def main():
    same = [[0] + ra(0, 1, 2, 3, 4, 5, 6, 7), [1]]
    # 1. the same lists on 8 GPUs: no deadlock (each GPU's lists alone pass check_deadlock_free too)
    g8 = [(tasks(), same, []) for _ in range(8)]
    for all_tasks, queues, groups in g8:
        ss.check_deadlock_free(all_tasks, queues, groups)
    assert not raises(g8), "same lists flagged"
    # 2. the review's case: GPU 0 SM 0 = [..., ra 5, ra 7], GPU 1 SM 0 = [..., ra 7, ra 5] (the other GPUs as GPU 0):
    #    each GPU alone passes, together they wait in a circle
    a = [[0] + ra(0, 1, 2, 3, 4, 6, 5, 7), [1]]
    b = [[0] + ra(0, 1, 2, 3, 4, 6, 7, 5), [1]]
    for q in (a, b):
        ss.check_deadlock_free(tasks(), q)
    assert raises([(tasks(), a, [])] + [(tasks(), b, [])] + [(tasks(), a, [])] * 6), "5/7 swap not flagged"
    # 2b. negative control: the same swap with the residual_add tasks not two-shot (one-shot: they wait only for the
    #     sends) passes
    assert not raises([(tasks(), a, [])] + [(tasks(), b, [])] + [(tasks(), a, [])] * 6, {1: "send"}), "one-shot flagged"
    # 3. the order swapped on different SMs only (task 5 and 7 each alone at an SM's head on one GPU): no circle
    c = [[0] + ra(0, 1, 2, 3, 4, 6, 5), [1] + ra(7)]
    d = [[0] + ra(0, 1, 2, 3, 4, 6, 7), [1] + ra(5)]
    assert not raises([(tasks(), c, []), (tasks(), d, [])]), "different SMs, no circle, flagged"
    # 4. the cross-GPU wait for the sends (one-shot too): GPU 0 SM 0 = [ra 0 (needs send 0 of every GPU), send 1],
    #    GPU 1 SM 0 = [ra 1 (needs send 1 of every GPU), send 0]; each GPU alone passes
    deps = {0: [0], 1: [1]}
    e = [[2, 1] + ra(2, 3, 4, 5, 6, 7), [0, 3]]
    f = [[3, 0] + ra(2, 3, 4, 5, 6, 7), [1, 2]]
    for q in (e, f):
        ss.check_deadlock_free(tasks(deps), q)
    assert raises([(tasks(deps), e, []), (tasks(deps), f, [])], {1: "send"}), "send circle not flagged"
    assert not raises([(tasks(deps), e, []), (tasks(deps), e, [])], {1: "send"}), "same lists (sends) flagged"
    print("test_static_twoshot_order: 7 checks OK")


if __name__ == "__main__":
    main()
