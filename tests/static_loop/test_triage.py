"""Triage on stamps computed by hand (python tests/static_loop/test_triage.py)."""
import fakes

fakes.import_mpk()
from mirage.mpk.loop import triage  # noqa: E402

# 1 GPU, 2 SMs: a0, a1 (node 0) -> b (node 1, waits for both) -> c (node 2, waits for b). SM 0: [a0, b], SM 1: [a1, c]
SCHED = {"nodes": {"0": {"name": "a", "grid": [2, 1, 1]}, "1": {"name": "b", "grid": [1, 1, 1]}, "2": {"name": "c", "grid": [1, 1, 1]}},
         "all_tasks": [{"node": 0, "pos": [0, 0, 0], "deps": []}, {"node": 0, "pos": [1, 0, 0], "deps": []},
                       {"node": 1, "pos": [0, 0, 0], "deps": [0, 1]}, {"node": 2, "pos": [0, 0, 0], "deps": [2]}],
         "worker_task_queues": [[0, 2], [1, 3]]}
BAR = 1000
TIMES = [[[(1500, 3500, 0, 0), (3600, 5000, 0, 0), (0, 0, 0, 0)],      # SM 0: a0, b
          [(1200, 4000, 0, 0), (4100, 7000, 0, 0), (0, 0, 0, 0)]]]     # SM 1: a1, c
TIMING = [(BAR, 7000)]


def close(a, b):
    assert abs(a - b) < 1e-9, (a, b)


def test_chain():
    # c (end 7000) began when b ended (5000 > SM 1's previous end 4000); b began when a1 ended (4000 > a0's 3500); a1 is first on
    # SM 1 with no inputs: start = 1200 - 1000 = 0.2 us. Segments 2.8 + 1.0 + 2.0, span 6.0
    cp = triage.critical_path([SCHED], TIMES, TIMING, pool_node=-1)
    assert [e["name"] for e in cp["path"]] == ["a", "b", "c"] and [e["task"] for e in cp["path"]] == [1, 2, 3], cp["path"]
    close(cp["span_us"], 6.0)
    close(cp["start_us"], 0.2)
    assert [e["seg"] for e in cp["path"]] == [2.8, 1.0, 2.0]
    # b waited 4000 - max(3600, 3500) = 0.4 us inside its body; c 5000 - max(4100, 4000) = 0.9 us
    assert [e["waited_us"] for e in cp["path"]] == [0.0, 0.4, 0.9]
    tot = triage.attribute(cp)
    close(tot["sum_us"], 6.0)
    assert tot["classes"] == {"work:a": 2.8, "work:c": 2.0, "work:b": 1.0, "start": 0.2}, tot


def test_pool_imbalance():
    # node a has one task per SM, task x on SM x: found as the pool. a1's end 4000 - mean(3500, 4000) = 0.25 us imbalance
    assert triage.find_pool_node([SCHED]) == 0
    cp = triage.critical_path([SCHED], TIMES, TIMING)
    close(cp["path"][0]["classes"]["imbalance"], 0.25)
    close(cp["path"][0]["classes"]["work:a"], 2.55)
    close(triage.attribute(cp)["sum_us"], 6.0)


def test_cross_gpu():
    # 2 GPUs. Tasks: 0 a0, 1 a1 (node a), 2 b0 (waits for a0, a1), 3 c (waits for b0), 4 b1 (node b, no inputs).
    # SM 0: [b1, a0, b0], SM 1: [a1, c]. b0 is warm (b1 ran on SM 0 before it). Node a's output counts as exchange data.
    sched = {"nodes": SCHED["nodes"], "worker_task_queues": [[4, 0, 2], [1, 3]],
             "all_tasks": SCHED["all_tasks"] + [{"node": 1, "pos": [1, 0, 0], "deps": []}]}
    g0 = [[(1100, 1300, 0, 0), (1300, 3500, 0, 0), (3600, 7500, 0, 0)], [(1200, 4000, 0, 0), (7600, 8000, 0, 0), (0, 0, 0, 0)]]
    g1 = [[(1100, 1300, 0, 0), (1300, 3500, 0, 0), (6100, 7100, 0, 0)], [(1200, 6000, 0, 0), (7200, 7400, 0, 0), (0, 0, 0, 0)]]
    cp = triage.critical_path([sched, sched], [g0, g1], [(BAR, 8000), (BAR, 7400)], exchange_nodes=[0], pool_node=-1)
    # GPU 0 ends last (7.0 us). Path: a1 (first on SM 1, 0.2 us start), b0 (began at a1's end 4000), c (began at b0's end 7500).
    # b's warm segments: GPU 0 b0 3.5 us, GPU 1 b0 1.1 us (began at 6000) -> median 2.3; GPU 0's b0 has 1.2 us above it; node a
    # ended last on GPU 1 at 5.0 us after its barrier vs b0's begin at 3.0: up to 2.0 us cross_gpu -> 1.2
    assert cp["gpu"] == 0 and [e["task"] for e in cp["path"]] == [1, 2, 3], cp["path"]
    b = cp["path"][1]
    assert b["classes"] == {"work:b": 2.3, "cross_gpu": 1.2}, b
    tot = triage.attribute(cp)
    close(tot["sum_us"], 7.0)
    assert tot["classes"] == {"work:a": 2.8, "work:b": 2.3, "cross_gpu": 1.2, "work:c": 0.5, "start": 0.2}, tot
    # without exchange_nodes, the 1.2 us is handoff
    cp = triage.critical_path([sched, sched], [g0, g1], [(BAR, 8000), (BAR, 7400)], pool_node=-1)
    assert cp["path"][1]["classes"] == {"work:b": 2.3, "handoff": 1.2}


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
