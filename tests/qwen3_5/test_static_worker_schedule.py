"""Static B8 worker plan validation without importing the GPU runtime."""

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "python/mirage/mpk/models/qwen3_5/static_worker_schedule.py"
SPEC = importlib.util.spec_from_file_location("static_worker_schedule", SOURCE)
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)


def graph():
    invalid = (1 << 63) - 2
    tasks = [
        dict(task_type=0, dependent_event=invalid, trigger_event=invalid),
        dict(task_type=10, dependent_event=invalid, trigger_event=1),
    ]
    tasks += [dict(task_type=556, dependent_event=invalid,
                   trigger_event=2 if i == 0 else invalid) for i in range(8)]
    tasks += [dict(task_type=253, dependent_event=2, trigger_event=invalid)]
    return dict(all_tasks=tasks, first_tasks=list(range(2, 10)), all_events=[
        dict(event_type=911, num_triggers=1),
        dict(event_type=903, first_task_id=2, last_task_id=len(tasks), num_triggers=1),
        dict(event_type=900, num_triggers=1),
    ])


def test_round_robin_covers_every_task_once(tmp_path):
    data = graph()
    lists = schedule.build_schedule(data, 4, strategy="round_robin")
    assert lists == [[2, 6, 10], [3, 7], [4, 8], [5, 9]]
    path = tmp_path / "b8.plan"
    schedule.write_schedule(path, data, lists)
    header = path.read_text().splitlines()[0].split()
    assert header[:3] == ["MPK_STATIC_WORKERS_V1", "4", "11"]
    assert int(header[3]) == schedule.graph_signature(data["all_tasks"])


def test_runtime_layout_reproduces_begin_task_cursor():
    lists = schedule.build_schedule(graph(), 4, strategy="runtime_layout",
                                    num_schedulers=2)
    assert lists == [[3, 7], [2, 6, 10], [4, 8], [5, 9]]


def test_balanced_requires_measured_costs():
    with pytest.raises(ValueError, match="missing measured task costs"):
        schedule.build_schedule(graph(), 4, strategy="balanced")
    lists = schedule.build_schedule(graph(), 4, {556: 2.0, 253: 9.0},
                                    strategy="balanced")
    assert sorted(task for ids in lists for task in ids) == list(range(2, 11))


def test_rejects_forward_dependency():
    data = graph()
    data["all_tasks"][2]["dependent_event"] = 2
    with pytest.raises(ValueError, match="non-topological dependency"):
        schedule.build_schedule(data, 4, strategy="round_robin")
