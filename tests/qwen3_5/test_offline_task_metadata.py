"""The offline Qwen graph must keep its request axis through native lowering."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/mirage/mpk/models"))
from qwen3_5.offline_contract import validate_offline_task_graph


def task(kind, slot, offset):
    return {"task_type": kind, "request_id": slot,
            "inputs": [{"offset": offset}], "outputs": [{"offset": offset}]}


def graph(batch=8):
    tasks = [task(560, row, row * 8) for row in range(batch)]
    for _ in range(16):
        tasks.extend(task(554, row, row * 4 * 256 * 2) for row in range(batch))
        tasks.extend(task(555, row, row * 24 * 256 * 2 + head * 256 * 2)
                     for head in range(24) for row in range(batch))
    return tasks


def test_qwen38_offline_request_axis():
    tasks = graph()
    validate_offline_task_graph(json.dumps({"all_tasks": tasks}), 8)
    tasks[1]["request_id"] = 0
    with pytest.raises(RuntimeError, match="tensor row 1 has request_id 0"):
        validate_offline_task_graph(json.dumps({"all_tasks": tasks}), 8)


def test_qwen38_offline_attention_head_cannot_be_request_id():
    tasks = graph()
    attention = next(task for task in tasks if task["task_type"] == 555 and
                     task["inputs"][0]["offset"] == 512)
    attention["request_id"] = 1
    with pytest.raises(RuntimeError, match="tensor row 0 has request_id 1"):
        validate_offline_task_graph(json.dumps({"all_tasks": tasks}), 8)


def test_qwen36_offline_uses_its_own_head_and_layer_geometry():
    batch = 8
    tasks = [task(560, row, row * 8) for row in range(batch)]
    for _ in range(10):
        tasks.extend(task(554, row, row * 2 * 256 * 2) for row in range(batch))
        tasks.extend(task(555, row, row * 16 * 256 * 2 + head * 256 * 2)
                     for head in range(16) for row in range(batch))
    graph_json = json.dumps({"all_tasks": tasks})
    validate_offline_task_graph(graph_json, batch, attention_layers=10, kv_heads=2, query_heads=16)
    with pytest.raises(RuntimeError, match="expected"):
        validate_offline_task_graph(graph_json, batch)
