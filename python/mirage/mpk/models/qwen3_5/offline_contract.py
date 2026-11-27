"""Pure-Python task metadata guard for Qwen offline request isolation."""

import json


def validate_offline_task_graph(graph_json, batch_size, *, attention_layers=16,
                                kv_heads=4, query_heads=24):
    """Reject native runtimes that assign attention heads to request slots."""
    tasks = json.loads(graph_json)["all_tasks"]
    layouts = ((560, "outputs", 8, batch_size),
               (554, "inputs", kv_heads * 256 * 2, attention_layers * batch_size),
               (555, "inputs", query_heads * 256 * 2,
                attention_layers * query_heads * batch_size))
    for task_type, side, row_bytes, expected_count in layouts:
        selected = [task for task in tasks if task["task_type"] == task_type]
        if len(selected) != expected_count:
            raise RuntimeError(f"Qwen offline task {task_type}: expected {expected_count}, got {len(selected)}")
        for task in selected:
            offset = task[side][0]["offset"]
            slot = offset // row_bytes
            if not 0 <= slot < batch_size or task["request_id"] != slot:
                raise RuntimeError(
                    f"Qwen offline task {task_type}: tensor row {slot} has request_id "
                    f"{task['request_id']}; rebuild libmirage_runtime and the Python extension")
