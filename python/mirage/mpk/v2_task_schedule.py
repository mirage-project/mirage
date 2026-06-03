EVENT_LAUNCH_TASKS = 901
EVENT_LAUNCH_MASSIVE_TASKS = 902
EVENT_LAUNCH_DEPENDENT_TASKS = 903

TASK_PUSHING_EVENT_TYPES = {
    EVENT_LAUNCH_TASKS,
    EVENT_LAUNCH_MASSIVE_TASKS,
    EVENT_LAUNCH_DEPENDENT_TASKS,
}


def build_v2_worker_task_queues(task_graph: dict, num_workers: int) -> list[list[int]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    queues = [[] for _ in range(num_workers)]
    next_worker = 0

    for event in task_graph.get("all_events", []):
        if int(event.get("event_type", -1)) not in TASK_PUSHING_EVENT_TYPES:
            continue

        first = int(event.get("first_task_id", 0))
        last = int(event.get("last_task_id", 0))
        if first >= last:
            continue

        for task_pos in range(first, last):
            queues[next_worker % num_workers].append(task_pos)
            next_worker += 1

    if task_graph.get("all_tasks"):
        queues[0].insert(0, 1)

    return queues
