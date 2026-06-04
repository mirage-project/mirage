EVENT_LAUNCH_TASKS = 901
EVENT_LAUNCH_MASSIVE_TASKS = 902
EVENT_LAUNCH_DEPENDENT_TASKS = 903

TASK_PUSHING_EVENT_TYPES = {
    EVENT_LAUNCH_TASKS,
    EVENT_LAUNCH_MASSIVE_TASKS,
    EVENT_LAUNCH_DEPENDENT_TASKS,
}


def build_v2_worker_task_queues(task_graph: dict, num_workers: int) -> list[list[int]]:
    """Per-SM task queues, in the exact order the v2 kernel will run them.

    MUST stay in lockstep with the C++ twin `build_v2_plan` in
    persistent_kernel_v2.cuh: the kernel executes the C++ plan, while THESE
    queues drive the SMEM page planner (add_v2_region_smem_plan). If the two
    orderings ever diverge, the page plan no longer describes the actual
    execution order (harmless today with cross-task page overlap disabled,
    page-plan-corrupting once Phase E turns it on). Same algorithm on both
    sides: walk all_events in order, keep the task-pushing event types,
    round-robin each event's [first, last) range with a CONTINUOUS worker
    cursor across events, then prepend task 1 to worker 0.
    """
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

    # Task 1 is always TASK_BEGIN_TASK_GRAPH (task 0 is TASK_TERMINATE). It is
    # not covered by any task-pushing event, so prepend it explicitly; worker 0
    # runs it first each iteration (the v2 controller arrives SEM_DEP_READY and
    # the page semaphores on its behalf since it has no role bodies). Guard
    # against a graph shape that DOES push task 1 — scheduling it twice would
    # desync the per-page parity protocol.
    if task_graph.get("all_tasks"):
        if any(1 in queue for queue in queues):
            raise ValueError(
                "task 1 (begin_task_graph) unexpectedly covered by a "
                "task-pushing event; it must be scheduled exactly once")
        queues[0].insert(0, 1)

    return queues
