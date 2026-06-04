import json


PAGE_SIZE = 16 * 1024
NUM_PAGES = 14
CAPACITY_BYTES = 225 * 1024 - 6 * 1024


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _default_plan() -> dict:
    return {
        "planned_smem_valid": True,
        "planned_smem_failure": "",
        "planned_smem_num_pages": NUM_PAGES,
        "planned_smem_release_order": list(range(NUM_PAGES)),
        "planned_smem_page_regions": [],
        "planned_smem_used_pages": [],
        "planned_smem_used_page_count": 0,
        "planned_smem_peak_pages": 0,
    }


def _region_from_json(raw: dict, index: int) -> tuple[dict | None, str]:
    size = int(raw.get("size", 0))
    alignment = int(raw.get("alignment", 1))
    page_count = int(raw.get("page_count", -1))
    release_step = int(raw.get("release_step", 0))
    contiguous = bool(raw.get("contiguous", True))

    if size < 0:
        return None, "invalid_size"
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        return None, "invalid_alignment"
    if page_count >= 0 and page_count * PAGE_SIZE < size:
        return None, "page_count_too_small"
    if not contiguous:
        return None, "non_contiguous_region_unsupported"

    return {
        "region_index": index,
        "name": raw.get("name", f"region_{index}"),
        "size": size,
        "alignment": alignment,
        "page_count": None if page_count < 0 else page_count,
        "can_pack": bool(raw.get("can_pack", False)),
        "release_step": release_step,
        "contiguous": contiguous,
    }, ""


class _PageAllocator:
    def __init__(self):
        self._page_allocs = [[] for _ in range(NUM_PAGES)]

    def allocate(self, region: dict) -> tuple[list[int], int] | None:
        size = region["size"]
        if size == 0:
            return [], 0

        page_count = region["page_count"]
        if page_count is None:
            page_count = _ceil_div(size, PAGE_SIZE)

        if region["can_pack"] and page_count <= 1 and size < PAGE_SIZE:
            return self._allocate_packed(size, region["alignment"], region)
        return self._allocate_pages(page_count, region)

    def _page_capacity(self, page: int) -> int:
        page_base = page * PAGE_SIZE
        if page_base >= CAPACITY_BYTES:
            return 0
        return min(PAGE_SIZE, CAPACITY_BYTES - page_base)

    def _allocate_packed(
        self, size: int, alignment: int, region: dict
    ) -> tuple[list[int], int] | None:
        for page in range(NUM_PAGES):
            page_capacity = self._page_capacity(page)
            if page_capacity <= 0:
                continue
            offset = 0
            for alloc in sorted(self._page_allocs[page], key=lambda item: item["offset"]):
                candidate = _align_up(offset, alignment)
                if candidate + size <= alloc["offset"]:
                    return self._record(page, candidate, size, region)
                offset = max(offset, alloc["offset"] + alloc["size"])

            candidate = _align_up(offset, alignment)
            if candidate + size <= page_capacity:
                return self._record(page, candidate, size, region)
        return None

    def _allocate_pages(self, page_count: int, region: dict) -> tuple[list[int], int] | None:
        for first_page in range(0, NUM_PAGES - page_count + 1):
            if first_page * PAGE_SIZE + region["size"] > CAPACITY_BYTES:
                continue
            pages = list(range(first_page, first_page + page_count))
            if any(self._page_allocs[page] for page in pages):
                continue
            for page in pages:
                self._page_allocs[page].append({
                    "offset": 0,
                    "size": PAGE_SIZE,
                    "region": region,
                })
            return pages, 0
        return None

    def _record(
        self, page: int, offset: int, size: int, region: dict
    ) -> tuple[list[int], int]:
        self._page_allocs[page].append({
            "offset": offset,
            "size": size,
            "region": region,
        })
        return [page], offset


def _release_order(page_regions: list[dict]) -> list[int]:
    release_step_by_page = [-1 for _ in range(NUM_PAGES)]
    for region in page_regions:
        for page in region["physical_pages"]:
            release_step_by_page[page] = max(
                release_step_by_page[page], region["release_step"])
    return sorted(range(NUM_PAGES), key=lambda page: (release_step_by_page[page], page))


def _validate_page_permutation(values: list[int], field_name: str) -> None:
    if sorted(values) != list(range(NUM_PAGES)):
        raise ValueError(f"{field_name} must be a permutation of 0..{NUM_PAGES - 1}")


def _is_consecutive(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _validate_contiguous_regions(plan: dict) -> None:
    for region in plan["planned_smem_page_regions"]:
        physical_pages = region["physical_pages"]
        if region["contiguous"] and not _is_consecutive(physical_pages):
            raise ValueError(
                f"SMEM region {region['name']} requires contiguous physical pages, "
                f"got physical pages {physical_pages}")


def _find_physical_run(page_count: int,
                       used_physical_pages: set[int],
                       preferred_order: list[int]) -> list[int] | None:
    # NOTE: preferred_order (the prior task's release order) is currently NOT
    # consulted — multi-page contiguous regions always take the lowest free
    # run. Harmless while cross-task page overlap is disabled on the device
    # (CROSS_TASK_PAGES=false: page lifetimes are task-scoped, so assignment
    # order has no effect). When Phase E turns overlap on, this must prefer
    # runs of earliest-released pages or the W regions (the main overlap
    # beneficiaries) won't follow the fill-fresh-then-earliest-released rule.
    for first_page in range(0, NUM_PAGES - page_count + 1):
        physical_pages = list(range(first_page, first_page + page_count))
        if any(page in used_physical_pages for page in physical_pages):
            continue
        return physical_pages
    return None


def _assign_physical_pages(page_regions: list[dict],
                           preferred_order: list[int]) -> list[int]:
    _validate_page_permutation(preferred_order, "preferred_physical_order")
    pid_order = [-1 for _ in range(NUM_PAGES)]
    used_physical_pages: set[int] = set()

    def assign_logical_pages(logical_pages: list[int],
                             physical_pages: list[int]) -> None:
        for logical_page, physical_page in zip(logical_pages, physical_pages):
            if pid_order[logical_page] not in (-1, physical_page):
                raise ValueError(
                    f"logical page {logical_page} assigned to multiple physical pages")
            pid_order[logical_page] = physical_page
            used_physical_pages.add(physical_page)

    multi_page_regions = [
        region for region in page_regions
        if region["contiguous"] and len(region["logical_pages"]) > 1
    ]
    for region in sorted(multi_page_regions,
                         key=lambda item: len(item["logical_pages"]),
                         reverse=True):
        logical_pages = region["logical_pages"]
        physical_pages = _find_physical_run(
            len(logical_pages), used_physical_pages, preferred_order)
        if physical_pages is None:
            raise ValueError(
                f"SMEM capacity exceeded while assigning contiguous physical pages "
                f"for region {region['name']}")
        assign_logical_pages(logical_pages, physical_pages)

    for region in page_regions:
        if len(region["logical_pages"]) != 1:
            continue
        logical_page = region["logical_pages"][0]
        if pid_order[logical_page] != -1:
            continue
        physical_page = next(
            (page for page in preferred_order if page not in used_physical_pages),
            None)
        if physical_page is None:
            raise ValueError(
                f"SMEM capacity exceeded while assigning physical page "
                f"for region {region['name']}")
        assign_logical_pages([logical_page], [physical_page])

    for logical_page in range(NUM_PAGES):
        if pid_order[logical_page] != -1:
            continue
        physical_page = next(
            page for page in preferred_order if page not in used_physical_pages)
        assign_logical_pages([logical_page], [physical_page])

    _validate_page_permutation(pid_order, "planned_smem_pid_order")
    return pid_order


def _plan_task(task: dict, preferred_physical_order: list[int]) -> dict:
    plan = _default_plan()
    raw_regions = task.get("smem_regions", [])
    if not raw_regions:
        # A task with no SMEM regions touches no pages, so it must not disturb
        # the cross-task page-id stability the worker queue is trying to build:
        # forward the incoming preferred order to the next task untouched.
        _validate_page_permutation(
            list(preferred_physical_order), "preferred_physical_order")
        plan["planned_smem_release_order"] = list(preferred_physical_order)
        return plan

    allocator = _PageAllocator()
    page_regions = []

    parsed_regions = []
    for index, raw_region in enumerate(raw_regions):
        region, failure = _region_from_json(raw_region, index)
        if region is None:
            raise ValueError(f"invalid SMEM region {index}: {failure}")
        parsed_regions.append(region)

    def allocation_priority(region: dict) -> tuple[int, int, int]:
        page_count = region["page_count"]
        if page_count is None:
            page_count = _ceil_div(region["size"], PAGE_SIZE)
        packable = region["can_pack"] and page_count <= 1 and region["size"] < PAGE_SIZE
        return (1 if packable else 0, -page_count, -region["size"])

    allocated_regions = []
    for region in sorted(parsed_regions, key=allocation_priority):
        allocation = allocator.allocate(region)
        if allocation is None:
            region_name = region["name"]
            raise ValueError(f"SMEM capacity exceeded while allocating region {region_name}")

        logical_pages, byte_offset = allocation
        allocated_regions.append({
            "region_index": region["region_index"],
            "name": region["name"],
            "size": region["size"],
            "page_count": len(logical_pages),
            "logical_pages": logical_pages,
            "byte_offset_in_first_page": byte_offset,
            "release_step": region["release_step"],
            "contiguous": region["contiguous"],
        })
    page_regions = sorted(allocated_regions, key=lambda region: region["region_index"])

    pid_order = _assign_physical_pages(page_regions, preferred_physical_order)
    for region in page_regions:
        region["physical_pages"] = [pid_order[page] for page in region["logical_pages"]]
        del region["logical_pages"]

    used_pages = sorted({
        page
        for region in page_regions
        for page in region["physical_pages"]
    })

    plan["planned_smem_page_regions"] = page_regions
    plan["planned_smem_used_pages"] = used_pages
    plan["planned_smem_used_page_count"] = len(used_pages)
    plan["planned_smem_peak_pages"] = len(used_pages)
    plan["planned_smem_release_order"] = _release_order(page_regions)
    _validate_page_permutation(plan["planned_smem_release_order"],
                               "planned_smem_release_order")
    _validate_contiguous_regions(plan)
    return plan


def _validate_worker_task_queues(queues: list, num_tasks: int) -> None:
    if not isinstance(queues, list):
        raise ValueError("v2_worker_task_queues must be a list")
    if not queues:
        raise ValueError("v2_worker_task_queues must contain at least one worker queue")

    for worker_id, queue in enumerate(queues):
        if not isinstance(queue, list):
            raise ValueError(f"worker queue {worker_id} must be a list")
        for task_pos in queue:
            if not isinstance(task_pos, int):
                raise ValueError(f"worker queue {worker_id} contains non-integer task id")
            if task_pos < 0 or task_pos >= num_tasks:
                raise ValueError(
                    f"worker queue {worker_id} contains out-of-range task id {task_pos}")


def add_v2_region_smem_plan(task_graph_json: str) -> str:
    graph = json.loads(task_graph_json)
    tasks = graph.get("all_tasks", [])
    queues = graph.get("v2_worker_task_queues")
    if queues is None:
        raise ValueError("v2_worker_task_queues is required before SMEM planning")
    _validate_worker_task_queues(queues, len(tasks))
    num_workers = len(queues)

    graph["v2_smem_planner"] = {
        "version": 1,
        "page_size": PAGE_SIZE,
        "num_pages": NUM_PAGES,
        "capacity_bytes": PAGE_SIZE * NUM_PAGES,
        "num_workers": num_workers,
    }

    for task in tasks:
        task.update(_default_plan())

    invalid_tasks = 0
    page_planned_tasks = 0
    zero_page_tasks = 0
    per_worker_peak_pages = [0 for _ in range(num_workers)]

    for worker_id, queue in enumerate(queues):
        preferred_physical_order = list(range(NUM_PAGES))

        for task_pos in queue:
            plan = _plan_task(tasks[task_pos], preferred_physical_order)
            tasks[task_pos].update(plan)

            if plan["planned_smem_valid"]:
                if plan["planned_smem_page_regions"]:
                    page_planned_tasks += 1
                else:
                    zero_page_tasks += 1

            per_worker_peak_pages[worker_id] = max(
                per_worker_peak_pages[worker_id],
                plan["planned_smem_used_page_count"])
            preferred_physical_order = plan["planned_smem_release_order"]

    global_peak_pages = max(per_worker_peak_pages) if per_worker_peak_pages else 0
    graph["v2_smem_planner"].update({
        "per_worker_peak_pages": per_worker_peak_pages,
        "global_peak_pages": global_peak_pages,
        "invalid_plan_tasks": invalid_tasks,
        "page_planned_tasks": page_planned_tasks,
        "zero_page_tasks": zero_page_tasks,
    })
    return json.dumps(graph, indent=2)
