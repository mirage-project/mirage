"""Render how the v2 SMEM planner assigns physical pages to tasks on one SM.

Produces a figure (PNG via matplotlib, ASCII fallback) where

    rows    = tasks of one worker (SM) queue, in execution order
    columns = physical SMEM pages (0 .. num_pages-1)
    color   = release_step of the region occupying that page
    label   = region short-name (W0, A3, scr, ...); gray = page unused

so you can read, per task, exactly which pages it occupies (e.g. task 1 uses
pages 0-3, task 2 uses 4-7, ...). Driven by the demo's --profiling flag with
--use-v2; read-only and GPU-free (consumes the planned task-graph JSON that
`mirage.mpk.v2_smem_planner.add_v2_region_smem_plan` produced).
"""
from __future__ import annotations

import json


def _short_name(name: str) -> str:
    """linear_W_3 -> W3 ; linear_A_2 -> A2 ; linear_scratch -> scr ; else trim."""
    if name.startswith("linear_W_"):
        return "W" + name[len("linear_W_"):]
    if name.startswith("linear_A_"):
        return "A" + name[len("linear_A_"):]
    if "scratch" in name:
        return "scr"
    return name[:4]


def _collect_rows(graph: dict, worker: int, max_tasks: int | None):
    """Each row = one page-touching task of `worker`'s queue, in order.

    row["cells"][page] = (release_step, short_label) or None for unused pages.
    """
    planner = graph.get("v2_smem_planner", {})
    num_pages = planner.get("num_pages", 14)
    all_tasks = graph.get("all_tasks", [])
    queues = graph.get("v2_worker_task_queues")
    if queues is None:
        raise ValueError("graph has no v2_worker_task_queues — not a planned "
                         "v2 task graph (run with --use-v2).")
    if not 0 <= worker < len(queues):
        raise ValueError(f"worker {worker} out of range (0..{len(queues)-1})")

    rows = []
    for task_pos in queues[worker]:
        task = all_tasks[task_pos]
        regions = task.get("planned_smem_page_regions", [])
        if not regions:
            continue  # task touches no SMEM pages
        cells = [None] * num_pages
        for r in regions:
            for p in r["physical_pages"]:
                # Packed sub-page regions can share a page; keep the larger
                # release_step (matches the planner's per-page max).
                rs = r["release_step"]
                if cells[p] is None or rs > cells[p][0]:
                    cells[p] = (rs, _short_name(r["name"]))
        rows.append({
            "task_pos": task_pos,
            "type": task.get("task_type", "?"),
            "cells": cells,
        })
        if max_tasks and len(rows) >= max_tasks:
            break
    return rows, num_pages


def _print_ascii(rows, num_pages):
    print(f"\nSMEM page usage  ({len(rows)} tasks x {num_pages} pages); "
          f"cell = release_step:region, '.' = unused\n")
    print("task   " + "".join(f"{p:>5}" for p in range(num_pages)))
    for row in rows:
        line = f"{row['task_pos']:>5}  "
        for c in row["cells"]:
            line += "    ." if c is None else f"{c[0]:>2}:{c[1]:<2}"[:5].rjust(5)
        print(line)
    print()


def save_page_plan_figure(task_graph_json: str | dict,
                          out_path: str,
                          worker: int = 0,
                          max_tasks: int | None = 64) -> str | None:
    """Write the per-SM page-usage figure for `worker`'s task queue.

    Returns the written path, or None if there was nothing to plot. Falls back
    to an ASCII table on stdout when matplotlib/numpy are unavailable.
    """
    graph = (json.loads(task_graph_json)
             if isinstance(task_graph_json, str) else task_graph_json)
    rows, num_pages = _collect_rows(graph, worker, max_tasks)
    if not rows:
        print(f"[page_plan_viz] worker {worker} has no page-touching tasks")
        return None

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[page_plan_viz] matplotlib/numpy unavailable ({e}); "
              f"ASCII fallback:")
        _print_ascii(rows, num_pages)
        return None

    n = len(rows)
    grid = np.full((n, num_pages), np.nan)
    for i, row in enumerate(rows):
        for p, c in enumerate(row["cells"]):
            if c is not None:
                grid[i, p] = c[0]

    fig_h = max(3.0, 0.34 * n + 1.5)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * num_pages, fig_h))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="0.9")  # unused pages -> light gray
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    for i, row in enumerate(rows):
        for p, c in enumerate(row["cells"]):
            if c is not None:
                ax.text(p, i, c[1], ha="center", va="center", fontsize=7,
                        color="white")
    ax.set_xticks(range(num_pages))
    ax.set_xticklabels(range(num_pages), fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"t{r['task_pos']} ({r['type']})" for r in rows],
                       fontsize=7)
    ax.set_xlabel("physical SMEM page")
    ax.set_ylabel(f"task (SM {worker} queue order)")
    ax.set_title("v2 SMEM planner: page usage per task\n"
                 "(color = region release_step; gray = unused)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("release_step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[page_plan_viz] wrote {out_path}  ({n} tasks x {num_pages} pages)")
    return out_path
