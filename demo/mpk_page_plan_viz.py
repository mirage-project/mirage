#!/usr/bin/env python3
"""Visualize how the v2 SMEM planner assigns physical pages to each task.

Reads a *planned* task-graph JSON (the output of
`mirage.mpk.v2_smem_planner.add_v2_region_smem_plan`, which the demo writes to
`<output_dir>/task_graph.json`) and renders a heatmap:

    rows    = tasks of one worker queue, in execution order
    columns = physical SMEM pages (0 .. num_pages-1)
    color   = release_step of the region occupying that physical page
    label   = region short-name (W0, A3, scr, ...); blank = unused page

This makes visible (a) which physical pages each task is assigned, and (b) how
the assignment shifts task-to-task via the planner's
`release_order -> next task's preferred_physical_order` chaining. With
release_step uniform (== 1) the chaining is degenerate and the map is a single
flat color; with distinct per-stage release_step it shows structure.

Read-only, no GPU. matplotlib optional (ASCII fallback otherwise).

Usage:
    python demo/mpk_page_plan_viz.py \
        --plan ./permanent_output_dir/task_graph.json \
        --worker 0 --task-filter linear --max-tasks 24 --out /tmp/page_plan.png
"""
import argparse
import json
import os
import sys


def short_name(name: str) -> str:
    """linear_W_3 -> W3 ; linear_A_2 -> A2 ; linear_scratch -> scr ; else trim."""
    if name.startswith("linear_W_"):
        return "W" + name[len("linear_W_"):]
    if name.startswith("linear_A_"):
        return "A" + name[len("linear_A_"):]
    if "scratch" in name:
        return "scr"
    return name[:4]


def load_graph(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"plan file not found: {path}\n"
                 f"Generate it by running the demo with --output-dir "
                 f"./permanent_output_dir (writes <dir>/task_graph.json).")
    with open(path) as f:
        return json.load(f)


def collect_rows(graph: dict, worker: int, task_filter: str, max_tasks: int):
    """Return (rows, num_pages). Each row = dict(task_pos, type, cells, order).

    cells[page] = (release_step, short_label) or None for unused pages.
    """
    planner = graph.get("v2_smem_planner", {})
    num_pages = planner.get("num_pages", 14)
    all_tasks = graph.get("all_tasks", [])
    queues = graph.get("v2_worker_task_queues")
    if queues is None:
        sys.exit("graph has no v2_worker_task_queues — not a planned v2 graph.")
    if worker < 0 or worker >= len(queues):
        sys.exit(f"--worker {worker} out of range (0..{len(queues) - 1})")

    rows = []
    for task_pos in queues[worker]:
        task = all_tasks[task_pos]
        regions = task.get("planned_smem_page_regions", [])
        if not regions:
            continue
        names = " ".join(r["name"] for r in regions)
        if task_filter and task_filter not in names:
            continue
        cells = [None] * num_pages
        for r in regions:
            for p in r["physical_pages"]:
                # If two regions share a page (packing), keep the larger
                # release_step (matches planner's release_step_by_page = max).
                rs = r["release_step"]
                lbl = short_name(r["name"])
                if cells[p] is None or rs > cells[p][0]:
                    cells[p] = (rs, lbl)
        rows.append({
            "task_pos": task_pos,
            "type": task.get("task_type", "?"),
            "cells": cells,
            "order": task.get("planned_smem_release_order", []),
        })
        if max_tasks and len(rows) >= max_tasks:
            break
    if not rows:
        sys.exit(f"no tasks matched (worker={worker}, filter={task_filter!r}). "
                 f"Try --task-filter '' or a different --worker.")
    return rows, num_pages


def print_ascii(rows, num_pages):
    print(f"\nPage-assignment map  (worker queue order, {len(rows)} tasks, "
          f"{num_pages} pages)")
    print("cell = release_step:region   '.' = unused\n")
    header = "task   " + "".join(f"{p:>5}" for p in range(num_pages))
    print(header)
    for row in rows:
        line = f"{row['task_pos']:>5}  "
        for c in row["cells"]:
            line += "    ." if c is None else f"{c[0]:>2}:{c[1]:<2}"[:5].rjust(5)
        print(line)
    print()


def plot(rows, num_pages, out_path):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[matplotlib/numpy unavailable: {e}] — ASCII fallback:")
        print_ascii(rows, num_pages)
        return

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
    ax.set_yticklabels([f"t{r['task_pos']}" for r in rows], fontsize=7)
    ax.set_xlabel("physical SMEM page")
    ax.set_ylabel("task (worker-queue order)")
    ax.set_title("v2 SMEM planner: physical-page assignment per task\n"
                 "(color = region release_step; gray = unused)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("release_step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}  ({n} tasks x {num_pages} pages)")
    print_ascii(rows, num_pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="./permanent_output_dir/task_graph.json",
                    help="planned task-graph JSON path")
    ap.add_argument("--worker", type=int, default=0, help="worker queue index")
    ap.add_argument("--task-filter", default="linear",
                    help="substring a task's region names must contain "
                         "(e.g. 'linear'); '' = all page-touching tasks")
    ap.add_argument("--max-tasks", type=int, default=24,
                    help="max rows (tasks) to plot; 0 = no limit")
    ap.add_argument("--out", default="/tmp/page_plan.png", help="output PNG")
    args = ap.parse_args()

    graph = load_graph(args.plan)
    rows, num_pages = collect_rows(graph, args.worker, args.task_filter,
                                   args.max_tasks)
    plot(rows, num_pages, args.out)


if __name__ == "__main__":
    main()
