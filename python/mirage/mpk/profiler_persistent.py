import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator

event_name_list = {
    10: "TASK_BEGIN_TASK_GRAPH",
    101: "TASK_EMBEDDING",
    102: "TASK_RMS_NORM_LINEAR",
    103: "TASK_ATTENTION_1",
    104: "TASK_ATTENTION_2",
    105: "TASK_SILU_MUL_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    109: "TASK_ARGMAX",
    110: "TASK_ARGMAX_PARTIAL",
    111: "TASK_ARGMAX_REDUCE",
    112: "TASK_FIND_NGRAM_PARTIAL",
    113: "TASK_FIND_NGRAM_GLOBAL",
    114: "TASK_TARGET_VERIFY_GREEDY",
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",
    116: "TASK_PAGED_ATTENTION_1",
    117: "TASK_PAGED_ATTENTION_2",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    121: "TASK_IDENTITY",
    150: "TASK_HOPPER_TASK_BEGIN",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    153: "TASK_PAGED_ATTENTION_HOPPER",
    154: "TASK_RMS_NORM_HOPPER",
    155: "TASK_LINEAR_SWAPAB_HOPPER",
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",
    157: "TASK_LINEAR_CUTLASS_HOPPER",
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",
    159: "TASK_SILU_MUL_HOPPER",
    160: "TASK_EMBEDDING_HOPPER",
    161: "TASK_MOE_W13_LINEAR_SM90",
    162: "TASK_MOE_W2_LINEAR_SM90",
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",
    198: "TASK_HOPPER_TASK_END",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    230: "TASK_SM100_TASK_BEGIN",
    248: "TASK_MOE_W13_FP8_SM100",
    249: "TASK_MOE_W2_FP8_SM100",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    257: "TASK_ATTN_SM100",
    258: "TASK_ARGMAX_REDUCE_SM100",
    259: "TASK_ARGMAX_PARTIAL_SM100",
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",
    261: "TASK_MOE_MUL_SUM_ADD_SM100",
    262: "TASK_TENSOR_INIT",
    280: "TASK_MOE_TOPK_SIGMOID_SM100",
    298: "TASK_SM100_TASK_END",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
    # v2 runtime pseudo-events (controller/phase tracks, see runtime_v2.cuh)
    204: "V2_PREPARE_BATCH",
    205: "V2_ITER_SYNC",
    206: "V2_GO_WAIT",
    207: "V2_DEP_WAIT",
    208: "V2_PAGE_WAIT",
    209: "V2_W_TMA_WAIT",
    210: "V2_MMA_EMPTY_WAIT",
    211: "V2_TMEM_READY_WAIT",
    212: "V2_MAINLOOP_WAIT",
    213: "V2_EPILOGUE_WAIT",
    214: "V2_CONSUMER_DONE_WAIT",
    # v2 linear task types
    244: "TASK_LINEAR_SM100_V2",
    245: "TASK_LINEAR_WITH_RESIDUAL_SM100_V2",
    281: "TASK_RMS_NORM_HOPPER_V2",
    282: "TASK_SILU_MUL_V2",
    283: "TASK_EMBEDDING_V2",
    284: "TASK_ATTN_SM100_V2",
    285: "TASK_ARGMAX_PARTIAL_SM100_V2",
    286: "TASK_ARGMAX_REDUCE_SM100_V2",
}


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    # layout (see profiler.h): [31:22 event_no][21:11 block_group]
    #                          [10:2 event_idx][1:0 type]
    event_no = tag >> 22
    block_group_tag = (tag >> 11) & 0x7FF
    event_idx = (tag >> 2) & 0x1FF
    event_type = tag & 0x3
    return (
        event_no,
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    file_name: str,
    task_graph=None,
) -> None:
    """task_graph (optional, v2): {"queues": per-SM task-position lists,
    "task_types": task_type per position}. When given, role-track slices are
    labeled with the global task index (..._t<pos>) so a slice can be looked
    up in task_graph.json / the page-plan figure directly."""

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    # Per-SM list of task positions that actually emit windows (everything
    # except BEGIN_TASK_GRAPH(10)/TERMINATE(0)); trace window k on a role
    # track = emitting[k % len] (profiling windows cover whole iterations).
    emitting = None
    if task_graph is not None:
        tt = task_graph["task_types"]
        emitting = [[p for p in q if tt[p] not in (0, 10)]
                    for q in task_graph["queues"]]
    win_counter = {}

    tgen = TraceGenerator(file_name)

    # v2 emits one track per (SM, warp role) plus phase tracks (sub-slices
    # inside a role's task window); name them accordingly.
    V2_ROLE_NAMES = ["consumer", "loader", "launcher", "storer", "controller",
                     "consumer-phase", "loader-phase", "launcher-phase"]

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            if num_groups >= 5 and group_idx < len(V2_ROLE_NAMES):
                gname = V2_ROLE_NAMES[group_idx]
            else:
                gname = f"group_{group_idx}"
            tid = pid.create_group(gname)
            tid_map[(block_idx, group_idx)] = tid

    # numpy fast path: visit only nonzero entries (the buffer is sparse and
    # iterating tens of millions of zeros in python dominates export time).
    buf_np = profiler_buffer_host.numpy()
    nz = buf_np.nonzero()[0]
    nz = nz[nz >= 1]
    if num_groups >= 5:
        # v2 reserves the buffer tail for raw accumulators + emitter cursors
        # (see V2_PROF_CURSOR_BASE in runtime_v2.cuh). Those are counters,
        # not events — decoding them produces garbage slices at t~0 that
        # stretch the whole timeline. Skip the tail (must match
        # V2_PROF_TAIL_ENTRIES on the device side).
        V2_PROF_TAIL_ENTRIES = (1048576 + 1) + 256 + 1024 + 256 * 7 + 256  # trig+misc+cursors+spin+suffix
        nz = nz[nz < len(buf_np) - V2_PROF_TAIL_ENTRIES]
    for i in nz:
        entry = int(buf_np[i])
        tag = entry & 0xFFFFFFFF
        timestamp = entry >> 32
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )

        # Unknown ids: junk/reserved entries (e.g. the v2 dep-spin
        # accumulators at the buffer tail) or types missing from the dict —
        # skip rather than crash the export.
        if event_idx not in event_name_list or (block_idx, group_idx) not in tid_map:
            continue
        if (emitting is not None and group_idx < 4
                and event_type == EventType.kBegin.value
                and block_idx < len(emitting) and emitting[block_idx]):
            k = win_counter.get((block_idx, group_idx), 0)
            win_counter[(block_idx, group_idx)] = k + 1
            pos = emitting[block_idx][k % len(emitting[block_idx])]
            event = event_name_list[event_idx] + f"_t{pos}"
        else:
            event = event_name_list[event_idx] + f"_{event_no}"
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()
