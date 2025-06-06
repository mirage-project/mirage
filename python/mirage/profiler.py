import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator

event_name_list = {
    2000: "TB_UNKOWN",
    2001: "TB_INPUT_OP",
    2002: "TB_OUTPUT_OP",
    2003: "TB_MATMUL_OP",
    2100: "TB_EXP_OP",
    2101: "TB_SQUARE_OP",
    2102: "TB_SQRT_OP",
    2103: "TB_MUL_SCALAR_OP",
    2104: "TB_SILU_OP",
    2105: "TB_SIGMOID_OP",
    2106: "TB_GELU_OP",
    2150: "TB_RELU_OP",
    2151: "TB_CLAMP_OP",
    2160: "TB_LOG_OP",
    2200: "TB_ADD_OP",
    2201: "TB_MUL_OP",
    2202: "TB_DIV_OP",
    2203: "TB_SUB_OP",
    2204: "TB_POW_OP",
    2300: "TB_REDUCTION_FIRST_OP_ID",
    2301: "TB_REDUCTION_0_OP",
    2302: "TB_REDUCTION_1_OP",
    2303: "TB_REDUCTION_2_OP",
    2304: "TB_REDUCTION_0_TO_DIMX_OP",
    2305: "TB_REDUCTION_1_TO_DIMX_OP",
    2306: "TB_REDUCTION_2_TO_DIMX_OP",
    2307: "TB_REDUCTION_0_MAX_OP",
    2308: "TB_REDUCTION_1_MAX_OP",
    2309: "TB_REDUCTION_2_MAX_OP",
    2349: "TB_REDUCTION_LAST_OP_ID",
    2350: "TB_RMS_NORM_OP",
    2400: "TB_CONCAT_FIRST_OP_ID",
    2401: "TB_CONCAT_1_OP",
    2402: "TB_CONCAT_2_OP",
    2409: "TB_CONCAT_LAST_OP_ID",
    2411: "TB_CONCAT_THEN_MATMUL_OP",
    2420: "TB_SPLIT_FIRST_OP_ID",
    2421: "TB_SPLIT_1_OP",
    2422: "TB_SPLIT_2_OP",
    2429: "TB_SPLIT_LAST_OP_ID",
    2500: "TB_FORLOOP_ACCUM_FIRST_OP",
    2501: "TB_FORLOOP_ACCUM_RED_LD_SUM_OP",
    2502: "TB_FORLOOP_ACCUM_RED_LD_MEAN_OP",
    2503: "TB_FORLOOP_ACCUM_RED_LD_RMS_OP",
    2504: "TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP",
    2505: "TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP",
    2506: "TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP",
    2507: "TB_FORLOOP_ACCUM_MAX_OP",
    2599: "TB_FORLOOP_ACCUM_LAST_OP",
    2999: "TB_CUSTOMIZED_OP",
}


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    event_no = tag >> 24
    block_group_tag = (tag >> 12) & 0xFFF
    event_idx = (tag >> 2) & 0x3FF
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
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        event = event_name_list[event_idx + 2000] + f"_{event_no}"
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
