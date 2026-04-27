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
    263: "TASK_PAGED_ATTENTION_SPLIT_KV_SM100",
    264: "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100",
    265: "TASK_SAMPLING_SM100",
    266: "TASK_MLA_DECODE_SM100",
    267: "TASK_MLA_REDUCE_SM100",
    268: "TASK_MLA_PREFILL_SM100",
    269: "TASK_MLA_MTP_DECODE_SM100",
    270: "TASK_MLA_MTP_REDUCE_SM100",
    271: "TASK_MTP_VERIFY_STRICT",
    272: "TASK_MTP_ACCEPT_COMMIT",
    273: "TASK_MTP_TOKEN_SCATTER",
    274: "TASK_MTP_PREPARE_VERIFY",
    275: "TASK_QUANTIZE_FP8_SM100",
    276: "TASK_LINEAR_FP8_SM100",
    277: "TASK_LINEAR_FP8_WITH_RESIDUAL_SM100",
    278: "TASK_MLA_KV_GATHER_SM100",
    280: "TASK_MOE_TOPK_SIGMOID_SM100",
    281: "TASK_ELEMENTWISE_ADD_SM100",
    282: "TASK_SOFTMAX_GATHER_SM100",
    283: "TASK_MTP_VERIFY_PROBABILISTIC",
    284: "TASK_PROB_SCATTER_SM100",
    285: "TASK_MTP_FLOAT_SCATTER",
    286: "TASK_PROB_EXTRACT_SM100",
    287: "TASK_MLA_MTP_DECODE_TP2_SM100",
    288: "TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100",
    289: "TASK_MLA_MTP_DECODE_TP4_SM100",
    290: "TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100",
    291: "TASK_MLA_MTP_DECODE_TP8_SM100",
    292: "TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100",
    293: "TASK_MLA_KV_GATHER_SPLIT_SM100",
    294: "TASK_MTP_BUILD_EMBED_INPUT",
    298: "TASK_SM100_TASK_END",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
}


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    event_no = tag >> 19
    block_group_tag = (tag >> 11) & 0xFF
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
