import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator

# Task-type id -> symbolic name, used to label profiler events.
#
# SOURCE OF TRUTH: `enum TaskType` in
# include/mirage/persistent_kernel/runtime_header.h. The profiler writes the
# raw enum value into the event tag (profiler.h `encode_tag`), so this table
# must mirror that enum EXACTLY -- every entry below is transcribed from it and
# carries the declaring line as of commit e779efb.
#
# Line numbers drift; ids and names do not. `tests/runtime_python/
# test_profiler_task_names.py` re-parses the header and asserts this map is
# complete and exact, so a new TaskType fails that test instead of silently
# showing up as `UNKNOWN_<id>` (or, worse, under a stale wrong name -- 298 read
# TASK_SM100_TASK_END here for several releases while the header had moved
# TASK_SM100_TASK_END to 299 and given 298 to TASK_DFLASH_KV_STORE_SM100).
event_name_list = {
    0: "TASK_TERMINATE",                               # runtime_header.h:85
    10: "TASK_BEGIN_TASK_GRAPH",                       # runtime_header.h:86
    101: "TASK_EMBEDDING",                             # runtime_header.h:88
    102: "TASK_RMS_NORM_LINEAR",                       # runtime_header.h:89
    103: "TASK_ATTENTION_1",                           # runtime_header.h:90
    104: "TASK_ATTENTION_2",                           # runtime_header.h:91
    105: "TASK_SILU_MUL_LINEAR_WITH_RESIDUAL",         # runtime_header.h:92
    106: "TASK_ALLREDUCE",                             # runtime_header.h:93
    107: "TASK_REDUCE",                                # runtime_header.h:94
    108: "TASK_LINEAR_WITH_RESIDUAL",                  # runtime_header.h:95
    109: "TASK_ARGMAX",                                # runtime_header.h:96
    110: "TASK_ARGMAX_PARTIAL",                        # runtime_header.h:97
    111: "TASK_ARGMAX_REDUCE",                         # runtime_header.h:98
    112: "TASK_FIND_NGRAM_PARTIAL",                    # runtime_header.h:99
    113: "TASK_FIND_NGRAM_GLOBAL",                     # runtime_header.h:100
    114: "TASK_TARGET_VERIFY_GREEDY",                  # runtime_header.h:101
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",         # runtime_header.h:102
    116: "TASK_PAGED_ATTENTION_1",                     # runtime_header.h:103
    117: "TASK_PAGED_ATTENTION_2",                     # runtime_header.h:104
    118: "TASK_SILU_MUL",                              # runtime_header.h:105
    119: "TASK_RMS_NORM",                              # runtime_header.h:106
    120: "TASK_LINEAR",                                # runtime_header.h:107
    121: "TASK_IDENTITY",                              # runtime_header.h:108
    150: "TASK_HOPPER_TASK_BEGIN",                     # runtime_header.h:110
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",           # runtime_header.h:111
    152: "TASK_LINEAR_HOPPER",                         # runtime_header.h:112
    153: "TASK_PAGED_ATTENTION_HOPPER",                # runtime_header.h:113
    154: "TASK_RMS_NORM_HOPPER",                       # runtime_header.h:114
    155: "TASK_LINEAR_SWAPAB_HOPPER",                  # runtime_header.h:115
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",    # runtime_header.h:116
    157: "TASK_LINEAR_CUTLASS_HOPPER",                 # runtime_header.h:117
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",   # runtime_header.h:118
    159: "TASK_SILU_MUL_HOPPER",                       # runtime_header.h:119
    160: "TASK_EMBEDDING_HOPPER",                      # runtime_header.h:120
    161: "TASK_MOE_W13_LINEAR_SM90",                   # runtime_header.h:121
    162: "TASK_MOE_W2_LINEAR_SM90",                    # runtime_header.h:122
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",           # runtime_header.h:123
    164: "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER",       # runtime_header.h:124
    198: "TASK_HOPPER_TASK_END",                       # runtime_header.h:125
    200: "TASK_SCHD_TASKS",                            # runtime_header.h:199
    201: "TASK_SCHD_EVENTS",                           # runtime_header.h:200
    202: "TASK_GET_EVENT",                             # runtime_header.h:201
    203: "TASK_GET_NEXT_TASK",                         # runtime_header.h:202
    204: "TASK_SCHD_PREPARE_BATCH",                    # runtime_header.h:203
    230: "TASK_SM100_TASK_BEGIN",                      # runtime_header.h:127
    231: "TASK_SM100_TMA_START_TASK",                  # runtime_header.h:128
    232: "TASK_COPY",                                  # runtime_header.h:129
    233: "TASK_CONCAT",                                # runtime_header.h:130
    234: "TASK_GDN_CONV1D_SM100",                      # runtime_header.h:134
    235: "TASK_EAGLE3_D2T_REMAP",                      # runtime_header.h:135
    236: "TASK_EAGLE3_COMMIT",                         # runtime_header.h:136
    237: "TASK_GDN_RECURRENT_SM100",                   # runtime_header.h:141
    238: "TASK_SIGMOID_GATE_MUL_ADD_SM100",            # runtime_header.h:148
    241: "TASK_MOE_W13_FP8_BLOCKSCALE_SM100",          # runtime_header.h:152
    242: "TASK_MOE_W2_FP8_BLOCKSCALE_SM100",           # runtime_header.h:153
    243: "TASK_MOE_SILU_MUL_QUANTIZE_FP8_SM100",       # runtime_header.h:161
    244: "TASK_RMS_NORM_QUANTIZE_FP8_SM100",           # runtime_header.h:168
    248: "TASK_MOE_W13_FP8_SM100",                     # runtime_header.h:154
    249: "TASK_MOE_W2_FP8_SM100",                      # runtime_header.h:138
    251: "TASK_SPLITK_LINEAR_SM100",                   # runtime_header.h:139
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",            # runtime_header.h:140
    253: "TASK_LINEAR_SM100",                          # runtime_header.h:141
    254: "TASK_MOE_W13_LINEAR_SM100",                  # runtime_header.h:142
    255: "TASK_MOE_W2_LINEAR_SM100",                   # runtime_header.h:143
    256: "TASK_SM100_TMA_END_TASK",                    # runtime_header.h:144
    257: "TASK_ATTN_SM100",                            # runtime_header.h:145
    258: "TASK_ARGMAX_REDUCE_SM100",                   # runtime_header.h:146
    259: "TASK_ARGMAX_PARTIAL_SM100",                  # runtime_header.h:147
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",                # runtime_header.h:148
    261: "TASK_MOE_MUL_SUM_ADD_SM100",                 # runtime_header.h:149
    262: "TASK_TENSOR_INIT",                           # runtime_header.h:150
    263: "TASK_PAGED_ATTENTION_SPLIT_KV_SM100",        # runtime_header.h:151
    264: "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100",  # runtime_header.h:152
    265: "TASK_SAMPLING_SM100",                        # runtime_header.h:153
    266: "TASK_MLA_DECODE_SM100",                      # runtime_header.h:154
    267: "TASK_MLA_REDUCE_SM100",                      # runtime_header.h:155
    268: "TASK_MLA_PREFILL_SM100",                     # runtime_header.h:156
    269: "TASK_MLA_MTP_DECODE_SM100",                  # runtime_header.h:157
    270: "TASK_MLA_MTP_REDUCE_SM100",                  # runtime_header.h:158
    271: "TASK_MTP_VERIFY_STRICT",                     # runtime_header.h:159
    272: "TASK_MTP_ACCEPT_COMMIT",                     # runtime_header.h:160
    273: "TASK_MTP_TOKEN_SCATTER",                     # runtime_header.h:161
    274: "TASK_MTP_PREPARE_VERIFY",                    # runtime_header.h:162
    275: "TASK_QUANTIZE_FP8_SM100",                    # runtime_header.h:163
    276: "TASK_LINEAR_FP8_SM100",                      # runtime_header.h:164
    277: "TASK_LINEAR_FP8_WITH_RESIDUAL_SM100",        # runtime_header.h:165
    278: "TASK_MLA_KV_GATHER_SM100",                   # runtime_header.h:166
    279: "TASK_LINEAR_FP8_BLOCKSCALE_SM100",           # runtime_header.h:170
    280: "TASK_MOE_TOPK_SIGMOID_SM100",                # runtime_header.h:171
    281: "TASK_ELEMENTWISE_ADD_SM100",                 # runtime_header.h:172
    282: "TASK_SOFTMAX_GATHER_SM100",                  # runtime_header.h:173
    283: "TASK_MTP_VERIFY_PROBABILISTIC",              # runtime_header.h:174
    284: "TASK_PROB_SCATTER_SM100",                    # runtime_header.h:175
    285: "TASK_MTP_FLOAT_SCATTER",                     # runtime_header.h:176
    286: "TASK_PROB_EXTRACT_SM100",                    # runtime_header.h:177
    287: "TASK_MLA_MTP_DECODE_TP2_SM100",              # runtime_header.h:179
    288: "TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100",       # runtime_header.h:180
    289: "TASK_MLA_MTP_DECODE_TP4_SM100",              # runtime_header.h:181
    290: "TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100",       # runtime_header.h:182
    291: "TASK_MLA_MTP_DECODE_TP8_SM100",              # runtime_header.h:183
    292: "TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100",       # runtime_header.h:184
    293: "TASK_MLA_KV_GATHER_SPLIT_SM100",             # runtime_header.h:186
    294: "TASK_MTP_BUILD_EMBED_INPUT",                 # runtime_header.h:189
    295: "TASK_MLA_PREFILL_TP8_SM100",                 # runtime_header.h:191
    296: "TASK_DFLASH_ATTENTION_SM100",                # runtime_header.h:193
    297: "TASK_DFLASH_NORM_ROPE_SM100",                # runtime_header.h:195
    298: "TASK_DFLASH_KV_STORE_SM100",                 # runtime_header.h:197
    299: "TASK_SM100_TASK_END",                        # runtime_header.h:198
    300: "TASK_MULTIGPU_TASK_BEGIN",                   # runtime_header.h:205
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",         # runtime_header.h:206
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",                # runtime_header.h:207
    349: "TASK_MULTIGPU_TASK_END",                     # runtime_header.h:208
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


def _decode_events(profiler_buffer: torch.Tensor):
    """Yield decoded events from an on-device profiler buffer.

    First yield is a header tuple ("__header__", num_blocks, num_groups);
    subsequent yields are (block_idx, group_idx, event_idx, event_no,
    event_type, timestamp).
    """
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    yield ("__header__", num_blocks, num_groups)

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue

        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        event_no, block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        yield (block_idx, group_idx, event_idx, event_no, event_type, timestamp)


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    file_name: str,
) -> None:
    events = _decode_events(profiler_buffer)
    _, num_blocks, num_groups = next(events)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    def _tid_for(block_idx, group_idx):
        """Look up a (block, group) track, creating it if the header didn't
        cover it.

        The header's `nblocks` is written by one block of one launch. It is
        supposed to describe the whole global block-index space (see
        `PROFILER_INIT_GLOBAL` in include/mirage/persistent_kernel/profiler.h),
        but a mismatch used to make this a hard `KeyError: (80, 0)` that killed
        the export -- and, because __call__ runs this before the CSV export,
        killed the CSV too. Grow on demand instead of crashing.
        """
        key = (block_idx, group_idx)
        if key not in tid_map:
            pid = tgen.create_group(f"block_{block_idx}")
            tid_map[key] = pid.create_group(f"group_{group_idx}")
        return tid_map[key]

    for block_idx, group_idx, event_idx, event_no, event_type, timestamp in events:
        name = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}")
        event = name + f"_{event_no}"
        tid = _tid_for(block_idx, group_idx)

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


def export_to_csv(
    profiler_buffer: torch.Tensor,
    file_name: str,
) -> None:
    """Write one CSV row per fully-paired task event (and per kInstant event).

    Columns: task_type_id, task_type_name, block_idx, group_idx, event_no,
             begin_ts, end_ts, duration_ns.

    Raises RuntimeError on a dangling BEGIN with no matching END (indicates
    profiler buffer overflow or a code bug). Timestamps are raw 32-bit
    %globaltimer_lo values; durations are computed modulo 2^32, so traces
    longer than ~4.3s will be incorrect — same limitation as the perfetto
    export.
    """
    events = _decode_events(profiler_buffer)
    next(events)  # discard header

    pending = {}  # (block, group, event_idx) -> (event_no, begin_ts)
    rows = []

    for block_idx, group_idx, event_idx, event_no, event_type, timestamp in events:
        key = (block_idx, group_idx, event_idx)
        name = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}")

        if event_type == EventType.kBegin.value:
            if key in pending:
                prev_no, prev_ts = pending[key]
                raise RuntimeError(
                    f"dangling BEGIN: block={block_idx} group={group_idx} "
                    f"event={name} event_no={prev_no} ts={prev_ts} has no END "
                    f"before next BEGIN at event_no={event_no}"
                )
            pending[key] = (event_no, timestamp)
        elif event_type == EventType.kEnd.value:
            if key not in pending:
                raise RuntimeError(
                    f"END without matching BEGIN: block={block_idx} "
                    f"group={group_idx} event={name} event_no={event_no}"
                )
            begin_no, begin_ts = pending.pop(key)
            duration = (timestamp - begin_ts) & 0xFFFFFFFF
            rows.append(
                (event_idx, name, block_idx, group_idx, begin_no,
                 begin_ts, timestamp, duration)
            )
        elif event_type == EventType.kInstant.value:
            rows.append(
                (event_idx, name, block_idx, group_idx, event_no,
                 timestamp, timestamp, 0)
            )

    if pending:
        (b, g, e), (no, ts) = next(iter(pending.items()))
        name = event_name_list.get(e, f"UNKNOWN_{e}")
        raise RuntimeError(
            f"{len(pending)} dangling BEGIN event(s) with no matching END "
            f"(profiler buffer likely overflowed). Example: block={b} "
            f"group={g} event={name} event_no={no} ts={ts}"
        )

    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_type_id", "task_type_name", "block_idx", "group_idx",
            "event_no", "begin_ts", "end_ts", "duration_ns",
        ])
        writer.writerows(rows)
