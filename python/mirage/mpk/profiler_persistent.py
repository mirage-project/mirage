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
    105: "TASK_SILU_MUL_LINEAR_WITH_RESIDUAL",
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
    164: "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER",
    198: "TASK_HOPPER_TASK_END",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    204: "TASK_SCHD_PREPARE_BATCH",
    230: "TASK_SM100_TASK_BEGIN",
    231: "TASK_SM100_TMA_START_TASK",
    246: "TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100",
    247: "TASK_LINEAR_FP8_SWAPAB_SM100",
    248: "TASK_MOE_W13_FP8_SM100",
    249: "TASK_MOE_W2_FP8_SM100",
    250: "TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    256: "TASK_SM100_TMA_END_TASK",
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
    279: "TASK_LINEAR_FP8_BMM_SM100",
    280: "TASK_MOE_TOPK_SIGMOID_SM100",
    281: "TASK_ELEMENTWISE_ADD_SM100",
    282: "TASK_SOFTMAX_GATHER_SM100",
    283: "TASK_MTP_VERIFY_PROBABILISTIC",
    284: "TASK_PROB_SCATTER_SM100",
    285: "TASK_MTP_FLOAT_SCATTER",
    286: "TASK_PROB_EXTRACT_SM100",
    287: "TASK_MLA_MTP_DECODE_TP2_SM100",
    288: "TASK_MLA_MTP_DECODE_TP_REDUCE_SM100",
    289: "TASK_MLA_MTP_DECODE_TP4_SM100",
    291: "TASK_MLA_MTP_DECODE_TP8_SM100",
    293: "TASK_MLA_KV_GATHER_SPLIT_SM100",
    294: "TASK_MTP_BUILD_EMBED_INPUT",
    295: "TASK_MLA_PREFILL_TP8_SM100",
    296: "TASK_MLA_UNIFIED_SM100",
    297: "TASK_MLA_KV_GATHER_UNIFIED_SM100",
    298: "TASK_MLA_PREFILL_TP8_CHUNKED_SM100",
    299: "TASK_MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
    303: "TASK_NVSHMEM_GLOBAL_ARGMAX",
    304: "TASK_DEEPSEEK_MLA_ROPE_SM100",
    305: "TASK_MLA_PREFILL_TP8_CHUNKED_REDUCE_SM100",
    306: "TASK_FP8_GEMM_DENSE_SM100",
    309: "TASK_FUSED_RMSNORM_QUANTIZE_FP8_SM100",
    311: "TASK_FP8_GROUP_GEMM_SMALLM_SM100",
    312: "TASK_FP8_GROUP_GEMM_LARGEM_SM100",
    313: "TASK_MOE_PERMUTE_SM100",
    314: "TASK_MOE_UNPERMUTE_SM100",
    315: "TASK_TRANSPOSE_SCALE_SM100",
    316: "TASK_ASSEMBLE_Q_DECODE_SM100",
    320: "TASK_SM100_TASK_END",
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
    pid_map = {}
    track_map = {}

    def _get_tid(block_idx, group_idx):
        # Lazily materialize the (block, group) track. The buffer header's
        # num_blocks/num_groups can under-count the worker grid at higher TP
        # (e.g. TP8 emitted events for block 48 while the header reported
        # fewer), which previously KeyError'd. Create on demand instead.
        key = (block_idx, group_idx)
        tid = tid_map.get(key)
        if tid is not None:
            return tid
        pid = pid_map.get(block_idx)
        if pid is None:
            pid = tgen.create_group(f"block_{block_idx}")
            pid_map[block_idx] = pid
        tid = pid.create_group(f"group_{group_idx}")
        tid_map[key] = tid
        return tid

    for block_idx in range(num_blocks):
        for group_idx in range(num_groups):
            _get_tid(block_idx, group_idx)

    for block_idx, group_idx, event_idx, event_no, event_type, timestamp in events:
        event = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}") + f"_{event_no}"
        tid = _get_tid(block_idx, group_idx)

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
    # A profile-start-step / single-graph capture clips begin/end pairs at the
    # window edges: an END can appear whose BEGIN was before the window, and a
    # BEGIN can be left open whose END is after it. Those boundary partials are
    # expected, so skip them (don't abort the whole export); count them so a
    # genuine buffer OVERFLOW (many dangling) still surfaces as a loud warning.
    n_orphan_end = 0
    n_redundant_begin = 0

    for block_idx, group_idx, event_idx, event_no, event_type, timestamp in events:
        key = (block_idx, group_idx, event_idx)
        name = event_name_list.get(event_idx, f"UNKNOWN_{event_idx}")

        if event_type == EventType.kBegin.value:
            if key in pending:
                # Prior BEGIN never closed (its END fell outside the window);
                # restart from this BEGIN rather than aborting.
                n_redundant_begin += 1
            pending[key] = (event_no, timestamp)
        elif event_type == EventType.kEnd.value:
            if key not in pending:
                # END whose BEGIN preceded the capture window — skip.
                n_orphan_end += 1
                continue
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

    if n_orphan_end or n_redundant_begin or pending:
        print(
            f"[profiler csv] boundary partials skipped: {n_orphan_end} orphan "
            f"END, {n_redundant_begin} re-opened BEGIN, {len(pending)} still-open "
            f"BEGIN at EOF (expected for a mid-flight capture; a LARGE count "
            f"relative to {len(rows)} complete events would indicate buffer "
            f"overflow)."
        )

    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_type_id", "task_type_name", "block_idx", "group_idx",
            "event_no", "begin_ts", "end_ts", "duration_ns",
        ])
        writer.writerows(rows)
