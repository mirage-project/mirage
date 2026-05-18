import torch

# This function returns the shared memory limit (in bytes)
# for the given GPU hardware architecture
def get_shared_memory_capacity(target_cc):
    if target_cc == 80:
        # A100 GPUs
        return 163 * 1024
    elif target_cc == 86:
        # A5000 GPUs
        return 99 * 1024
    elif target_cc == 89:
        # A6000 GPUs
        return 99 * 1024
    elif target_cc == 90:
        # H100 GPUs
        return 223 * 1024
    elif target_cc == 100:
        # B200 GPUs
        return 227 * 1024
    else:
        assert False, "Unsupported compute capacity: {}".format(target_cc)


def get_scheduler(sm_cnt, worker):
    scheduler = 4 * (sm_cnt - worker)
    assert scheduler > 0, "worker count is not compatible with sm count on"
    "the GPU"
    return 4 * (sm_cnt - worker)

# This method auto probe GPUs and return the worker and scheduler count for
# them.
def get_configurations_from_gpu(rank):
    # Reference: https://github.com/mirage-project/mirage/issues/354
    import os as _os
    props = torch.cuda.get_device_properties(rank)
    sm_cnt = props.multi_processor_count
    print("sm_cnt: ", sm_cnt)
    # MPK_SM_BUDGET env override (experimental). Reduces the effective
    # SM count fed into the worker/scheduler split. Useful when
    # cooperative-launch capacity is the bottleneck: on B200 (148 SMs)
    # the default `128 worker + 20 scheduler-SM = 148` launch maxes out
    # the cooperative grid; some configurations (TP=2 with 56 NVSHMEM
    # teams) need a margin or the cuLaunchCooperativeKernel rejects the
    # launch with `cudaErrorLaunchFailed`. Setting MPK_SM_BUDGET=144
    # trims 4 SMs of margin and rebalances workers+schedulers
    # accordingly.
    sm_budget_override = _os.environ.get("MPK_SM_BUDGET")
    if sm_budget_override is not None:
        sm_budget = int(sm_budget_override)
        assert 8 <= sm_budget <= sm_cnt, (
            f"MPK_SM_BUDGET={sm_budget} out of range [8, {sm_cnt}]")
        sm_cnt_for_split = sm_budget
    else:
        sm_cnt_for_split = sm_cnt
    # MPK_FORCE_NUM_WORKERS env override (experimental). Use only when you
    # know what scheduler count the rest of the chip still has room for —
    # `get_scheduler` uses `4 * (sm_cnt - worker)`, so worker=136 on a
    # 148-SM B200 leaves 48 schedulers (12 SMs × 4) which is plenty for
    # DSv3-scale graphs but margin-tight; worker=144 leaves only 16
    # schedulers and may starve the event-dispatcher.
    forced = _os.environ.get("MPK_FORCE_NUM_WORKERS")
    if forced is not None:
        worker = int(forced)
        assert worker < sm_cnt_for_split, (
            f"MPK_FORCE_NUM_WORKERS={worker} >= sm_budget={sm_cnt_for_split} "
            "would leave 0 schedulers")
        return worker, get_scheduler(sm_cnt_for_split, worker)
    worker = 0
    if sm_cnt_for_split >= 160:
        worker = 144
    elif sm_cnt_for_split >= 144:
        worker = 128
    elif sm_cnt_for_split >= 132:
        worker = 128
    elif sm_cnt_for_split >= 108:
        worker = 96
    elif sm_cnt_for_split >= 68:
        worker = 64
    elif sm_cnt_for_split >= 40:
        worker = 30
    else:
        worker = 20
    return worker, get_scheduler(sm_cnt_for_split, worker)
