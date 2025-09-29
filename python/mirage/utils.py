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
    props = torch.cuda.get_device_properties(rank)
    sm_cnt = props.multi_processor_count
    print("sm_cnt: ", sm_cnt)
    worker = 0
    if sm_cnt >= 160:
        worker = 144
    elif sm_cnt >= 132:
        worker = 128
    elif sm_cnt >= 108:
        worker = 96
    elif sm_cnt >= 68:
        worker = 64
    elif sm_cnt >= 40:
        worker = 30
    else:
        worker = 20
    return worker, get_scheduler(sm_cnt, worker)
