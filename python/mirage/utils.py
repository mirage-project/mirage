class GPUInfo:

    # Add more archs
    MAX_SHARED_MEMORY_MAP = {
        80: 96 * 1024,  # Ampere
        86: 96 * 1024,  # A6000
    }


def get_shared_memory_capacity(target_cc):
    return GPUInfo.MAX_SHARED_MEMORY_MAP.get(target_cc, 0)
