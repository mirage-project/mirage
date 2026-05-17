"""Smoke test for layers.argmax.nvshmem_global_argmax.NVShmemGlobalArgmax.

Multi-GPU NVSHMEM only — single-GPU test SKIPS.
"""


def test_nvshmem_global_argmax_skipped():
    print("SKIPPED: NVShmemGlobalArgmax requires multi-GPU NVSHMEM "
          "(world_size > 1, use_nvshmem). For single-rank tests use "
          "layers.Argmax / ArgmaxPartial+ArgmaxReduce instead.")


if __name__ == "__main__":
    test_nvshmem_global_argmax_skipped()
