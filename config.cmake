set(USE_CUDA ON)
set(USE_CUDNN OFF)
set(BUILD_CPP_EXAMPLES OFF)
set(USE_NKI OFF)
# AMD GPUs (ROCm): the core as host-only C++ (no CUDA headers or libraries); the static megakernel builds its layers with
# hipcc (python/mirage/mpk/gpu_backend.py). Exactly one of USE_CUDA, USE_NKI, USE_HIP is ON.
set(USE_HIP OFF)
set(USE_FORMAL_VERIFIER OFF)
