#ifndef MIRAGE_USE_CUTLASS_KERNEL
// Mirage use a flag (use_cutlass_kernel) to control whether use the cutlass
// version kernel or not.
#define MIRAGE_USE_CUTLASS_KERNEL 1
#endif // MIRAGE_USE_CUTLASS_KERNEL

#include "argmax.cuh"
#include "embedding.cuh"
#include "identity.cuh"
#include "multitoken_paged_attention.cuh"
#include "reduction.cuh"
#include "rmsnorm.cuh"
#include "rotary_embedding.cuh"
#include "silu_mul.cuh"

#ifdef USE_NVSHMEM
#include "allreduce.cuh"
#endif

#if MIRAGE_USE_CUTLASS_KERNEL
#include "linear_cutlass.cuh"
#include "moe_linear.cuh"
#else
#include "linear.cuh"
#endif // MIRAGE_USE_CUTLASS_KERNEL