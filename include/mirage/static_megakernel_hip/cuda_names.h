// cuda_names.h -- the CUDA runtime names the generated layer's host code uses
// (static_schedule.py HOST_CODE and kernel_code), as their HIP equivalents. The
// generator writes one host code for both backends; on HIP this header comes
// first (static_schedule.BACKEND_INCLUDES). Each name has the same meaning in
// HIP. Only the names the generated code uses are here: a new name in the
// generator fails to compile until it is added.
#pragma once
#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotReady hipErrorNotReady
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// the generated cleanup calls these without using the result (as on CUDA); HIP
// marks the hipError_t type nodiscard, so these four return a result that
// converts to hipError_t and may be dropped
struct StaticHipResult {
  hipError_t e;
  operator hipError_t() const {
    return e;
  }
};
static inline StaticHipResult static_hip_set_device(int d) {
  return {hipSetDevice(d)};
}
static inline StaticHipResult static_hip_free(void *p) {
  return {hipFree(p)};
}
static inline StaticHipResult static_hip_event_destroy(hipEvent_t e) {
  return {hipEventDestroy(e)};
}
static inline StaticHipResult static_hip_stream_destroy(hipStream_t s) {
  return {hipStreamDestroy(s)};
}
#define cudaSetDevice static_hip_set_device
#define cudaMalloc hipMalloc
#define cudaFree static_hip_free
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy static_hip_stream_destroy

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy static_hip_event_destroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventQuery hipEventQuery
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaFuncSetAttribute(f, a, v) hipFuncSetAttribute((const void *)(f), a, v)
#define cudaFuncAttributeMaxDynamicSharedMemorySize                            \
  hipFuncAttributeMaxDynamicSharedMemorySize

// the kernel's parameter qualifier on CUDA (a parameter in constant space); HIP
// kernel arguments are in constant memory already
#define __grid_constant__
