#include <cassert>

#include <cublas_v2.h>

namespace kn {

// Convert from T to cudaDataType_t
template <typename T>
static inline cudaDataType_t data_t2cuda_data_type_t() {
  if (std::is_same_v<T, half> || std::is_same_v<T, cute::half_t>) {
    return CUDA_R_16F;
  } else if (std::is_same_v<T, nv_bfloat16> ||
             std::is_same_v<T, cute::bfloat16_t>) {
    return CUDA_R_16BF;
  } else if (std::is_same_v<T, float>) {
    return CUDA_R_32F;
  } else {
    assert(false);
  }
  __builtin_unreachable();
}

// Get the alpha and beta type for a given compute type
template <cublasComputeType_t compute_type>
struct CublasComputeType2AlphaBetaType {};

template <>
struct CublasComputeType2AlphaBetaType<CUBLAS_COMPUTE_16F> {
  using type = half;
};

template <>
struct CublasComputeType2AlphaBetaType<CUBLAS_COMPUTE_16F_PEDANTIC> {
  using type = half;
};

template <>
struct CublasComputeType2AlphaBetaType<CUBLAS_COMPUTE_32F> {
  using type = float;
};

template <>
struct CublasComputeType2AlphaBetaType<CUBLAS_COMPUTE_32F_PEDANTIC> {
  using type = float;
};

// A small utility for keeping the cublas handle
class CublasHandleKeeper {
private:
  cublasHandle_t handle;

public:
  CublasHandleKeeper() {
    cublasCreate(&handle);
  }
  ~CublasHandleKeeper() {
    cublasDestroy(handle);
  }
  cublasHandle_t get() {
    return handle;
  }
} cublas_handle_keeper;

// Perform GEMM
// A should have a shape of batch_size x m x k
// B should have a shape of batch_size x k x n
// C should have a shape of batch_size x m x n
template <cublasComputeType_t COMPUTE_TYPE,
          typename TA,
          typename TB,
          typename TC>
static void gemm(
    // Arguments for matrix info
    TC *C,
    const TA *A,
    const TB *B,
    int m,
    int n,
    int k,
    size_t stride_m_A,
    size_t stride_k_A,
    size_t stride_k_B,
    size_t stride_n_B,
    size_t stride_m_C,
    size_t stride_n_C,

    // Arguments for batch info
    size_t batch_size,
    size_t batch_stride_A,
    size_t batch_stride_B,
    size_t batch_stride_C) {
  assert(stride_m_A == 1 || stride_k_A == 1);
  assert(stride_k_B == 1 || stride_n_B == 1);
  assert(stride_m_C == 1 || stride_n_C == 1);
  // Here "Transposed" means that the last dimension is the innermost dim,
  // while "Normal" means that the 2nd last dimension is the innermost dim
  // (analogous to the cuBLAS convention)
  // (Transposed = row major, Normal = column major)
  bool trans_C = stride_n_C == 1;
  if (trans_C) {
    // If C is going to be interpreted in row-major, since cuBLAS stores C in
    // column-major, we need to let cuBLAS calculate C^T = B^T A^T and store
    // it, so that when reading, we will get (C^T)^T = AB
    std::swap(m, n);
    std::swap(stride_m_C, stride_n_C);
    std::swap(A, B);
    std::swap(stride_m_A, stride_n_B);
    std::swap(stride_k_A, stride_k_B);
    std::swap(batch_stride_A, batch_stride_B);
  }
  bool trans_A = stride_k_A == 1;
  bool trans_B = stride_n_B == 1;
  using alpha_beta_t =
      typename CublasComputeType2AlphaBetaType<COMPUTE_TYPE>::type;
  alpha_beta_t alpha = 1.0;
  alpha_beta_t beta = 0.0;
  cublasStatus_t status =
      cublasGemmStridedBatchedEx(cublas_handle_keeper.get(),
                                 trans_A ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 trans_B ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 m,
                                 n,
                                 k,
                                 &alpha,
                                 A,
                                 data_t2cuda_data_type_t<TA>(),
                                 stride_m_A + stride_k_A - 1,
                                 batch_stride_A,
                                 B,
                                 data_t2cuda_data_type_t<TB>(),
                                 stride_k_B + stride_n_B - 1,
                                 batch_stride_B,
                                 &beta,
                                 C,
                                 data_t2cuda_data_type_t<TC>(),
                                 stride_n_C,
                                 batch_stride_C,
                                 batch_size,
                                 COMPUTE_TYPE,
                                 CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "CUBLAS GEMM failed with status %d: %s (%s)\n",
            status,
            cublasGetStatusName(status),
            cublasGetStatusString(status));
    assert(false);
  }
}

} // namespace kn
