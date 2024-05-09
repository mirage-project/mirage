/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/threadblock/cuda/concat.h"
#include "mirage/threadblock/cuda/element_binary.h"
#include "mirage/threadblock/cuda/element_unary.h"
#include "mirage/threadblock/cuda/input_loader.h"
#include "mirage/threadblock/cuda/matmul.h"
#include "mirage/threadblock/cuda/output_saver.h"
#include "mirage/threadblock/cuda/reduction.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/serializer/concat_serializer.h"
#include "mirage/threadblock/serializer/element_binary_serializer.h"
#include "mirage/threadblock/serializer/element_unary_serializer.h"
#include "mirage/threadblock/serializer/input_loader_serializer.h"
#include "mirage/threadblock/serializer/matmul_serializer.h"
#include "mirage/threadblock/serializer/output_saver_serializer.h"
#include "mirage/threadblock/serializer/reduction_serializer.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/warp/cuda/matmul.h"

namespace mirage {
namespace kernel {

__global__ void customized_kernel_function(
    mirage::threadblock::NewKernelParams const new_params, int forloop_range) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  if (blockDim.y > 1 || blockDim.z > 1) {
    assert(false && "blockDim.y and blockDim.z must be 1");
  }

  extern __shared__ char smem_buffer[];

  int param_idx = 0;
  for (int i = 0; i < forloop_range; i++) {
    // start executing operators
    param_idx = 0;
    for (int op = 0; op < new_params.num_operators; op++) {
      mirage::type::TBOperatorType op_type = new_params.operator_types[op];
      if (op_type == mirage::type::TB_INPUT_OP) {
        // Assume that InputLoaders are the first operators
        void *dtensor_ptr = new_params.dmem_input_ptrs[op];
        int3 input_matrix_row_offset_block_stride;
        int3 input_matrix_column_offset_block_stride;
        int input_matrix_row_offset_forloop_stride;
        int input_matrix_column_offset_forloop_stride;
        int3 global_offset_block_stride;
        int global_offset_forloop_stride;
        int2 dtensor_matrix_shape, stensor_matrix_shape;
        int input_smem_offset;
        mirage::layout::DmemLayout dtensor_layout;
        mirage::layout::SmemLayout stensor_layout;
        mirage::threadblock::deserialize_input_loader_parameters(
            new_params.parameters,
            param_idx,
            input_matrix_row_offset_block_stride,
            input_matrix_column_offset_block_stride,
            input_matrix_row_offset_forloop_stride,
            input_matrix_column_offset_forloop_stride,
            global_offset_block_stride,
            global_offset_forloop_stride,
            dtensor_matrix_shape,
            stensor_matrix_shape,
            dtensor_layout,
            stensor_layout,
            input_smem_offset);

        int tb_offset_row =
            blockIdx.x * input_matrix_row_offset_block_stride.x +
            blockIdx.y * input_matrix_row_offset_block_stride.y +
            blockIdx.z * input_matrix_row_offset_block_stride.z +
            i * input_matrix_row_offset_forloop_stride;
        int tb_offset_column =
            blockIdx.x * input_matrix_column_offset_block_stride.x +
            blockIdx.y * input_matrix_column_offset_block_stride.y +
            blockIdx.z * input_matrix_column_offset_block_stride.z +
            i * input_matrix_column_offset_forloop_stride;
        int global_offset = blockIdx.x * global_offset_block_stride.x +
                            blockIdx.y * global_offset_block_stride.y +
                            blockIdx.z * global_offset_block_stride.z +
                            i * global_offset_forloop_stride;
        cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
        cutlass::half_t *stensor_ptr =
            (cutlass::half_t *)(smem_buffer + input_smem_offset);
        mirage::threadblock::GenericInputLoader loader(dtensor_ptr,
                                                    stensor_ptr,
                                                    dtensor_matrix_shape,
                                                    stensor_matrix_shape,
                                                    dtensor_layout,
                                                    stensor_layout,
                                                    threadIdx.x,
                                                    blockDim.x,
                                                    matrix_offset,
                                                    global_offset);
        __syncthreads();
      } else if (op_type == mirage::type::TB_OUTPUT_OP) {
        // Only save outputs after forloop
        // So we do nothing for output saver
      } else if (op_type == mirage::type::TB_MATMUL_OP) {
        int thread_idx = threadIdx.x;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;
        int m, n, k;
        int A_smem_offset, B_smem_offset, C_smem_offset;
        mirage::threadblock::deserialize_matmul_op_parameters(
            new_params.parameters,
            param_idx,
            m,
            n,
            k,
            A_smem_offset,
            B_smem_offset,
            C_smem_offset);

        cutlass::half_t *A_ptr =
            (cutlass::half_t *)(smem_buffer + A_smem_offset);
        cutlass::half_t *B_ptr =
            (cutlass::half_t *)(smem_buffer + B_smem_offset);
        cutlass::half_t *C_ptr =
            (cutlass::half_t *)(smem_buffer + C_smem_offset);

        mirage::type::ActivationType act_type =
            mirage::utils::get_matmul_activation_type(
                new_params.operator_types, op, new_params.num_operators);

        if (act_type == mirage::type::ACT_NONE) {
          mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_NONE> executor(
              A_ptr, B_ptr, C_ptr, m, n, k, thread_idx, warp_idx, lane_idx);

        } else if (act_type == mirage::type::ACT_EXP) {
          // fuse this matmul with next op
          int smem_offset, num_elements;
          mirage::threadblock::deserialize_elementunary_op_parameters(
              new_params.parameters, param_idx, smem_offset, num_elements);
          C_ptr = (cutlass::half_t *)(smem_buffer + smem_offset);
          mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_EXP> executor(
              A_ptr, B_ptr, C_ptr, m, n, k, thread_idx, warp_idx, lane_idx);
          op += 1;
        }
        __syncthreads();
      } else if (op_type == mirage::type::TB_EXP_OP) {
        int smem_offset, num_elements;
        mirage::threadblock::deserialize_elementunary_op_parameters(
            new_params.parameters, param_idx, smem_offset, num_elements);
        cutlass::half_t *base_ptr =
            (cutlass::half_t *)(smem_buffer + smem_offset);
        mirage::threadblock::ElementUnaryExecutor<cutlass::half_t> executor(
            op_type, base_ptr, num_elements, threadIdx.x, blockDim.x);
        __syncthreads();
      } else if (op_type == mirage::type::TB_DIV_OP) {
        int3 input1_shape, input2_shape;
        int input1_smem_offset, input2_smem_offset, output_smem_offset;
        mirage::threadblock::deserialize_elementbinary_op_parameters(
            new_params.parameters,
            param_idx,
            input1_shape,
            input2_shape,
            input1_smem_offset,
            input2_smem_offset,
            output_smem_offset);
        cutlass::half_t *input1_ptr =
            (cutlass::half_t *)(smem_buffer + input1_smem_offset);
        cutlass::half_t *input2_ptr =
            (cutlass::half_t *)(smem_buffer + input2_smem_offset);
        cutlass::half_t *output_ptr =
            (cutlass::half_t *)(smem_buffer + output_smem_offset);
        mirage::threadblock::ElementBinaryExecutor<cutlass::half_t> executor(
            op_type,
            input1_ptr,
            input2_ptr,
            output_ptr,
            input1_shape,
            input2_shape,
            threadIdx.x,
            blockDim.x);
        __syncthreads();
      } else if ((op_type >= mirage::type::TB_REDUCTION_FIRST_OP_ID) &&
                 (op_type <= mirage::type::TB_REDUCTION_LAST_OP_ID)) {
        int output_num_elements, reduction_degree, inner_range;
        int input_smem_offset, output_smem_offset;
        mirage::threadblock::deserialize_reduction_op_parameters(
            new_params.parameters,
            param_idx,
            output_num_elements,
            reduction_degree,
            inner_range,
            input_smem_offset,
            output_smem_offset);
        cutlass::half_t *input_ptr =
            (cutlass::half_t *)(smem_buffer + input_smem_offset);
        cutlass::half_t *output_ptr =
            (cutlass::half_t *)(smem_buffer + output_smem_offset);

        mirage::threadblock::SimpleRedunctionExecutor<cutlass::half_t> executor(
            // new_params.operator_types[op],
            input_ptr,
            output_ptr,
            output_num_elements,
            reduction_degree,
            inner_range,
            threadIdx.x,
            blockDim.x);
        __syncthreads();
      } else if ((op_type >= mirage::type::TB_CONCAT_FIRST_OP_ID) &&
                 (op_type <= mirage::type::TB_CONCAT_LAST_OP_ID)) {
        int output_num_elements, A_concat_dim_size, B_concat_dim_size,
            inner_size;
        int A_smem_offset, B_smem_offset, output_smem_offset;
        mirage::threadblock::deserialize_concat_op_parameters(
            new_params.parameters,
            param_idx,
            output_num_elements,
            A_concat_dim_size,
            B_concat_dim_size,
            inner_size,
            A_smem_offset,
            B_smem_offset,
            output_smem_offset);
        // Do nothing since we can avoid concat mem copy by
        // updating the input tensors smem_offset
      } else {
        assert(false && "Unsupported threadblock operator");
      }
    }
  }
  // Save output
  int output_saver_start_idx =
      new_params.num_operators - new_params.num_dmem_outputs;
  for (int op = output_saver_start_idx; op < new_params.num_operators; op++) {
    assert(new_params.operator_types[op] == mirage::type::TB_OUTPUT_OP);
    void *dtensor_ptr =
        new_params.dmem_output_ptrs[op - output_saver_start_idx];
    int3 output_matrix_row_offset_block_stride;
    int3 output_matrix_column_offset_block_stride;
    int3 global_offset_block_stride;
    int2 dtensor_matrix_shape, stensor_matrix_shape;
    int input_smem_offset, accum_smem_offset;
    mirage::layout::DmemLayout dtensor_layout;
    mirage::layout::SmemLayout stensor_layout;
    mirage::threadblock::deserialize_output_saver_parameters(
        new_params.parameters,
        param_idx,
        output_matrix_row_offset_block_stride,
        output_matrix_column_offset_block_stride,
        global_offset_block_stride,
        dtensor_matrix_shape,
        stensor_matrix_shape,
        dtensor_layout,
        stensor_layout,
        input_smem_offset,
        accum_smem_offset);
    int tb_offset_row = blockIdx.x * output_matrix_row_offset_block_stride.x +
                        blockIdx.y * output_matrix_row_offset_block_stride.y +
                        blockIdx.z * output_matrix_row_offset_block_stride.z;
    int tb_offset_column =
        blockIdx.x * output_matrix_column_offset_block_stride.x +
        blockIdx.y * output_matrix_column_offset_block_stride.y +
        blockIdx.z * output_matrix_column_offset_block_stride.z;
    // calculate global offset beyond the last two dimensions
    // global_offset captures offsets caused by partitioning other dimensions
    // such as batch matmul
    // global_offset is directly added to dtensor_ptr by the output saver
    int global_offset = blockIdx.x * global_offset_block_stride.x +
                        blockIdx.y * global_offset_block_stride.y +
                        blockIdx.z * global_offset_block_stride.z;

    // FIXME: use cutlass prologue for loading data into shared memory
    // examples/13_two_tensor_op_fusion/threadblock/
    // b2b_mma_pipelined_smem_accumulator.h prologue iterators
    cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t *)(smem_buffer + accum_smem_offset);
    mirage::threadblock::GenericOutputSaver saver(dtensor_ptr,
                                               stensor_ptr,
                                               dtensor_matrix_shape,
                                               stensor_matrix_shape,
                                               dtensor_layout,
                                               stensor_layout,
                                               threadIdx.x,
                                               blockDim.x,
                                               matrix_offset,
                                               global_offset);
    // No need to synchronize for output saver
    //__syncthreads();
  }
  assert(new_params.num_parameters == param_idx);
}

__global__ void compute_customizedop_fingerprint(
    mirage::threadblock::NewKernelParams new_params,
    int forloop_range,
    mirage::type::FPType *exp_lookup_table,
    mirage::type::FPType *div_p_lookup_table,
    mirage::type::FPType *div_q_lookup_table) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  extern __shared__ char smem_buffer[];
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  int param_idx = 0;
  int output_saver_start_idx =
      new_params.num_operators - new_params.num_dmem_outputs;
  for (int i = 0; i < forloop_range; i++) {
    param_idx = 0;
    // start executing operators
    for (int op = 0; op < new_params.num_operators; op++) {
      switch (new_params.operator_types[op]) {
        case mirage::type::TB_INPUT_OP: {
          mirage::type::FPType *dtensor_ptr =
              (mirage::type::FPType *)new_params.dmem_input_ptrs[op];
          int3 input_matrix_row_offset_block_stride;
          int3 input_matrix_column_offset_block_stride;
          int input_matrix_row_offset_forloop_stride;
          int input_matrix_column_offset_forloop_stride;
          int3 global_offset_block_stride;
          int global_offset_forloop_stride;
          int2 dtensor_matrix_shape, stensor_matrix_shape;
          int input_smem_offset;
          mirage::layout::DmemLayout dtensor_layout;
          mirage::layout::SmemLayout stensor_layout;
          mirage::threadblock::deserialize_input_loader_parameters(
              new_params.parameters,
              param_idx,
              input_matrix_row_offset_block_stride,
              input_matrix_column_offset_block_stride,
              input_matrix_row_offset_forloop_stride,
              input_matrix_column_offset_forloop_stride,
              global_offset_block_stride,
              global_offset_forloop_stride,
              dtensor_matrix_shape,
              stensor_matrix_shape,
              dtensor_layout,
              stensor_layout,
              input_smem_offset);

          // Note that input_matrix_offset_forloop_stride's x and y indicates
          // row and column
          int tb_offset_row =
              blockIdx.x * input_matrix_row_offset_block_stride.x +
              blockIdx.y * input_matrix_row_offset_block_stride.y +
              blockIdx.z * input_matrix_row_offset_block_stride.z +
              i * input_matrix_row_offset_forloop_stride;
          int tb_offset_column =
              blockIdx.x * input_matrix_column_offset_block_stride.x +
              blockIdx.y * input_matrix_column_offset_block_stride.y +
              blockIdx.z * input_matrix_column_offset_block_stride.z +
              i * input_matrix_column_offset_forloop_stride;
          int global_offset = blockIdx.x * global_offset_block_stride.x +
                              blockIdx.y * global_offset_block_stride.y +
                              blockIdx.z * global_offset_block_stride.z +
                              i * global_offset_forloop_stride;
          cutlass::MatrixCoord matrix_offset = {tb_offset_row,
                                                tb_offset_column};
          mirage::type::FPType *stensor_ptr =
              (mirage::type::FPType *)(smem_buffer + input_smem_offset);
          mirage::threadblock::TBInputLoaderFingerprinter fp(dtensor_ptr,
                                                          stensor_ptr,
                                                          dtensor_matrix_shape,
                                                          stensor_matrix_shape,
                                                          dtensor_layout,
                                                          stensor_layout,
                                                          threadIdx.x,
                                                          blockDim.x,
                                                          matrix_offset,
                                                          global_offset);
          __syncthreads();
          break;
        }
        case mirage::type::TB_OUTPUT_OP: {
          int3 output_matrix_row_offset_block_stride;
          int3 output_matrix_column_offset_block_stride;
          int3 global_offset_block_stride;
          int2 dtensor_matrix_shape, stensor_matrix_shape;
          int input_smem_offset, accum_smem_offset;
          mirage::layout::DmemLayout dtensor_layout;
          mirage::layout::SmemLayout stensor_layout;
          mirage::threadblock::deserialize_output_saver_parameters(
              new_params.parameters,
              param_idx,
              output_matrix_row_offset_block_stride,
              output_matrix_column_offset_block_stride,
              global_offset_block_stride,
              dtensor_matrix_shape,
              stensor_matrix_shape,
              dtensor_layout,
              stensor_layout,
              input_smem_offset,
              accum_smem_offset);
          mirage::type::FPType *input_stensor_ptr =
              (mirage::type::FPType *)(smem_buffer + input_smem_offset);
          mirage::type::FPType *accum_stensor_ptr =
              (mirage::type::FPType *)(smem_buffer + accum_smem_offset);
          mirage::threadblock::TBOutputAccumFingerprinter fp(input_stensor_ptr,
                                                          accum_stensor_ptr,
                                                          stensor_matrix_shape,
                                                          (i == 0),
                                                          threadIdx.x,
                                                          blockDim.x);
          __syncthreads();
          // Step 2: Save final output to dmem if this is the last forloop
          if (i == forloop_range - 1) {
            assert(op >= output_saver_start_idx);
            mirage::type::FPType *dtensor_ptr =
                (mirage::type::FPType *)
                    new_params.dmem_output_ptrs[op - output_saver_start_idx];
            int tb_offset_row =
                blockIdx.x * output_matrix_row_offset_block_stride.x +
                blockIdx.y * output_matrix_row_offset_block_stride.y +
                blockIdx.z * output_matrix_row_offset_block_stride.z;
            int tb_offset_column =
                blockIdx.x * output_matrix_column_offset_block_stride.x +
                blockIdx.y * output_matrix_column_offset_block_stride.y +
                blockIdx.z * output_matrix_column_offset_block_stride.z;
            // calculate global offset beyond the last two dimensions
            // global_offset captures offsets caused by partitioning other
            // dimensions such as batch matmul global_offset is directly added
            // to dtensor_ptr by the output saver
            int global_offset = blockIdx.x * global_offset_block_stride.x +
                                blockIdx.y * global_offset_block_stride.y +
                                blockIdx.z * global_offset_block_stride.z;
            cutlass::MatrixCoord matrix_offset = {tb_offset_row,
                                                  tb_offset_column};
            mirage::threadblock::TBOutputSaverFingerprinter fp(
                dtensor_ptr,
                accum_stensor_ptr,
                dtensor_matrix_shape,
                stensor_matrix_shape,
                dtensor_layout,
                stensor_layout,
                threadIdx.x,
                blockDim.x,
                matrix_offset,
                global_offset);
            // No need to syncthread when saving output to dmem
            //__syncthreads();
          }
          break;
        }
        case mirage::type::TB_MATMUL_OP: {
          int m, n, k;
          int A_smem_offset, B_smem_offset, C_smem_offset;
          mirage::threadblock::deserialize_matmul_op_parameters(
              new_params.parameters,
              param_idx,
              m,
              n,
              k,
              A_smem_offset,
              B_smem_offset,
              C_smem_offset);
          mirage::type::FPType *A_ptr =
              (mirage::type::FPType *)(smem_buffer + A_smem_offset);
          mirage::type::FPType *B_ptr =
              (mirage::type::FPType *)(smem_buffer + B_smem_offset);
          mirage::type::FPType *C_ptr =
              (mirage::type::FPType *)(smem_buffer + C_smem_offset);

          mirage::threadblock::TBMatmulFingerprinter fp(
              A_ptr, B_ptr, C_ptr, m, n, k, threadIdx.x, blockDim.x);
          __syncthreads();
          break;
        }
        case mirage::type::TB_EXP_OP: {
          int smem_offset, num_elements;
          mirage::threadblock::deserialize_elementunary_op_parameters(
              new_params.parameters, param_idx, smem_offset, num_elements);
          mirage::type::FPType *base_ptr =
              (mirage::type::FPType *)(smem_buffer + smem_offset);
          mirage::threadblock::TBElementUnaryFingerPrinter fp(
              new_params.operator_types[op],
              exp_lookup_table /*lookup_table*/,
              base_ptr,
              num_elements,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case mirage::type::TB_DIV_OP: {
          int3 input1_shape, input2_shape;
          int input1_smem_offset, input2_smem_offset, output_smem_offset;
          mirage::threadblock::deserialize_elementbinary_op_parameters(
              new_params.parameters,
              param_idx,
              input1_shape,
              input2_shape,
              input1_smem_offset,
              input2_smem_offset,
              output_smem_offset);
          mirage::type::FPType *input1_ptr =
              (mirage::type::FPType *)(smem_buffer + input1_smem_offset);
          mirage::type::FPType *input2_ptr =
              (mirage::type::FPType *)(smem_buffer + input2_smem_offset);
          mirage::type::FPType *output_ptr =
              (mirage::type::FPType *)(smem_buffer + output_smem_offset);
          mirage::threadblock::TBElementBinaryFingerPrinter fp(
              new_params.operator_types[op],
              div_p_lookup_table /*div_p_lookup*/,
              div_q_lookup_table /*div_q_lookup*/,
              input1_ptr,
              input2_ptr,
              output_ptr,
              input1_shape,
              input2_shape,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case mirage::type::TB_REDUCTION_0_OP:
        case mirage::type::TB_REDUCTION_1_OP:
        case mirage::type::TB_REDUCTION_2_OP:
        case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
        case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
        case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
          int output_num_elements, reduction_degree, inner_range;
          int input_smem_offset, output_smem_offset;
          mirage::threadblock::deserialize_reduction_op_parameters(
              new_params.parameters,
              param_idx,
              output_num_elements,
              reduction_degree,
              inner_range,
              input_smem_offset,
              output_smem_offset);
          mirage::type::FPType *output_ptr =
              (mirage::type::FPType *)(smem_buffer + output_smem_offset);
          mirage::type::FPType *input_ptr =
              (mirage::type::FPType *)(smem_buffer + input_smem_offset);
          mirage::threadblock::TBReductionFingerprinter fp(
              new_params.operator_types[op],
              input_ptr,
              output_ptr,
              output_num_elements,
              reduction_degree,
              inner_range,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case mirage::type::TB_CONCAT_0_OP:
        case mirage::type::TB_CONCAT_1_OP:
        case mirage::type::TB_CONCAT_2_OP: {
          int output_num_elements, A_concat_dim_size, B_concat_dim_size,
              inner_size;
          int A_smem_offset, B_smem_offset, output_smem_offset;
          mirage::threadblock::deserialize_concat_op_parameters(
              new_params.parameters,
              param_idx,
              output_num_elements,
              A_concat_dim_size,
              B_concat_dim_size,
              inner_size,
              A_smem_offset,
              B_smem_offset,
              output_smem_offset);
          mirage::type::FPType *A_ptr =
              (mirage::type::FPType *)(smem_buffer + A_smem_offset);
          mirage::type::FPType *B_ptr =
              (mirage::type::FPType *)(smem_buffer + B_smem_offset);
          mirage::type::FPType *output_ptr =
              (mirage::type::FPType *)(smem_buffer + output_smem_offset);
          mirage::threadblock::TBConcatFingerprinter fp(A_ptr,
                                                     B_ptr,
                                                     output_ptr,
                                                     output_num_elements,
                                                     A_concat_dim_size,
                                                     B_concat_dim_size,
                                                     inner_size,
                                                     threadIdx.x,
                                                     blockDim.x);
          __syncthreads();
          break;
        }
        default: {
          assert(false && "Unsupported threadblock operator");
        }
      }
    }
    assert(new_params.num_parameters == param_idx);
  }
}

void KNCustomizedOp::run() {
  // mirage::threadblock::KernelParams params = bgraph.get_kernel_params();
  mirage::threadblock::NewKernelParams new_params =
      bgraph.get_new_kernel_params(false /*fingerprint_kernel*/);
  customized_kernel_function<<<bgraph.grid_dim,
                               bgraph.block_dim,
                               bgraph.smem_offset>>>(new_params,
                                                     bgraph.forloop_range);
}

bool KNCustomizedOp::profile(ProfileResult &result) {
  printf("smem_offset = %d\n", bgraph.smem_offset);
  int max_smem_size = mirage::type::MAX_SMEM_SIZE;
  assert(bgraph.smem_offset <= max_smem_size);
  if (bgraph.smem_offset > 48 * 1024) {
    checkCUDA(cudaFuncSetAttribute(customized_kernel_function,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  // mirage::threadblock::KernelParams params = bgraph.get_kernel_params();
  mirage::threadblock::NewKernelParams new_params =
      bgraph.get_new_kernel_params(false /*fingerprint_kernel*/);
  for (int i = 0; i < 1024; i++) {
    customized_kernel_function<<<bgraph.grid_dim,
                                 bgraph.block_dim,
                                 bgraph.smem_offset>>>(new_params,
                                                       bgraph.forloop_range);
  }
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < ProfileResult::NUM_ITERATIONS; i++) {
    customized_kernel_function<<<bgraph.grid_dim,
                                 bgraph.block_dim,
                                 bgraph.smem_offset>>>(new_params,
                                                       bgraph.forloop_range);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / ProfileResult::NUM_ITERATIONS;
  printf("KNCustomizedOp: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

bool KNCustomizedOp::fingerprint(void) {
  int max_smem_size = mirage::type::MAX_SMEM_SIZE;
  // mirage::threadblock::KernelParams params = bgraph.get_kernel_params();
  mirage::threadblock::NewKernelParams new_params =
      bgraph.get_new_kernel_params(true /*fingerprint_kernel*/);
  assert(bgraph.smem_offset <= max_smem_size);
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  if (bgraph.smem_offset > 48 * 1024) {
    checkCUDA(cudaFuncSetAttribute(compute_customizedop_fingerprint,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
  compute_customizedop_fingerprint<<<bgraph.grid_dim,
                                     bgraph.block_dim,
                                     bgraph.smem_offset>>>(
      new_params,
      bgraph.forloop_range,
      dmm->exp_lookup_table,
      dmm->div_p_lookup_table,
      dmm->div_q_lookup_table);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
