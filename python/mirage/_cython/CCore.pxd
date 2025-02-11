# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef unsigned long int size_t

cdef extern from "vector_types.h":
    ctypedef struct dim3:
        unsigned int x
        unsigned int y
        unsigned int z
    ctypedef struct int3:
        int x
        int y
        int z

cdef extern from "mirage/type.h" namespace "mirage::type":
    # This must be consistent with mirage/type.h
    cdef enum DataType:
        DT_INT8 = 900,
        DT_UINT16 = 910,
        DT_BFLOAT16 = 920,
        DT_FLOAT16 = 921,
        DT_FLOAT32 = 930,
        DT_DOUBLE = 940,
        DT_UNKNOWN = 999,
    cdef enum TBEpilogueType:
        TB_EPILOGUE_NONE = 3100,
        TB_EPILOGUE_ALLREDUCE = 3101,
        TB_EPILOGUE_ALLTOALL = 3102,
        TB_EPILOGUE_INVALID = 3199,
    cdef enum KNOperatorType:
        KN_UNKOWN = 1000,
        KN_INPUT_OP = 1001,
        KN_OUTPUT_OP = 1002,
        KN_MATMUL_OP = 1003,
        # ElementUnary
        KN_EXP_OP = 1100,
        KN_SQUARE_OP = 1101,
        KN_SQRT_OP = 1102,
        KN_SILU_OP = 1103,
        KN_GELU_OP = 1104,
        # ElementBinary
        KN_ADD_OP = 1200,
        KN_MUL_OP = 1201,
        KN_DIV_OP = 1202,
        # Reduction & Normalization
        KN_REDUCTION_0_OP = 1300,
        KN_REDUCTION_1_OP = 1301,
        KN_REDUCTION_2_OP = 1302,
        KN_RMS_NORM_OP = 1350,
        # Communication
        KN_ALLREDUCE_OP = 1400,
        KN_CUSTOMIZED_OP = 1999,
    cdef enum TBOperatorType:
        TB_UNKOWN = 2000,
        TB_INPUT_OP = 2001,
        TB_OUTPUT_OP = 2002,
        TB_MATMUL_OP = 2003,
        # ElementUnary
        TB_EXP_OP = 2100,
        TB_SQUARE_OP = 2101,
        TB_SQRT_OP = 2102,
        TB_SILU_OP = 2103,
        TB_GELU_OP = 2105,
        TB_MUL_SCALAR_OP = 2104,
        # ElementBinary
        TB_ADD_OP = 2200,
        TB_MUL_OP = 2201,
        TB_DIV_OP = 2202,
        # Reduction and Normalization
        TB_REDUCTION_FIRST_OP_ID = 2300,
        TB_REDUCTION_0_OP = 2301,
        TB_REDUCTION_1_OP = 2302,
        TB_REDUCTION_2_OP = 2303,
        TB_REDUCTION_0_TO_DIMX_OP = 2304,
        TB_REDUCTION_1_TO_DIMX_OP = 2305,
        TB_REDUCTION_2_TO_DIMX_OP = 2306,
        TB_REDUCTION_LAST_OP_ID = 2349,
        TB_RMS_NORM_OP = 2350,
        # Concat
        TB_CONCAT_FIRST_OP_ID = 2400,
        TB_CONCAT_0_OP = 2400,
        TB_CONCAT_1_OP = 2401,
        TB_CONCAT_2_OP = 2402,
        TB_CONCAT_LAST_OP_ID = 2410,
        TB_CONCAT_THEN_MATMUL_OP = 2411,
        # Forloop Accum
        # LD indicates last dimension
        TB_FORLOOP_ACCUM_FIRST_OP = 2500,
        TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
        TB_FORLOOP_ACCUM_RED_LD_SUM_OP = 2501,
        TB_FORLOOP_ACCUM_RED_LD_MEAN_OP = 2502,
        TB_FORLOOP_ACCUM_RED_LD_RMS_OP = 2503,
        TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP = 2504,
        TB_FORLOOP_ACCUM_LAST_OP = 2599,
        TB_CUSTOMIZED_OP = 2999

cdef extern from "mirage/layout.h" namespace "mirage::layout":
    # This must be consistent with mirage/layout.h
    cdef enum DmemLayout:
        DmemRowMajor = 100,
        DmemColumnMajor = 101,
        DmemUnknownLayout = 199,
    cdef enum SmemLayout:
        SmemRowMajor = 200,
        SmemColumnMajor = 201,
        SmemUnknownLayout = 299

cdef cppclass CppTBGraph "mirage::threadblock::Graph"

cdef extern from "mirage/kernel/device_tensor.h" namespace "mirage::kernel":
    cdef struct CppDTensor "mirage::kernel::DTensor":
        DataType data_type
        DmemLayout layout
        int num_dims
        int dim[4]
        size_t guid
        #KNOperator *owner_op
        #void *data_ptr
        int owner_ts_idx

cdef extern from "mirage/kernel/graph.h" namespace "mirage::kernel":

    cdef cppclass CppKNOperator "mirage::kernel::KNOperator":
        KNOperatorType op_type
        vector[CppDTensor] input_tensors
        vector[CppDTensor] output_tensors
        int get_input_dtensors(CppDTensor** cinputs)
        int get_output_dtensors(CppDTensor** cinputs)
 
    cdef cppclass CppKNCustomizedOp "mirage::kernel::KNCustomizedOp"(CppKNOperator):
        CppTBGraph bgraph
        void get_bgraph(CppTBGraph** bgraph)

    cdef cppclass CppKNGraph "mirage::kernel::Graph":
        CppKNGraph()
        CppDTensor* new_input_ptr(vector[int] dims,
                                  vector[size_t] strides,
                                  DataType data_type,
                                  DmemLayout layout)
        void mark_output(const CppDTensor* A, vector[size_t] strides)
        CppDTensor* matmul(const CppDTensor* A, const CppDTensor* B)
        CppDTensor* reduction(const CppDTensor* input, int dim, int size)
        CppDTensor* rms_norm(const CppDTensor* input, vector[int])
        CppDTensor* exp(const CppDTensor* input)
        CppDTensor* silu(const CppDTensor* input)
        CppDTensor* gelu(const CppDTensor* input)
        CppDTensor* add(const CppDTensor* op1, const CppDTensor* op2)
        CppDTensor* mul(const CppDTensor* op1, const CppDTensor* op2)
        CppDTensor* div(const CppDTensor* op1, const CppDTensor* op2)
        int customized(vector[const CppDTensor*] inputs,
                       CppDTensor** outputs,
                       CppTBGraph* bgraph)
        int get_input_dtensors(CppDTensor** cinputs)
        int get_input_dtensor_layout(const CppDTensor *input, int *strides)
        void generate_triton_program(const char *filepath)
        void generate_cuda_program(const char *filepath)
        vector[CppKNOperator*] operators

cdef extern from "mirage/threadblock/graph.h" namespace "mirage::threadblock":
    ctypedef struct CppSTensor "mirage::threadblock::STensor":
        DataType data_type
        SmemLayout layout
        int num_dims
        int dim[4]
        int owner_ts_idx
        size_t guid
    
    cdef cppclass CppTBOperator "mirage::threadblock::TBOperator":
        TBOperatorType op_type
        vector[CppSTensor] input_tensors
        vector[CppSTensor] output_tensors
        int get_input_stensors(CppSTensor** cinputs)
        int get_output_stensors(CppSTensor** cinputs)

    cdef cppclass CppTBInputOp "mirage::threadblock::TBInputOp"(CppTBOperator):
        int forloop_dim
        int3 input_map
        size_t get_dtensor_guid()

    cdef cppclass CppTBOutputOp "mirage::threadblock::TBOutputOp"(CppTBOperator):
        int forloop_dim
        int3 output_map
        size_t get_dtensor_guid()

    cdef cppclass CppTBGraph "mirage::threadblock::Graph":
        CppTBGraph(dim3 grid_dim,
                   dim3 block_dim,
                   int forloop_range,
                   int reduction_dimx)

        CppSTensor* new_input(const CppDTensor* dtensor,
                           int3 input_map,
                           int forloop_dim,
                           SmemLayout layout)
        CppDTensor* new_output(const CppSTensor* stensor,
                            int3 output_map,
                            int forloop_dim,
                            TBEpilogueType epilogue)
        CppSTensor* matmul(const CppSTensor *A,
                        const CppSTensor *B)
        CppSTensor* exp(const CppSTensor *A)
        CppSTensor* silu(const CppSTensor *A)
        CppSTensor* gelu(const CppSTensor *A)
        CppSTensor* square(const CppSTensor *A)
        CppSTensor* sqrt(const CppSTensor *A)
        CppSTensor* add(const CppSTensor *A,
                     const CppSTensor *B)
        CppSTensor* mul(const CppSTensor *A,
                     const CppSTensor *B)
        CppSTensor* div(const CppSTensor *A,
                     const CppSTensor *B)
        CppSTensor* reduction(const CppSTensor *A, int dim)
        CppSTensor* rms_norm(const CppSTensor *A)
        CppSTensor* concat(const CppSTensor *A,
                        const CppSTensor *B,
                        int dim)
        CppSTensor* forloop_accum(const CppSTensor *A,
                               TBOperatorType optype)
        dim3 grid_dim
        dim3 block_dim
        int forloop_range
        int reduction_dimx
        vector[CppTBOperator*] operators

cdef extern from "mirage/search/search_c.h" namespace "mirage::search_c":
    ctypedef struct MInt3:
        int x
        int y
        int z
    ctypedef struct MDim3:
        unsigned int x
        unsigned int y
        unsigned int z

    cdef int cython_search(const CppKNGraph *input_graph,
                           int max_num_new_graphs,
                           CppKNGraph** new_graphs,
                           vector[MInt3] imaps,
                           vector[MInt3] omaps,
                           vector[MDim3] griddims,
                           vector[MDim3] blockdims,
                           vector[int] fmaps,
                           vector[int] franges,
                           const char * filename,
                           bool verbose,
                           const char * default_config)

cdef extern from "mirage/transpiler/transpile.h" namespace "mirage::transpiler":
    ctypedef struct TranspilerConfig:
        int target_cc
        int num_consumer_wgs
        int num_producer_wgs;
        int pipeline_stages;
    ctypedef struct OutputTensorDirective:
        size_t alloc_size
        vector[int] shape
        vector[size_t] strides
    ctypedef struct TranspileResult:
        string code
        size_t buf_size
        size_t max_smem_size
        vector[OutputTensorDirective] output_directives
    cdef TranspileResult transpile(const CppKNGraph *graph,
                       const TranspilerConfig config,
                       vector[vector[size_t]] input_strides)

cdef extern from "mirage/nki_transpiler/transpile.h" namespace "mirage::nki_transpiler":
    ctypedef struct NKITranspilerConfig:
        int target_cc
    ctypedef struct NKIErrorInfo:
        vector[string] errors
    ctypedef struct NKITranspileResult:
        string code
        NKIErrorInfo error_state
    cdef NKITranspileResult transpile(const CppKNGraph *graph,
                                      const NKITranspilerConfig config)

cdef extern from "mirage/triton_transpiler/transpile.h" namespace "mirage::triton_transpiler":
    ctypedef struct TritonTranspilerConfig:
        int target_cc
    ctypedef struct TritonTranspileResult:
        string code
        vector[vector[int]] output_shapes
    cdef TritonTranspileResult transpile(const CppKNGraph *graph,
                                         const TritonTranspilerConfig config)