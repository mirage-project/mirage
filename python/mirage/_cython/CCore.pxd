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
    cdef enum TBOperatorType:
        TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
        TB_FORLOOP_ACCUM_RED_LD_SUM_OP = 2501,
        TB_FORLOOP_ACCUM_RED_LD_MEAN_OP = 2502,
        TB_FORLOOP_ACCUM_RED_LD_RMS_OP = 2503,
        TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP = 2504,

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

cdef extern from "mirage/kernel/graph.h" namespace "mirage::kernel":
    cdef cppclass KNOperator:
        pass
    ctypedef struct CppDTensor "mirage::kernel::DTensor":
        DataType data_type
        DmemLayout layout
        int num_dims
        int dim[4]
        size_t guid
        #KNOperator *owner_op
        #void *data_ptr
        int owner_ts_idx
        pass

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

cdef extern from "mirage/threadblock/graph.h" namespace "mirage::threadblock":
    cdef cppclass TBOperator:
        pass
    ctypedef struct CppSTensor "mirage::threadblock::STensor":
        DataType data_type
        SmemLayout layout
        int num_dims
        int dim[4]
        int owner_ts_id

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
                           bool verbose,
                           const char * default_config)

cdef extern from "mirage/transpiler/transpile.h" namespace "mirage::transpiler":
    ctypedef struct TranspilerConfig:
        int target_cc
    ctypedef struct OutputTensorDirective:
        size_t alloc_size
        vector[int] shape
        vector[size_t] strides
    ctypedef struct TranspileResult:
        string code
        size_t buf_size
        vector[OutputTensorDirective] output_directives
    cdef TranspileResult transpile(const CppKNGraph *graph,
                       const TranspilerConfig config,
                       vector[vector[size_t]] input_strides)
