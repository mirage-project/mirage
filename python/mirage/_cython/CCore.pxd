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
    ctypedef struct DTensor:
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
        DTensor* new_input_ptr(vector[int] dims,
                               DataType data_type,
                               DmemLayout layout)
        DTensor* matmul(const DTensor* A, const DTensor* B)
        DTensor* reduction(const DTensor* input, int dim, int size)
        DTensor* exp(const DTensor* input)
        DTensor* add(const DTensor* op1, const DTensor* op2)
        DTensor* mul(const DTensor* op1, const DTensor* op2)
        DTensor* div(const DTensor* op1, const DTensor* op2)
        int customized(vector[const DTensor*] inputs,
                       DTensor** outputs,
                       CppTBGraph* bgraph)
        void generate_triton_program(const char *filepath)
        void generate_cuda_program(const char *filepath)

cdef extern from "mirage/threadblock/graph.h" namespace "mirage::threadblock":
    cdef cppclass TBOperator:
        pass
    ctypedef struct STensor:
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
        STensor* new_input(const DTensor* dtensor,
                           int3 input_map,
                           int forloop_dim,
                           SmemLayout layout)
        DTensor* new_output(const STensor* stensor,
                            int3 output_map,
                            int forloop_dim,
                            TBEpilogueType epilogue)
        STensor* matmul(const STensor *A,
                        const STensor *B)
        STensor* exp(const STensor *A)
        STensor* silu(const STensor *A)
        STensor* square(const STensor *A)
        STensor* sqrt(const STensor *A)
        STensor* add(const STensor *A,
                     const STensor *B)
        STensor* mul(const STensor *A,
                     const STensor *B)
        STensor* div(const STensor *A,
                     const STensor *B)
        STensor* reduction(const STensor *A, int dim)
        STensor* concat(const STensor *A,
                        const STensor *B,
                        int dim)
        STensor* forloop_accum(const STensor *A,
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

    cdef int cython_optimize(const CppKNGraph *input_graph,
                             int max_num_new_graphs,
                             CppKNGraph** new_graphs,
                             vector[MInt3] imaps,
                             vector[MInt3] omaps,
                             vector[MDim3] griddims,
                             vector[MDim3] blockdims,
                             vector[int] fmaps,
                             vector[int] franges,
                             const char * check_point_file_path,
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
                       vector[vector[size_t]] input_strides,
                       vector[const DTensor*] output_tensors)
