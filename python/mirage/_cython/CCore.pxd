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
from libcpp.vector cimport vector
from libcpp cimport bool

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
  
cdef extern from "mirage/layout.h" namespace "mirage::layout":
    # This must be consistent with mirage/layout.h
    cdef enum DmemLayout:
        DmemRowMajor = 100,
        DmemColumnMajor = 101,
        DmemUnknowLayout = 199,

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

    cdef cppclass Graph:
        Graph()
        DTensor* new_input_ptr(vector[int] dims,
                               DataType data_type,
                               DmemLayout layout)
        DTensor* matmul(const DTensor* A, const DTensor* B)
        DTensor* reduction(const DTensor* input, int dim, int size)
        DTensor* exp(const DTensor* input)
        DTensor* add(const DTensor* op1, const DTensor* op2)
        DTensor* mul(const DTensor* op1, const DTensor* op2)
        DTensor* div(const DTensor* op1, const DTensor* op2)
        void generate_triton_program(const char *filepath)

cdef extern from "mirage/search/search_c.h" namespace "mirage::search_c":
    ctypedef struct MInt3:
        int x
        int y
        int z
    ctypedef struct MDim3:
        unsigned int x
        unsigned int y
        unsigned int z
    cdef int cython_optimize(const Graph *input_graph,
                             int max_num_new_graphs,
                             Graph** new_graphs,
                             vector[MInt3] imaps,
                             vector[MInt3] omaps,
                             vector[MDim3] griddims,
                             vector[MDim3] blockdims,
                             vector[int] fmaps,
                             vector[int] franges,
                             const char * check_point_file_path,
                             const char * default_config)
