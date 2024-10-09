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

from CCore cimport *
from cpython cimport array
import ctypes
import array
import numpy as np
from libcpp.string cimport string

# Code snippet from OpenAI Triton

class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES, name

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, ))

    def __str__(self):
        return self.name

    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES

# data types
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')

def convert_dtype_to_ctype(type : dtype):
    if type.is_int8():
        return DT_INT8
    elif type.is_uint16():
        return DT_UINT16
    elif type.is_fp16():
        return DT_FLOAT16
    elif type.is_bf16():
        return DT_BFLOAT16
    elif type.is_fp32():
        return DT_FLOAT32
    elif type.is_fp64():
        return DT_DOUBLE
    else:
        return DT_UNKNOWN

def convert_ctype_to_dtype(type):
    if type == DT_INT8:
        return int8
    elif type == DT_UINT16:
        return uint16
    elif type == DT_FLOAT16:
        return float16
    elif type == DT_BFLOAT16:
        return bfloat16
    elif type == DT_FLOAT32:
        return float32
    elif type == DT_DOUBLE:
        return float64
    else:
        return None

def string_to_tbepilogue(epilogue):
    if epilogue is None:
        return TB_EPILOGUE_NONE
    elif epilogue == "allreduce":
        return TB_EPILOGUE_ALLREDUCE
    else:
        assert False, "Unsupported threadblock epilogue"
        return None

def string_to_accum_optype(acc):
    if acc is None:
        return TB_FORLOOP_ACCUM_NO_RED_OP
    elif acc == "sum":
        return TB_FORLOOP_ACCUM_RED_LD_SUM_OP
    elif acc == "mean":
        return TB_FORLOOP_ACCUM_RED_LD_MEAN_OP
    elif acc == "rms":
        return TB_FORLOOP_ACCUM_RED_LD_RMS_OP
    elif acc == "sum_todimx":
        return TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP
    else:
        assert False, "Unsupported accum optype"
        return None

cdef class DTensor:
    cdef CppDTensor* c_ptr # Hold a Tensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppDTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppDTensor*>(ptr)

    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return convert_ctype_to_dtype(self.c_ptr.data_type)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None

cdef class STensor:
    cdef CppSTensor* c_ptr # Hold a CppSTensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppSTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppSTensor*>(ptr)

    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return convert_ctype_to_dtype(self.c_ptr.data_type)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None


cdef class CyKNGraph:
    cdef CppKNGraph *p_kgraph #Hold a CppKNGraph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        if graph is None:
            self.p_kgraph = new CppKNGraph()
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_kgraph = <CppKNGraph*>(ptr)

    def new_input(self, tuple dims, tuple strides, dtype : dtype = float16):
        cdef vector[int] cdims
        cdef vector[size_t] cstrides
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        cstrides.resize(len(strides))
        for i in range(len(strides)):
            cstrides[i] = strides[i]

        c_type = convert_dtype_to_ctype(dtype)
        cdef CppDTensor* ptr = self.p_kgraph.new_input_ptr(cdims, cstrides, c_type, DmemRowMajor)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def mark_output(self, DTensor A, tuple strides):
        cdef vector[size_t] cstrides
        if strides is None:
            cstrides.resize(0)
        else:
            cstrides.resize(len(strides))
            for i in range(len(strides)):
                cstrides[i] = strides[i]
        self.p_kgraph.mark_output(A.c_ptr, cstrides)

    def matmul(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.matmul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def reduction(self, DTensor input, int dim):
        cdef CppDTensor* ptr = self.p_kgraph.reduction(input.c_ptr, dim, 1)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def rms_norm(self, DTensor input, tuple normalized_shape):
        cdef vector[int] cshape
        cshape.resize(len(normalized_shape))
        for i in range(len(normalized_shape)):
            cshape[i] = normalized_shape[i]
        cdef CppDTensor* ptr = self.p_kgraph.rms_norm(input.c_ptr, cshape)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def exp(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.exp(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def silu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.silu(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def add(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.add(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def mul(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.mul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def div(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.div(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def customized(self, list[DTensor] inputs, CyTBGraph bgraph):
        cdef vector[const CppDTensor*] cinputs
        cinputs.resize(len(inputs))
        cdef DTensor t
        for i in range(len(inputs)):
            assert(type(inputs[i]) == DTensor)
            t = inputs[i]
            cinputs[i] = t.c_ptr
        cdef CppDTensor* coutputs[1024]
        num_outputs = self.p_kgraph.customized(cinputs, coutputs, bgraph.p_bgraph)
        outputs = list()
        for i in range(num_outputs):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(DTensor(ptr))
        return outputs

    def generate_triton_program(self, str filepath):
        assert filepath is not None, "filepath cannot be empty"
        py_byte_string = filepath.encode('UTF-8')
        cdef char* cfilepath = NULL
        cfilepath = py_byte_string
        self.p_kgraph.generate_triton_program(cfilepath)

    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.p_kgraph.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(DTensor(ptr))
        return inputs

    def get_input_dtensor_layout(self, DTensor A):
        cdef int cstrides[128]
        num = self.p_kgraph.get_input_dtensor_layout(A.c_ptr, cstrides)
        strides = list()
        for i in range(num):
            strides.append(cstrides[i])
        return tuple(strides)

cdef class CyTBGraph:
    cdef CppTBGraph *p_bgraph #Hold a CppTBGraph instance

    def __cinit__(self, tuple grid_dim, tuple block_dim, int forloop_range, int dimx):
        assert len(grid_dim) == 3, "grid_dim must include 3 dimensions"
        assert len(block_dim) == 3, "block_dim must include 3 dimensions"
        cdef dim3 c_grid_dim
        c_grid_dim.x = grid_dim[0]
        c_grid_dim.y = grid_dim[1]
        c_grid_dim.z = grid_dim[2]
        cdef dim3 c_block_dim
        c_block_dim.x = block_dim[0]
        c_block_dim.y = block_dim[1]
        c_block_dim.z = block_dim[2]
        self.p_bgraph = new CppTBGraph(c_grid_dim, c_block_dim, forloop_range, dimx)
    
    def new_input(self, DTensor dtensor, tuple input_map, int forloop_dim):
        assert len(input_map) == 3, "input_map must be of length 3"
        cdef int3 c_input_map
        c_input_map.x = input_map[0]
        c_input_map.y = input_map[1]
        c_input_map.z = input_map[2]
        cdef CppSTensor* ptr = self.p_bgraph.new_input(dtensor.c_ptr, c_input_map, forloop_dim, SmemRowMajor)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def new_output(self, STensor stensor, tuple output_map, int forloop_dim, str epilogue = None):
        assert len(output_map) == 3, "output_map must be of length 3"
        cdef int3 c_output_map
        c_output_map.x = output_map[0]
        c_output_map.y = output_map[1]
        c_output_map.z = output_map[2]
        epilogue_type = string_to_tbepilogue(epilogue)
        self.p_bgraph.new_output(stensor.c_ptr, c_output_map, forloop_dim, epilogue_type)  

    def matmul(self, STensor A, STensor B):
        cdef CppSTensor* ptr = self.p_bgraph.matmul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def exp(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.exp(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def silu(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.silu(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def square(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.square(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def sqrt(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.sqrt(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def add(self, STensor A, STensor B):
        cdef CppSTensor* ptr = self.p_bgraph.add(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def mul(self, STensor A, STensor B):
        cdef CppSTensor* ptr = self.p_bgraph.mul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def div(self, STensor A, STensor B):
        cdef CppSTensor* ptr = self.p_bgraph.div(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def reduction(self, STensor A, int dim):
        cdef CppSTensor* ptr = self.p_bgraph.reduction(A.c_ptr, dim)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def rms_norm(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.rms_norm(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def concat(self, STensor A, STensor B, int dim):
        cdef CppSTensor* ptr = self.p_bgraph.concat(A.c_ptr, B.c_ptr, dim)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def forloop_accum(self, STensor A, str acc):
        optype = string_to_accum_optype(acc)
        cdef CppSTensor* ptr = self.p_bgraph.forloop_accum(A.c_ptr, optype)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

def search(CyKNGraph input_graph, *, int max_num_new_graphs = 1024, list imaps = None, list omaps = None, list griddims = None, list blockdims = None, list fmaps = None, list franges = None, str previous_checkpoint = None, bool verbose, str default_config = None):
    # set cimaps
    cdef vector[MInt3] cimaps
    cimaps.resize(0)
    if imaps is not None:
        cimaps.resize(len(imaps))
        for i in range(len(imaps)):
            assert type(imaps[i]) is tuple, "Each imap must be a tuple of 3 integers"
            assert len(imaps[i]) == 3, "Each imap must be a tuple of 3 integers"
            cimaps[i].x = imaps[i][0]
            cimaps[i].y = imaps[i][1]
            cimaps[i].z = imaps[i][2]
    #set comaps
    cdef vector[MInt3] comaps
    comaps.resize(0)
    if omaps is not None:
        comaps.resize(len(omaps))
        for i in range(len(omaps)):
            assert type(omaps[i]) is tuple, "Each omap must be a tuple of 3 integers"
            assert len(omaps[i]) == 3, "Each omap must be a tuple of 3 integers"
            comaps[i].x = omaps[i][0]
            comaps[i].y = omaps[i][1]
            comaps[i].z = omaps[i][2]
    # set griddims
    cdef vector[MDim3] cgriddims
    cgriddims.resize(0)
    if griddims is not None:
        cgriddims.resize(len(griddims))
        for i in range(len(griddims)):
            assert type(griddims[i]) is tuple, "Each griddim must be a tuple of 3 integers"
            assert len(griddims[i]) == 3, "Each griddim must be a tuple of 3 integers"
            cgriddims[i].x = griddims[i][0]
            cgriddims[i].y = griddims[i][1]
            cgriddims[i].z = griddims[i][2]
    # set blockdims
    assert blockdims is None, "TODO: support blockdims"
    cdef vector[MDim3] cblockdims
    cblockdims.resize(0)
    # set fmaps
    cdef vector[int] cfmaps
    cfmaps.resize(0)
    if fmaps is not None:
        cfmaps.resize(len(fmaps))
        for i in range(len(fmaps)):
            cfmaps[i] = fmaps[i]
    #set franges
    cdef vector[int] cfranges
    cfranges.resize(0)
    if franges is not None:
        cfranges.resize(len(franges))
        for i in range(len(franges)):
            cfranges[i] = franges[i]
    # allocate new graphs
    # currently support up to 1024 new graphs
    assert max_num_new_graphs <= 1024
    cdef CppKNGraph* cnewgraphs[1024]
    # set verbose
    cverbose = verbose
    # convert config description
    cdef char* cconfig = NULL
    if default_config is not None:
        py_byte_string = default_config.encode('UTF-8')
        cconfig = py_byte_string
    num = cython_search(input_graph.p_kgraph, max_num_new_graphs, cnewgraphs, cimaps, comaps, cgriddims, cblockdims, cfmaps, cfranges, cverbose, cconfig)
    new_graphs = list()
    for i in range(num):
        ptr = ctypes.cast(<unsigned long long>cnewgraphs[i], ctypes.c_void_p)
        new_graphs.append(CyKNGraph(ptr))

    return new_graphs

# Generate CUDA program for a uGraph
# Return (CUDA code, buffer size in bytes)
def generate_cuda_program(CyKNGraph input_graph, *, int target_cc, list input_strides) -> dict:
    # Set transpiler_config
    cdef TranspilerConfig transpiler_config
    transpiler_config.target_cc = target_cc
    
    # Set input_strides
    cdef vector[vector[size_t]] cinput_strides
    cinput_strides.resize(len(input_strides))
    for i in range(len(input_strides)):
        cinput_strides[i].resize(len(input_strides[i]))
        for j in range(len(input_strides[i])):
            cinput_strides[i][j] = input_strides[i][j]
    
    # Call transpile
    cdef TranspileResult result = transpile(input_graph.p_kgraph, transpiler_config, cinput_strides)

    # Get output directives
    cdef list[dict] output_directives = list()
    # cdef list[int] cur_output_shape
    # cdef list[int] cur_output_strides
    for i in range(len(result.output_directives)):
        cur_output_shape = list()
        cur_output_strides = list()
        num_dims = len(result.output_directives[i].shape)
        for j in range(num_dims):
            cur_output_shape.append(result.output_directives[i].shape[j])
            cur_output_strides.append(result.output_directives[i].strides[j])
        output_directives.append({
            "alloc_size": result.output_directives[i].alloc_size,
            "shape": cur_output_shape,
            "strides": cur_output_strides
        })

    return {
        "code": result.code.decode("UTF-8"),
        "buf_size": result.buf_size,
        "output_directives": output_directives
    }
