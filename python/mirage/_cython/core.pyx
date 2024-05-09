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

# Code snippet from OpenAi Triton

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

cdef class PyTensor:
    cdef DTensor* c_ptr # Hold a Tensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <DTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <DTensor*>(ptr)

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

cdef class PyGraph:
    cdef Graph *p_graph #Hold a Graph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        if graph is None:
            self.p_graph = new Graph()
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_graph = <Graph*>(ptr)

    def new_input(self, tuple dims, dtype : dtype = float16):
        cdef vector[int] cdims
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        c_type = convert_dtype_to_ctype(dtype)
        cdef DTensor* ptr = self.p_graph.new_input_ptr(cdims, c_type, DmemRowMajor)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def matmul(self, PyTensor A, PyTensor B):
        cdef DTensor* ptr = self.p_graph.matmul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def reduction(self, PyTensor input, int dim):
        cdef DTensor* ptr = self.p_graph.reduction(input.c_ptr, dim, 1)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def exp(self, PyTensor input):
        cdef DTensor* ptr = self.p_graph.exp(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def add(self, PyTensor A, PyTensor B):
        cdef DTensor* ptr = self.p_graph.add(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def mul(self, PyTensor A, PyTensor B):
        cdef DTensor* ptr = self.p_graph.mul(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def div(self, PyTensor A, PyTensor B):
        cdef DTensor* ptr = self.p_graph.div(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return PyTensor(t)

    def generate_triton_program(self, str filepath):
        assert filepath is not None, "filepath cannot be empty"
        py_byte_string = filepath.encode('UTF-8')
        cdef char* cfilepath = NULL
        cfilepath = py_byte_string
        self.p_graph.generate_triton_program(cfilepath)

def optimize(PyGraph input_graph, *, list imaps = None, list omaps = None, list griddims = None, list blockdims = None, list fmaps = None, list franges = None, str previous_checkpoint = None, str default_config = None):
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
    cdef Graph* cnewgraphs[1024]
    # convert file path
    cdef char* cfilepath = NULL
    if previous_checkpoint is not None:
        py_byte_string = previous_checkpoint.encode('UTF-8')
        cfilepath = py_byte_string
    # convert config description
    cdef char* cconfig = NULL
    if default_config is not None:
        py_byte_string = default_config.encode('UTF-8')
        cconfig = py_byte_string
    num = cython_optimize(input_graph.p_graph, 1024, cnewgraphs, cimaps, comaps, cgriddims, cblockdims, cfmaps, cfranges, cfilepath, cconfig)
    new_graphs = list()
    for i in range(num):
        ptr = ctypes.cast(<unsigned long long>cnewgraphs[i], ctypes.c_void_p)
        new_graphs.append(PyGraph(ptr))
    return new_graphs
