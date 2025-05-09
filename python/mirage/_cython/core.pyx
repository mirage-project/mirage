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
import torch
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

def get_kn_operator_type_string(int op_type):
    if op_type == KN_UNKOWN:
        return "kn_unknown"
    elif op_type == KN_INPUT_OP:
        return "kn_input_op"
    elif op_type == KN_OUTPUT_OP:
        return "kn_output_op"
    elif op_type == KN_MATMUL_OP:
        return "kn_matmul_op"
    elif op_type == KN_EXP_OP:
        return "kn_exp_op"
    elif op_type == KN_SQUARE_OP:
        return "kn_square_op"
    elif op_type == KN_SQRT_OP:
        return "kn_sqrt_op"
    elif op_type == KN_MUL_SCALAR_OP:
        return "kn_mul_scalar_op"
    elif op_type == KN_SILU_OP:
        return "kn_silu_op"
    elif op_type == KN_SIGMOID_OP:
        return "kn_sigmoid_op"
    elif op_type == KN_GELU_OP:
        return "kn_gelu_op"
    elif op_type == KN_RELU_OP:
        return "kn_relu_op"
    elif op_type == KN_CLAMP_OP:
        return "kn_clamp_op"
    elif op_type == KN_LOG_OP:
        return "kn_log_op"
    elif op_type == KN_ADD_OP:
        return "kn_add_op"
    elif op_type == KN_MUL_OP:
        return "kn_mul_op"
    elif op_type == KN_DIV_OP:
        return "kn_div_op"
    elif op_type == KN_POW_OP:
        return "kn_pow_op"
    elif op_type == KN_REDUCTION_0_OP:
        return "kn_reduction_0_op"
    elif op_type == KN_REDUCTION_1_OP:
        return "kn_reduction_1_op"
    elif op_type == KN_REDUCTION_2_OP:
        return "kn_reduction_2_op"
    elif op_type == KN_RMS_NORM_OP:
        return "kn_rms_norm_op"
    elif op_type == KN_CONCAT_FIRST_OP_ID:
        return "kn_concat_first_op_id"
    elif op_type == KN_CONCAT_0_OP:
        return "kn_concat_0_op"
    elif op_type == KN_CONCAT_1_OP:
        return "kn_concat_1_op"
    elif op_type == KN_CONCAT_2_OP:
        return "kn_concat_2_op"
    elif op_type == KN_CONCAT_LAST_OP_ID:
        return "kn_concat_last_op_id"
    elif op_type == KN_SPLIT_FIRST_OP_ID:
        return "kn_split_first_op_id"
    elif op_type == KN_SPLIT_0_OP:
        return "kn_split_0_op"
    elif op_type == KN_SPLIT_1_OP:
        return "kn_split_1_op"
    elif op_type == KN_SPLIT_2_OP:
        return "kn_split_2_op"
    elif op_type == KN_CHUNK_0_OP:
        return "kn_chunk_0_op"
    elif op_type == KN_CHUNK_1_OP:
        return "kn_chunk_1_op"
    elif op_type == KN_CHUNK_2_OP:
        return "kn_chunk_2_op"
    elif op_type == KN_SPLIT_LAST_OP_ID:
        return "kn_split_last_op_id"
    elif op_type == KN_ALLREDUCE_OP:
        return "kn_allreduce_op"
    elif op_type == KN_CUSTOMIZED_OP:
        return "kn_customized_op"
    else:
        return "unknown_op_type" + str(op_type)


def get_tb_operator_type_string(int op_type):
    if op_type == TB_UNKOWN:
        return "tb_unknown"
    elif op_type == TB_INPUT_OP:
        return "tb_input_op"
    elif op_type == TB_OUTPUT_OP:
        return "tb_output_op"
    elif op_type == TB_MATMUL_OP:
        return "tb_matmul_op"
    elif op_type == TB_EXP_OP:
        return "tb_exp_op"
    elif op_type == TB_SQUARE_OP:
        return "tb_square_op"
    elif op_type == TB_SQRT_OP:
        return "tb_sqrt_op"
    elif op_type == TB_MUL_SCALAR_OP:
        return "tb_mul_scalar_op"
    elif op_type == TB_SILU_OP:
        return "tb_silu_op"
    elif op_type == TB_SIGMOID_OP:
        return "tb_sigmoid_op"
    elif op_type == TB_GELU_OP:
        return "tb_gelu_op"
    elif op_type == TB_RELU_OP:
        return "tb_relu_op"
    elif op_type == TB_CLAMP_OP:
        return "tb_clamp_op"
    elif op_type == TB_LOG_OP:
        return "tb_log_op"
    elif op_type == TB_ADD_OP:
        return "tb_add_op"
    elif op_type == TB_MUL_OP:
        return "tb_mul_op"
    elif op_type == TB_DIV_OP:
        return "tb_div_op"
    elif op_type == TB_SUB_OP:
        return "tb_sub_op"
    elif op_type == TB_POW_OP:
        return "tb_pow_op"
    elif op_type == TB_REDUCTION_FIRST_OP_ID:
        return "tb_reduction_first_op_id"
    elif op_type == TB_REDUCTION_0_OP:
        return "tb_reduction_0_op"
    elif op_type == TB_REDUCTION_1_OP:
        return "tb_reduction_1_op"
    elif op_type == TB_REDUCTION_2_OP:
        return "tb_reduction_2_op"
    elif op_type == TB_REDUCTION_0_TO_DIMX_OP:
        return "tb_reduction_0_to_dimx_op"
    elif op_type == TB_REDUCTION_1_TO_DIMX_OP:
        return "tb_reduction_1_to_dimx_op"
    elif op_type == TB_REDUCTION_2_TO_DIMX_OP:
        return "tb_reduction_2_to_dimx_op"
    elif op_type == TB_REDUCTION_0_MAX_OP:
        return "tb_reduction_0_max_op"
    elif op_type == TB_REDUCTION_1_MAX_OP:
        return "tb_reduction_1_max_op"
    elif op_type == TB_REDUCTION_2_MAX_OP:
        return "tb_reduction_2_max_op"
    elif op_type == TB_REDUCTION_LAST_OP_ID:
        return "tb_reduction_last_op_id"
    elif op_type == TB_RMS_NORM_OP:
        return "tb_rms_norm_op"
    elif op_type == TB_CONCAT_FIRST_OP_ID:
        return "tb_concat_first_op_id"
    elif op_type == TB_CONCAT_0_OP:
        return "tb_concat_0_op"
    elif op_type == TB_CONCAT_1_OP:
        return "tb_concat_1_op"
    elif op_type == TB_CONCAT_2_OP:
        return "tb_concat_2_op"
    elif op_type == TB_CONCAT_LAST_OP_ID:
        return "tb_concat_last_op_id"
    elif op_type == TB_CONCAT_THEN_MATMUL_OP:
        return "tb_concat_then_matmul_op"
    elif op_type == TB_SPLIT_FIRST_OP_ID:
        return "tb_split_first_op_id"
    elif op_type == TB_SPLIT_0_OP:
        return "tb_split_0_op"
    elif op_type == TB_SPLIT_1_OP:
        return "tb_split_1_op"
    elif op_type == TB_SPLIT_2_OP:
        return "tb_split_2_op"
    elif op_type == TB_SPLIT_LAST_OP_ID:
        return "tb_split_last_op_id"
    elif op_type == TB_FORLOOP_ACCUM_NO_RED_OP:
        return "tb_forloop_accum_no_red_op"
    elif op_type == TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
        return "tb_forloop_accum_red_ld_sum_op"
    elif op_type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
        return "tb_forloop_accum_red_ld_mean_op"
    elif op_type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
        return "tb_forloop_accum_red_ld_rms_op"
    elif op_type == TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP:
        return "tb_forloop_accum_redtox_ld_sum_op"
    elif op_type == TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP:
        return "tb_forloop_accum_no_red_rescale_op"
    elif op_type == TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP:
        return "tb_forloop_accum_red_ld_sum_rescale_op"
    elif op_type == TB_FORLOOP_ACCUM_MAX_OP:
        return "tb_forloop_accum_max_op"
    elif op_type == TB_FORLOOP_ACCUM_LAST_OP:
        return "tb_forloop_accum_last_op"
    elif op_type == TB_CUSTOMIZED_OP:
        return "tb_customized_op"
    else:
        return "unknown_op_type" + str(op_type)


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

def convert_dtype_to_torch_type(type : dtype):
    if type.is_int8():
        return torch.int8
    elif type.is_uint16():
        return torch.uint16
    elif type.is_fp16():
        return torch.float16
    elif type.is_bf16():
        return torch.bfloat16
    elif type.is_fp32():
        return torch.float32
    elif type.is_fp64():
        return torch.float64
    else:
        assert False, "Unsupported dtype: {}".format(type)

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

def string_to_accum_rescale_optype(acc):
     if acc is None:
         return TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP
     elif acc == "sum":
         return TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP
     else:
         assert False, "Unsupported accum rescale optype"
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

    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid

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
                print("Error: tensor is None in num_dims property")
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
    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid
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

cdef class CyKNOperator:
    cdef CppKNOperator* c_ptr # Hold a CppKNOperator instance

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppKNOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppKNOperator*>(ptr)
    
    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.c_ptr.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(DTensor(ptr))
        return inputs

    def get_output_dtensors(self):
        cdef CppDTensor* coutputs[1024]
        num = self.c_ptr.get_output_dtensors(coutputs)
        outputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(DTensor(ptr))
        return outputs

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_kn_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op):
        self._set_operator(op)

cdef class CyKNCustomizedOp(CyKNOperator):
    cdef CppKNCustomizedOp* c_customized_ptr

    def __cinit__(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_customized_ptr = <CppKNCustomizedOp*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_customized_ptr = <CppKNCustomizedOp*>(ptr)

    def get_bgraph(self):
        cdef CppTBGraph* bgraph
        self.c_customized_ptr.get_bgraph(&bgraph)

        ptr = ctypes.cast(<unsigned long long>bgraph, ctypes.c_void_p)
        cybgraph = CyTBGraph(bgraph = ptr)
        return cybgraph

cdef class CyTBOperator:
    cdef CppTBOperator* c_ptr # Hold a CppTBOperator instance

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppTBOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppTBOperator*>(ptr)

    def get_input_stensors(self):
        cdef CppSTensor* cinputs[1024]
        num = self.c_ptr.get_input_stensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(STensor(ptr))
        return inputs

    def get_output_stensors(self):
        cdef CppSTensor* coutputs[1024]
        num = self.c_ptr.get_output_stensors(coutputs)
        outputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(STensor(ptr))
        return outputs

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_tb_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op):
        self._set_operator(op)

cdef class CyTBInputOp(CyTBOperator):
    cdef CppTBInputOp* c_input_ptr

    def __cinit__(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_input_ptr = <CppTBInputOp*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_input_ptr = <CppTBInputOp*>(ptr)

    property input_map:
        def __get__(self):
            if self.c_input_ptr == NULL:
                return None
            else:
                return {
                    "x": self.c_input_ptr.input_map.x,
                    "y": self.c_input_ptr.input_map.y,
                    "z": self.c_input_ptr.input_map.z
                }

    property forloop_dim:
        def __get__(self):
            if self.c_input_ptr == NULL:
                return None
            else:
                return self.c_input_ptr.forloop_dim

    property dtensor_guid:
        def __get__(self):
            if self.c_input_ptr == NULL:
                return None
            else:
                return self.c_input_ptr.get_dtensor_guid()

cdef class CyTBOutputOp(CyTBOperator):
    cdef CppTBOutputOp* c_output_ptr

    def __cinit__(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_output_ptr = <CppTBOutputOp*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_output_ptr = <CppTBOutputOp*>(ptr)

    property output_map:
        def __get__(self):
            if self.c_output_ptr == NULL:
                return None
            else:
                return {
                    "x": self.c_output_ptr.output_map.x,
                    "y": self.c_output_ptr.output_map.y,
                    "z": self.c_output_ptr.output_map.z
                }

    property forloop_dim:
        def __get__(self):
            if self.c_output_ptr == NULL:
                return None
            else:
                return self.c_output_ptr.forloop_dim

    property dtensor_guid:
        def __get__(self):
            if self.c_output_ptr == NULL:
                return None
            else:
                return self.c_output_ptr.get_dtensor_guid()

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
    
    def gelu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.gelu(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)
    
    def relu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.relu(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)
    
    def clamp(self, DTensor input, float min_val, float max_val):
        cdef CppDTensor* ptr = self.p_kgraph.clamp(input.c_ptr, min_val, max_val)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def sqrt(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.sqrt(input.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def square(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.square(input.c_ptr)
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

    def pow(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.pow(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def customized(self, list inputs, CyTBGraph bgraph):
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
    
    def get_owner_independent_hash(self):
        return self.p_kgraph.get_owner_independent_hash()

    # visualizer utils

    def _kn_tensor_to_dict(self, DTensor t):
        return {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }

    def _tb_tensor_to_dict(self, STensor t):
        return {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }

    def _get_tb_operator_info(self, CyTBOperator op):
        ans = {
            "op_type": op.op_type,
            "input_tensors": [self._tb_tensor_to_dict(t) for t in op.get_input_stensors()],
            "output_tensors": [self._tb_tensor_to_dict(t) for t in op.get_output_stensors()],
        }
        if "input" in op.op_type:
            input_op = CyTBInputOp(ctypes.cast(<unsigned long long>(op.c_ptr), ctypes.c_void_p))
            ans["input_map"] = input_op.input_map
            ans["forloop_dim"] = input_op.forloop_dim
            ans["dtensor"] = {
                "guid": input_op.dtensor_guid
            }
        elif "output" in op.op_type:
            output_op = CyTBOutputOp(ctypes.cast(<unsigned long long>(op.c_ptr), ctypes.c_void_p))
            ans["output_map"] = output_op.output_map
            ans["forloop_dim"] = output_op.forloop_dim
            ans["dtensor"] = {
                "guid": output_op.dtensor_guid
            }
        return ans

    def _get_bgraph_info(self, CyKNOperator op):
        cop = CyKNCustomizedOp(ctypes.cast(<unsigned long long>(op.c_ptr), ctypes.c_void_p))
        bgraph = cop.get_bgraph()
        return {
            "grid_dim": bgraph.grid_dim,
            "forloop_range": bgraph.forloop_range,
            "operators": [self._get_tb_operator_info(i) for i in bgraph.operators]
        }

    def _get_kn_operator_info(self, CyKNOperator op):
        if op.op_type == "kn_customized_op":
            return {
                "op_type": op.op_type,
                "input_tensors": [self._kn_tensor_to_dict(t) for t in op.get_input_dtensors()],
                "output_tensors": [self._kn_tensor_to_dict(t) for t in op.get_output_dtensors()],
                "bgraph": self._get_bgraph_info(op)
            }
        else:
            return {
                "op_type": op.op_type,
                "input_tensors": [self._kn_tensor_to_dict(t) for t in op.get_input_dtensors()],
                "output_tensors": [self._kn_tensor_to_dict(t) for t in op.get_output_dtensors()],
            }

    def get_graph_structure(self):
        operators = []
        ops = self.p_kgraph.operators
        for i in range(ops.size()):
            op = CyKNOperator(None)
            op.c_ptr = ops[i]
            operators.append(self._get_kn_operator_info(op))
        return operators

    def get_num_inputs(self):
        return self.p_kgraph.get_num_input_dtensors()

    def get_num_outputs(self):
        return self.p_kgraph.get_num_output_dtensors()

    def get_input_dtensor_shape_and_stride(self, DTensor A):
        cdef int cstrides[128]
        cdef int cdims[128]
        num = self.p_kgraph.get_input_dtensor_shape_and_stride(A.c_ptr, cstrides, cdims)
        strides = list()
        dims = list()
        for i in range(num):
            strides.append(cstrides[i])
            dims.append(cdims[i])
        return tuple(dims), tuple(strides)

cdef class CyTBGraph:
    cdef CppTBGraph *p_bgraph #Hold a CppTBGraph instance

    def __cinit__(self, tuple grid_dim = (), tuple block_dim = (), int forloop_range = -1, int dimx = -1, bgraph = None):
        cdef unsigned long long ptr
        cdef dim3 c_grid_dim
        cdef dim3 c_block_dim
        if bgraph is None:
            if len(grid_dim) == 0 or len(block_dim) == 0 or forloop_range == -1 or dimx == -1:
                assert False, "grid_dim, block_dim, forloop_range, dimx must be provided"
            assert len(grid_dim) == 3, "grid_dim must include 3 dimensions"
            assert len(block_dim) == 3, "block_dim must include 3 dimensions"
            c_grid_dim.x = grid_dim[0]
            c_grid_dim.y = grid_dim[1]
            c_grid_dim.z = grid_dim[2]
            c_block_dim.x = block_dim[0]
            c_block_dim.y = block_dim[1]
            c_block_dim.z = block_dim[2]
            self.p_bgraph = new CppTBGraph(c_grid_dim, c_block_dim, forloop_range, dimx)
        else:
            ptr = ctypes.cast(bgraph, ctypes.c_void_p).value
            if isinstance(bgraph, int):
                self.p_bgraph = <CppTBGraph*>(ptr)
            elif isinstance(bgraph, ctypes.c_void_p):
                self.p_bgraph = <CppTBGraph*>(ptr)
            else:
                assert False, "bgraph must be an integer or ctypes.c_void_p, but got " + str(type(bgraph))
    
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
    
    def gelu(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.gelu(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)
    
    def relu(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.relu(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)
    
    def clamp(self, STensor A, float min_val, float max_val):
        cdef CppSTensor* ptr = self.p_bgraph.clamp(A.c_ptr, min_val, max_val)
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

    def mul_scalar(self, STensor A, float scalar):
        cdef CppSTensor* ptr = self.p_bgraph.mul_scalar(A.c_ptr, scalar)
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

    def sub(self, STensor A, STensor B):
        cdef CppSTensor* ptr = self.p_bgraph.sub(A.c_ptr, B.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def reduction(self, STensor A, int dim):
        cdef CppSTensor* ptr = self.p_bgraph.reduction(A.c_ptr, dim)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def reduction_max(self, STensor A, int dim):
        cdef vector[CppSTensor*] ptr = self.p_bgraph.reduction_max(A.c_ptr, dim)
        t0 = ctypes.cast(<unsigned long long>ptr[0], ctypes.c_void_p)
        t1 = ctypes.cast(<unsigned long long>ptr[1], ctypes.c_void_p)
        return STensor(t0), STensor(t1)

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

    def forloop_accum_rescale(self, STensor A, STensor B, str acc):
        optype = string_to_accum_rescale_optype(acc)
        cdef CppSTensor* ptr = self.p_bgraph.forloop_accum_rescale(A.c_ptr, B.c_ptr, optype)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    def forloop_accum_max(self, STensor A):
        cdef CppSTensor* ptr = self.p_bgraph.forloop_accum_max(A.c_ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    property grid_dim:
        def __get__(self):
            return {
                "x": self.p_bgraph.grid_dim.x,
                "y": self.p_bgraph.grid_dim.y,
                "z": self.p_bgraph.grid_dim.z
            }

    property forloop_range:
        def __get__(self):
            return self.p_bgraph.forloop_range

    property operators:
        def __get__(self):
            cdef vector[CppTBOperator*] coperators
            coperators = self.p_bgraph.operators
            operators = list()
            for i in range(coperators.size()):
                ptr = ctypes.cast(<unsigned long long>coperators[i], ctypes.c_void_p)
                operators.append(CyTBOperator(ptr))
            return operators

def search(CyKNGraph input_graph, *, int max_num_new_graphs = 1024, list imaps = None, list omaps = None, list griddims = None, list blockdims = None, list fmaps = None, list franges = None, str previous_checkpoint = None, bool verbose, str default_config = None, bool is_formal_verified):
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
    # set previous_checkpoint
    cdef char* cprevious_checkpoint = NULL
    if previous_checkpoint is not None:
        py_byte_string = previous_checkpoint.encode('UTF-8')
        cprevious_checkpoint = py_byte_string
    # convert config description
    cdef char* cconfig = NULL
    if default_config is not None:
        py_byte_string = default_config.encode('UTF-8')
        cconfig = py_byte_string
    # set is_formal_verified
    cis_formal_verifed = is_formal_verified
    num = cython_search(input_graph.p_kgraph, max_num_new_graphs, cnewgraphs, cimaps, comaps, cgriddims, cblockdims, cfmaps, cfranges, cprevious_checkpoint, cverbose, cconfig, cis_formal_verifed)
    new_graphs = list()
    for i in range(num):
        ptr = ctypes.cast(<unsigned long long>cnewgraphs[i], ctypes.c_void_p)
        new_graphs.append(CyKNGraph(ptr))

    return new_graphs

# Generate CUDA program for a uGraph
# Return (CUDA code, buffer size in bytes)
def generate_cuda_program(CyKNGraph input_graph, *, int target_cc, list input_strides, int num_warp_groups = -1, int pipeline_stages = -1, bool profiling = False, bool enable_online_softmax = False) -> dict:
    # Set transpiler_config
    cdef TranspilerConfig transpiler_config
    transpiler_config.target_cc = target_cc
    transpiler_config.profiling = profiling
    transpiler_config.enable_online_softmax = enable_online_softmax

    if num_warp_groups != -1 and pipeline_stages != -1:
        transpiler_config.num_producer_wgs = 1;
        transpiler_config.num_consumer_wgs = num_warp_groups - 1;
        transpiler_config.pipeline_stages = pipeline_stages;
    
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
        "max_smem_size": result.max_smem_size,
        "profiler_buf_size": result.profiler_buf_size,
        "output_directives": output_directives
    }

def generate_nki_program(CyKNGraph input_graph, *, int target_cc) -> dict:
    # Set transpiler_config
    cdef NKITranspilerConfig transpiler_config
    transpiler_config.target_cc = target_cc
    
    # Call transpile
    cdef NKITranspileResult result = transpile(input_graph.p_kgraph, transpiler_config)
    cdef list error_list = [error.decode("UTF-8") for error in result.error_state.errors]

    return {
        "code": result.code.decode("UTF-8"),
        "errors": error_list,
    }

def generate_triton_program(CyKNGraph input_graph, *, int target_cc) -> dict:
    cdef TritonTranspilerConfig transpiler_config
    transpiler_config.target_cc = target_cc

    cdef TritonTranspileResult result = transpile(input_graph.p_kgraph, transpiler_config)

    return {
        "code": result.code.decode("UTF-8"),
        "output_shapes": result.output_shapes
    }

def set_gpu_device_id(gpu_id: int):
    cython_set_gpu_device_id(gpu_id)

def cy_to_json(CyKNGraph input_graph, str filename):
    cfilename = filename.encode('UTF-8')
    cython_to_json(input_graph.p_kgraph, cfilename)

def cy_from_json(str filename):
    cfilename = filename.encode('UTF-8')
    ptr = cython_from_json(cfilename)
    graph = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
    return CyKNGraph(graph)