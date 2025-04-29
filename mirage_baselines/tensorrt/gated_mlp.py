import tensorrt as trt
import torch
import numpy as np
from common_runtime import *
from typing import Optional, List, Union
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
runtime = trt.Runtime(TRT_LOGGER)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10*1024*1024*1024)

A_shape = [4096, 4096]
B_shape = [4096, 4096]
X_shape = [1, 4096]

A = network.add_input("A", dtype=trt.float16, shape=A_shape)
B = network.add_input("B", dtype=trt.float16, shape=B_shape)
X = network.add_input("X", dtype=trt.float16, shape=X_shape)

C = network.add_matrix_multiply(X, trt.MatrixOperation.NONE, A, trt.MatrixOperation.NONE)
D = network.add_matrix_multiply(X, trt.MatrixOperation.NONE, B, trt.MatrixOperation.NONE)
O = network.add_elementwise(C.get_output(0), D.get_output(0), trt.ElementWiseOperation.SUM)

network.mark_output(O.get_output(0))

plan = builder.build_serialized_network(network, config)
engine = runtime.deserialize_cuda_engine(plan)


inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
