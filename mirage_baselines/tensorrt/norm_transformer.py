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
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024*1024*1024)

X_shape = [args.batch, 4096]
scale_shape = [1, 4096]
bias_shape = [1, 4096]
alpha_shape = [args.batch, 4096]

alpha = network.add_input("A", dtype=trt.float32, shape=alpha_shape)
scale = network.add_input("scale", dtype=trt.float32, shape=scale_shape)
bias = network.add_input("bias", dtype=trt.float32, shape=bias_shape)
X = network.add_input("X", dtype=trt.float32, shape=X_shape)

Xnorm = network.add_normalization(X, scale, bias, (1 << 0 | 1 << 1))
A = network.add_elementwise(alpha, Xnorm.get_output(0), trt.ElementWiseOperation.SUM)
B = network.add_elementwise(alpha, A.get_output(0), trt.ElementWiseOperation.PROD)
C = network.add_elementwise(X, B.get_output(0), trt.ElementWiseOperation.SUM)
O = network.add_normalization(C.get_output(0), scale, bias, (1 << 1 | 1 << 0))

network.mark_output(O.get_output(0))

plan = builder.build_serialized_network(network, config)
engine = runtime.deserialize_cuda_engine(plan)


inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
