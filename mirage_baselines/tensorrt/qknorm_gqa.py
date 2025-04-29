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
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 * 1024*1024*1024)

Q_shape = [args.batch, 2, 256, 64]
K_shape = [args.batch, 2, 64, 4096]
V_shape = [args.batch, 2, 4096, 64]
Q_scale_shape = [args.batch, 2, 256, 1]
V_scale_shape = [args.batch, 2, 4096, 1]


Q = network.add_input("Q", dtype=trt.float16, shape=Q_shape)
K = network.add_input("K", dtype=trt.float16, shape=K_shape)
V = network.add_input("V", dtype=trt.float16, shape=V_shape)
Qscale = network.add_input("Qscale", dtype=trt.float16, shape=Q_scale_shape)
Qbias = network.add_input("Qbias", dtype=trt.float16, shape=Q_scale_shape)
Vscale = network.add_input("Vscale", dtype=trt.float16, shape=V_scale_shape)
Vbias = network.add_input("Vbias", dtype=trt.float16, shape=V_scale_shape)


Q = network.add_normalization(Q, Qscale, Qbias, (1 << 3))
V = network.add_normalization(V, Vscale, Vbias, (1 << 3))
S = network.add_matrix_multiply(Q.get_output(0), trt.MatrixOperation.NONE, K, trt.MatrixOperation.NONE)
S = network.add_softmax(S.get_output(0))
S.axes = 1 << 3;
O = network.add_matrix_multiply(S.get_output(0), trt.MatrixOperation.NONE, V.get_output(0), trt.MatrixOperation.NONE)

network.mark_output(O.get_output(0))

plan = builder.build_serialized_network(network, config)
engine = runtime.deserialize_cuda_engine(plan)


inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
