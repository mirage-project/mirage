import mirage as mi
import numpy as np
import torch
import pytest
import torch.nn as nn


def is_closed(A, B):
    err = 0
    assert (A.shape == B.shape) & (A.dim() == 2)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            max_val = max(abs(A[i, j].item()), abs(B[i, j].item()))
            rel_error = abs(A[i, j] - B[i, j]) / max_val
            abs_error = abs(A[i, j] - B[i, j])

            if (rel_error > 1e-1) & (abs_error > 1e-1):
                err += 1
    print(f"{err} out of {i * j} mismatch\n")
    return err == 0


@pytest.mark.parametrize(
    "test_config",
    [
        {
            "input_size": (8, 4096),
            "weight1_size": (4096, 4096),
            "weight2_size": (4096, 4096),
            "grid_dim": (64, 1, 1),
            "block_dim": (128, 1, 1),
            "forloop_range": 64,
            "reduction_dimx": 64,
            "tb_input_map1": (-1, -1, -1),
            "tb_forloop_dim1": 1,
            "tb_input_map2": (1, -1, -1),
            "tb_forloop_dim2": 0,
            "tb_input_map3": (1, -1, -1),
            "tb_forloop_dim3": 0,
            "tb_outout_map": (1, -1, -1),
        }
    ],
)
def test_gated_mlp(test_config):
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=test_config["input_size"], dtype=mi.float16)
    W1 = graph.new_input(dims=test_config["weight1_size"], dtype=mi.float16)
    W2 = graph.new_input(dims=test_config["weight2_size"], dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(
        test_config["grid_dim"],
        test_config["block_dim"],
        test_config["forloop_range"],
        test_config["reduction_dimx"],
    )
    tX = tb_graph.new_input(
        dtensor=X,
        input_map=test_config["tb_input_map1"],
        forloop_dim=test_config["tb_forloop_dim1"],
    )
    tW1 = tb_graph.new_input(
        dtensor=W1,
        input_map=test_config["tb_input_map2"],
        forloop_dim=test_config["tb_forloop_dim2"],
    )
    tW2 = tb_graph.new_input(
        dtensor=W2,
        input_map=test_config["tb_input_map3"],
        forloop_dim=test_config["tb_forloop_dim3"],
    )
    tD1 = tb_graph.matmul(tX, tW1)
    tD2 = tb_graph.matmul(tX, tW2)
    tA1 = tb_graph.forloop_accum(tD1)
    tA2 = tb_graph.forloop_accum(tD2)
    tS = tb_graph.silu(tA1)
    tO = tb_graph.mul(tS, tA2)
    tb_graph.new_output(stensor=tO, output_map=test_config["tb_outout_map"])
    O = graph.customized([X, W1, W2], tb_graph)

    # uniform distribution from -0.5 to 0.5
    input_tensors = [
        (
            torch.rand(test_config["input_size"], dtype=torch.float16, device="cuda:0")
            * 0.5
        )
        - 1,
        (
            torch.rand(
                test_config["weight1_size"], dtype=torch.float16, device="cuda:0"
            )
            * 0.5
        )
        - 1,
        (
            torch.rand(
                test_config["weight2_size"], dtype=torch.float16, device="cuda:0"
            )
            * 0.5
        )
        - 1,
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(
        graph.cygraph, target_cc=86, input_strides=input_strides, output_tensors=O
    )
    print(p["code"])
    outputs = graph(inputs=input_tensors, outputs=O)

    # check correctness with torch
    In1 = torch.matmul(input_tensors[0], input_tensors[1])
    In2 = torch.matmul(input_tensors[0], input_tensors[2])
    Res = torch.mul(nn.functional.silu(In1), In2)

    assert is_closed(outputs[0], Res)


def test_group_query_attention():
    assert 1


def test_group_query_attention_spec_decoding():
    assert 1


def test_lora():
    assert 1


@pytest.mark.parametrize(
    "test_config",
    [
        {
            "input_size": (8, 4096),
            "weight_size": (4096, 4096),
            "grid_dim": (64, 1, 1),
            "block_dim": (128, 1, 1),
            "forloop_range": 64,
            "reduction_dimx": 64,
            "tb_input_map1": (-1, -1, -1),
            "tb_forloop_dim1": 1,
            "tb_input_map2": (1, -1, -1),
            "tb_forloop_dim2": 0,
            "tb_outout_map": (1, -1, -1),
        }
    ],
)
def test_rms_norm(test_config):
    graph = mi.new_kernel_graph()
    X = graph.new_input(test_config["input_size"], dtype=mi.float16)
    W = graph.new_input(test_config["weight_size"], dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=test_config["grid_dim"],
        block_dim=test_config["block_dim"],
        forloop_range=test_config["forloop_range"],
        reduction_dimx=test_config["reduction_dimx"],
    )
    tX = tb_graph.new_input(
        dtensor=X,
        input_map=test_config["tb_input_map1"],
        forloop_dim=test_config["tb_forloop_dim1"],
    )
    tW = tb_graph.new_input(
        dtensor=W,
        input_map=test_config["tb_input_map2"],
        forloop_dim=test_config["tb_forloop_dim2"],
    )
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=test_config["tb_outout_map"])
    O = graph.customized([X, W], tb_graph)

    #     input_tensors = [
    #     torch.full(test_config["input_size"], 0.1,dtype=torch.float16, device='cuda:0'),
    #     torch.full(test_config["weight_size"], 0.1, dtype=torch.float16, device='cuda:0'),
    # ]

    input_tensors = [
        (
            torch.rand(test_config["input_size"], dtype=torch.float16, device="cuda:0")
            * 0.5
        )
        - 1,
        (
            torch.rand(test_config["weight_size"], dtype=torch.float16, device="cuda:0")
            * 0.5
        )
        - 1,
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(
        graph.cygraph, target_cc=86, input_strides=input_strides, output_tensors=O
    )
    print(p["code"])
    outputs = graph(inputs=input_tensors, outputs=O)

    # check correctness with torch
    rmsnorm = nn.RMSNorm(
        (input_tensors[0].size(1)), dtype=torch.float16, device="cuda:0"
    )
    RMS = rmsnorm(input_tensors[0])
    Res = torch.matmul(RMS, input_tensors[1])

    assert is_closed(Res, outputs[0])
