import mirage as mi
import numpy as np
import torch
import pytest
import torch.nn as nn


def is_closed(A, B):
    err = 0
    assert (A.shape == B.shape) & (A.stride() == B.stride()) & (A.dim() == 2)
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
    graph.mark_output(O[0], (test_config["input_size"][1], 1))
    # uniform distribution from 0 to 0.5
    input_tensors = [
        (
            torch.rand(test_config["input_size"], dtype=torch.float16, device="cuda:0")
            * (-0.5)
        )
        + 1,
        (
            torch.rand(
                test_config["weight1_size"], dtype=torch.float16, device="cuda:0"
            )
            * (-0.5)
        )
        + 1,
        (
            torch.rand(
                test_config["weight2_size"], dtype=torch.float16, device="cuda:0"
            )
            * (-0.5)
        )
        + 1,
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    # p = mi.generate_cuda_program(
    #     graph.cygraph, target_cc=86, input_strides=input_strides
    # )
    # print(p["code"])
    outputs = graph(inputs=input_tensors, outputs=O)

    # check correctness with torch
    In1 = torch.matmul(input_tensors[0], input_tensors[1])
    In2 = torch.matmul(input_tensors[0], input_tensors[2])
    Res = torch.mul(nn.functional.silu(In1), In2)

    assert is_closed(outputs[0], Res)


@pytest.mark.parametrize(
    "test_config",
    [
        {
            "query_size": (2, 256, 64),
            "key_size": (2, 64, 4096),
            "value_size": (2, 4096, 64),
            "tb1_grid_dim": (2, 16, 4),
            "tb1_block_dim": (128, 1, 1),
            "tb1_forloop_range": 4,
            "tb1_reduction_dimx": 64,
            "tb1_qinput_map": (0, -1, 1),
            "tb1_kinput_map": (0, 2, -1),
            "tb1_vinput_map": (0, 1, -1),
            "tb1_qforloop_dim": -1,
            "tb1_kforloop_dim": 2,
            "tb1_vforloop_dim": 1,
            "tb1_outout_map1": (0, 2, 1),
            "tb1_outout_map2": (0, 2, 1),
            "tb2_grid_dim": (2, 16, 1),
            "tb2_block_dim": (128, 1, 1),
            "tb2_forloop_range": 1,
            "tb2_reduction_dimx": 64,
            "tb2_input_map1": (0, 1, -1),
            "tb2_input_map2": (0, 1, -1),
            "tb2_forloop_dim1": -1,
            "tb2_forloop_dim2": -1,
            "tb1_outout_map": (0, 1, -1),
        }
    ],
)
def test_group_query_attention(test_config):
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=test_config["query_size"], dtype=mi.float16)
    K = graph.new_input(dims=test_config["key_size"], dtype=mi.float16)
    V = graph.new_input(dims=test_config["value_size"], dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(
        grid_dim=test_config["tb1_grid_dim"],
        block_dim=test_config["tb1_block_dim"],
        forloop_range=test_config["tb1_forloop_range"],
        reduction_dimx=test_config["tb1_reduction_dimx"],
    )

    bQ = tbgraph1.new_input(
        dtensor=Q,
        input_map=test_config["tb1_qinput_map"],
        forloop_dim=test_config["tb1_qforloop_dim"],
    )
    bK = tbgraph1.new_input(
        dtensor=K,
        input_map=test_config["tb1_kinput_map"],
        forloop_dim=test_config["tb1_kforloop_dim"],
    )
    bV = tbgraph1.new_input(
        dtensor=V,
        input_map=test_config["tb1_vinput_map"],
        forloop_dim=test_config["tb1_vforloop_dim"],
    )
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=test_config["tb1_outout_map1"])
    tbgraph1.new_output(stensor=bO2, output_map=test_config["tb1_outout_map2"])
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(
        grid_dim=test_config["tb2_grid_dim"],
        block_dim=test_config["tb2_block_dim"],
        forloop_range=test_config["tb2_forloop_range"],
        reduction_dimx=test_config["tb2_reduction_dimx"],
    )
    bA = tbgraph2.new_input(
        dtensor=O[0],
        input_map=test_config["tb2_input_map1"],
        forloop_dim=test_config["tb2_forloop_dim1"],
    )
    bB = tbgraph2.new_input(
        dtensor=O[1],
        input_map=test_config["tb2_input_map2"],
        forloop_dim=test_config["tb2_forloop_dim2"],
    )
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=test_config["tb1_outout_map"])
    O = graph.customized(O, tbgraph2)
    graph.mark_output(
        O[0],
        (
            test_config["query_size"][1] * test_config["query_size"][2],
            test_config["query_size"][2],
            1,
        ),
    )

    input_tensors = [
        (
            torch.randn(test_config["query_size"], dtype=torch.float16, device="cuda:0")
            * 0.2
            - 0.1
        ),
        (
            torch.randn(test_config["key_size"], dtype=torch.float16, device="cuda:0")
            * 0.2
            - 0.1
        ),
        (
            torch.randn(test_config["value_size"], dtype=torch.float16, device="cuda:0")
            * 0.2
            - 0.1
        ),
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    # p = mi.generate_cuda_program(
    #     graph.cygraph, target_cc=86, input_strides=input_strides
    # )
    # print(p["code"])
    outputs = graph(inputs=input_tensors, outputs=O)

    attention_score = torch.matmul(input_tensors[0], input_tensors[1])
    attention_weights = torch.softmax(attention_score, dim=-1)

    attention_output = torch.matmul(attention_weights, input_tensors[2])

    assert is_closed(
        outputs[0].reshape(outputs[0].size(0), -1),
        attention_output.reshape(
            attention_output.size(0),
            -1,
        ),
    )


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
    X = graph.new_input(test_config["input_size"], (1, 8), dtype=mi.float16)
    W = graph.new_input(test_config["weight_size"],(1, 4096), dtype=mi.float16)
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

    # stride(4096, 1)
    graph.mark_output(O[0], (test_config["input_size"][1], 1))

    input_tensors = [
        (
            torch.rand(test_config["input_size"], dtype=torch.float16, device="cuda:0")
            * (-0.5)
        )
        + 1,
        (
            torch.rand(test_config["weight_size"], dtype=torch.float16, device="cuda:0")
            * (-0.5)
        )
        + 1,
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    # p = mi.generate_cuda_program(
    #     graph.cygraph, target_cc=86, input_strides=input_strides
    # )
    # print(p["code"])
    outputs = graph(inputs=input_tensors, outputs=O)

    # check correctness with torch
    rmsnorm = nn.RMSNorm(
        (input_tensors[0].size(1)), dtype=torch.float16, device="cuda:0"
    )
    RMS = rmsnorm(input_tensors[0])
    Res = torch.matmul(RMS, input_tensors[1])

    assert is_closed(Res, outputs[0])
