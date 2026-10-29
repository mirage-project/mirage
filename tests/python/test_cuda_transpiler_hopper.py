# Copyright 2026 CMU
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

import re

import pytest

import mirage as mi


def rms_norm_matmul_graph():
    # O = (X @ W) / rms(X): the matmul accumulator can live in registers, the
    # rms accumulator (square, scale, accumulate) lives in shared memory. Each
    # block loads a [64, 64] tile of W, so the threadblock graph stays well
    # under the CUDA build's 96 KiB limit on shared memory at construction
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(8, 64), strides=(64, 1), dtype=mi.bfloat16)
    W = graph.new_input(dims=(64, 256), strides=(1, 64), dtype=mi.bfloat16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(4, 1, 1), block_dim=(128, 1, 1), forloop_range=1, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=-1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=-1)
    tR = tb_graph.forloop_accum(tX, "rms")
    tM = tb_graph.forloop_accum(tb_graph.matmul(tX, tW))
    tO = tb_graph.div(tM, tR)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0], (256, 1))
    return graph


@pytest.mark.parametrize("num_warp_groups", [2, 3, 4])
@pytest.mark.parametrize("pipeline_stages", [2, 3, 4])
def test_hopper_writes_back_register_accumulators(num_warp_groups, pipeline_stages):
    result = mi.generate_cuda_program(
        rms_norm_matmul_graph().cygraph,
        target_cc=90,
        input_strides=[(64, 1), (1, 64)],
        num_warp_groups=num_warp_groups,
        pipeline_stages=pipeline_stages,
    )
    code = result["code"]
    in_reg = set(
        re.findall(r"auto matmul_(\d+)_accum = Matmul\1Kernel::get_mma_rC\(", code)
    )
    written_back = set(
        re.findall(
            r"Matmul(\d+)Kernel::write_back_mma_rC\(stensor\1_ptr, matmul_\1_accum,",
            code,
        )
    )
    # The matmul accumulator is held in registers and must reach shared memory
    # after the loop; nothing else is written back from registers
    assert len(in_reg) == 1
    assert written_back == in_reg
    for guid in re.findall(r"Matmul(\d+)Kernel::", code):
        assert f"using Matmul{guid}Kernel =" in code
