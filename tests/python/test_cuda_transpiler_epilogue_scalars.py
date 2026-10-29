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


def exp_mul_scalar_graph():
    # After the loop, exp is followed by mul_scalar(0.5): the transpiler fuses
    # mul_scalar into exp's epilogue, so its scalar travels in the epilogue's
    # scalar array
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(64, 256), strides=(256, 1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1), forloop_range=1, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=-1)
    tA = tb_graph.forloop_accum(tX)
    tO = tb_graph.mul_scalar(tb_graph.exp(tA), 0.5)
    tb_graph.new_output(stensor=tO, output_map=(-1, -1, -1))
    O = graph.customized([X], tb_graph)
    graph.mark_output(O[0], (256, 1))
    return graph


@pytest.mark.parametrize("target_cc", [80, 90, 100])
def test_fused_unary_epilogue_scalar(target_cc):
    result = mi.generate_cuda_program(
        exp_mul_scalar_graph().cygraph,
        target_cc=target_cc,
        input_strides=[(256, 1)],
    )
    scalars = re.findall(r"const float scalars\[\] = \{([^}]*)\};", result["code"])
    # The epilogue chain exp -> mul_scalar -> store reads one scalar per
    # fused op and one for the store: 0.5 for mul_scalar, then 0 for the store
    assert any(
        [s.strip() for s in line.split(",")] == ["0.500000f", "0.0f"]
        for line in scalars
    )
