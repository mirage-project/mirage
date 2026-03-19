import mirage as mi


def _build_simple_pallas_graph():
    graph = mi.new_kernel_graph()
    A = graph.new_input(dims=(8, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, 8), dtype=mi.float16)

    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=1,
        reduction_dimx=8,
    )
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=-1)
    tB = tb_graph.new_input(dtensor=B, input_map=(-1, -1, -1), forloop_dim=-1)
    tC = tb_graph.matmul(tA, tB)
    tAcc = tb_graph.forloop_accum(tC)
    tO = tb_graph.relu(tAcc)
    tb_graph.new_output(stensor=tO, output_map=(-1, -1, -1))

    O = graph.customized([A, B], tb_graph)
    graph.mark_output(O[0])
    return graph


def _build_unsupported_pallas_graph():
    graph = mi.new_kernel_graph()
    A = graph.new_input(dims=(8, 8), dtype=mi.float16)
    B = graph.new_input(dims=(8, 8), dtype=mi.float16)

    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=1,
        reduction_dimx=8,
    )
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=-1)
    tB = tb_graph.new_input(dtensor=B, input_map=(-1, -1, -1), forloop_dim=-1)
    tC = tb_graph.concat(tA, tB, dim=1)
    tAcc = tb_graph.forloop_accum(tC)
    tb_graph.new_output(stensor=tAcc, output_map=(-1, -1, -1))

    O = graph.customized([A, B], tb_graph)
    graph.mark_output(O[0])
    return graph


def test_generate_pallas_program_codegen():
    graph = _build_simple_pallas_graph()

    result = mi.generate_pallas_program(graph.cygraph, debug=True)

    assert result["errors"] == []
    assert result["output_shapes"] == [[8, 8]]
    assert "pl.pallas_call" in result["code"]
    assert "def custom_kernel_" in result["code"]
    assert "def execute_mugraph" in result["code"]
    assert "jnp.matmul" in result["code"]


def test_generate_pallas_program_reports_unsupported_tb_ops():
    graph = _build_unsupported_pallas_graph()

    result = mi.generate_pallas_program(graph.cygraph, debug=True)

    assert result["errors"]
    assert any("unsupported threadblock op" in err.lower() for err in result["errors"])
