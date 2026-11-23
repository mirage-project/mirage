import mirage as mi

BF16_BYTES = 2


def test_narrow_of_split_uses_parent_row_stride():
    graph = mi.new_kernel_graph()
    fused = graph.new_input(dims=(8, 64), dtype=mi.bfloat16)
    first, second = graph.split(fused, [48, 16], 1)
    assert second.stride == (64, 1)
    assert second.view_offset == 48 * BF16_BYTES

    row = graph.narrow(second, 0, 1, 1)
    # fused[1, 48]
    assert row.view_offset == (1 * 64 + 48) * BF16_BYTES

    rows = graph.split(first, 4, 0)
    # fused[2 * k, 0]
    assert [r.view_offset for r in rows] == [
        2 * k * 64 * BF16_BYTES for k in range(4)
    ]


def test_narrow_of_narrow():
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(8, 64), dtype=mi.bfloat16)
    cols = graph.narrow(x, 1, 0, 32)
    rows = graph.narrow(cols, 0, 2, 2)
    # x[2, 0]
    assert rows.view_offset == 2 * 64 * BF16_BYTES


def test_narrow_of_input_with_padded_rows():
    graph = mi.new_kernel_graph()
    # e.g. an attached torch slice wide[:, :32] of an (8, 64) tensor
    x = graph.new_input(dims=(8, 32), strides=(64, 1), dtype=mi.bfloat16)
    rows = graph.narrow(x, 0, 3, 2)
    assert rows.view_offset == 3 * 64 * BF16_BYTES
