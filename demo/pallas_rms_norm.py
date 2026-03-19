import jax.numpy as jnp

import mirage as mi


def build_rms_norm_graph(rows: int = 8, cols: int = 16):
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(rows, cols), dtype=mi.float16)

    # The Pallas v1 backend supports RMSNorm through customized TB graphs,
    # which are normalized into primitive ops before codegen.
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=1,
        reduction_dimx=cols,
    )
    tx = tb_graph.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=-1)
    ty = tb_graph.rms_norm(tx)
    # Threadblock outputs in Mirage are expected to come after an accum op.
    # With forloop_range=1, this is a semantic no-op.
    ty = tb_graph.forloop_accum(ty)
    tb_graph.new_output(stensor=ty, output_map=(-1, -1, -1))

    y = graph.customized([x], tb_graph)[0]
    graph.mark_output(y)
    graph.backend = "pallas"
    return graph


def rms_norm_reference(x):
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(mean_square)


if __name__ == "__main__":
    graph = build_rms_norm_graph()
    result = mi.generate_pallas_program(graph.cygraph, debug=True)

    if result["errors"]:
        print("Pallas transpilation errors:")
        for err in result["errors"]:
            print(f"  - {err}")
        raise SystemExit(1)

    x = (jnp.arange(8 * 16, dtype=jnp.float16).reshape(8, 16) + 1).astype(jnp.float16)
    y = graph(inputs=[x], debug=True)[0].block_until_ready()
    ref = rms_norm_reference(x).block_until_ready()
    max_err = float(jnp.max(jnp.abs(y.astype(jnp.float32) - ref.astype(jnp.float32))))

    print("output_shape:", y.shape)
    print("max_abs_err:", max_err)
    print("output_sum:", float(y.astype(jnp.float32).sum()))
