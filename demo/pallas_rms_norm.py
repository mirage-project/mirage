import jax.numpy as jnp

import mirage as mi


def build_graph(rows: int = 8, hidden: int = 16, out_features: int = 8):
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(rows, hidden), dtype=mi.float16)
    w = graph.new_input(dims=(hidden, out_features), dtype=mi.float16)
    y = graph.rms_norm(x, normalized_shape=(hidden,))
    z = graph.matmul(y, w)
    graph.mark_output(z)
    return graph


def reference(x, w):
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    y = x / jnp.sqrt(mean_square)
    return jnp.matmul(y, w)


if __name__ == "__main__":
    graph = build_graph()
    optimized_graph = graph.superoptimize(
        config="mlp",
        backend="pallas",
        franges=[1],
        use_graph_dataset=False,
        use_cached_graphs=False,
    )

    result = mi.generate_pallas_program(optimized_graph.cygraph, debug=True)
    if result["errors"]:
        print("Pallas transpilation errors:")
        for err in result["errors"]:
            print(f"  - {err}")
        raise SystemExit(1)

    x = (jnp.arange(8 * 16, dtype=jnp.float16).reshape(8, 16) + 1).astype(jnp.float16)
    w = (jnp.arange(16 * 8, dtype=jnp.float16).reshape(16, 8) + 1).astype(jnp.float16)

    out = optimized_graph(inputs=[x, w], debug=True)[0].block_until_ready()
    ref = reference(x, w).block_until_ready()
    max_err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))

    print("output_shape:", out.shape)
    print("max_abs_err:", max_err)
    print("output_sum:", float(out.astype(jnp.float32).sum()))
