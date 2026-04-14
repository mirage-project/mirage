import jax.numpy as jnp

import mirage as mi


def build_graph(
    num_heads: int = 2,
    num_kv_heads: int = 1,
    seq_len: int = 64,
    head_dim: int = 32,
):
    # GQA: Q has num_heads, K/V have num_kv_heads (num_heads % num_kv_heads == 0)
    # Q: (num_heads, seq_len, head_dim)
    # K: (num_kv_heads, head_dim, seq_len)   <- already transposed for matmul
    # V: (num_kv_heads, seq_len, head_dim)
    assert num_heads % num_kv_heads == 0
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(num_heads, seq_len, head_dim), dtype=mi.float16)
    K = graph.new_input(dims=(num_kv_heads, head_dim, seq_len), dtype=mi.float16)
    V = graph.new_input(dims=(num_kv_heads, seq_len, head_dim), dtype=mi.float16)

    A = graph.matmul(Q, K)        # (num_heads, seq_len, seq_len)  -- Q @ K^T
    E = graph.exp(A)              # softmax numerator
    S = graph.reduction(E, 2)     # sum over last dim -> (num_heads, seq_len, 1)
    D = graph.div(E, S)           # normalized attention weights
    O = graph.matmul(D, V)        # (num_heads, seq_len, head_dim)
    graph.mark_output(O)
    return graph


def reference(Q, K, V):
    # Q: (num_heads, seq_len, head_dim)
    # K: (num_kv_heads, head_dim, seq_len)
    # V: (num_kv_heads, seq_len, head_dim)
    A = jnp.matmul(Q, K)
    E = jnp.exp(A)
    S = jnp.sum(E, axis=-1, keepdims=True)
    D = E / S
    return jnp.matmul(D, V)


if __name__ == "__main__":
    num_heads  = 2
    num_kv_heads = 1
    seq_len    = 64
    head_dim   = 32

    graph = build_graph(num_heads, num_kv_heads, seq_len, head_dim)
    optimized_graph = graph.superoptimize(
        config="attention",
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

    import jax
    key = jax.random.PRNGKey(0)
    scale = 0.1
    Q = (jax.random.normal(key, (num_heads, seq_len, head_dim)) * scale).astype(jnp.float16)
    K = (jax.random.normal(key, (num_kv_heads, head_dim, seq_len)) * scale).astype(jnp.float16)
    V = (jax.random.normal(key, (num_kv_heads, seq_len, head_dim)) * scale).astype(jnp.float16)

    out = optimized_graph(inputs=[Q, K, V], debug=True)[0].block_until_ready()
    ref = reference(Q, K, V).block_until_ready()
    max_err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))

    print("output_shape:", out.shape)
    print("max_abs_err:", max_err)
    print("output_sum:", float(out.astype(jnp.float32).sum()))
