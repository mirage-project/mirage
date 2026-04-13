"""
GQA attention with RoPE positional embeddings on TPU via Mirage + Pallas.

RoPE is applied to Q and K outside the Mirage graph (JAX preprocessing),
because Mirage has no slice op to split head_dim in half. The Mirage graph
superoptimizes the attention kernel that follows.

RoPE formula for head vector x of dimension d:
  x_rope[2i]   = x[2i]   * cos(pos * theta_i) - x[2i+1] * sin(pos * theta_i)
  x_rope[2i+1] = x[2i+1] * cos(pos * theta_i) + x[2i]   * sin(pos * theta_i)
where theta_i = 1 / 10000^(2i / d)

Equivalently with the first-half / second-half split:
  x_rope = concat(x[:d/2] * cos - x[d/2:] * sin,
                  x[d/2:] * cos + x[:d/2] * sin)
"""

import jax
import jax.numpy as jnp
import mirage as mi


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def get_rope_embeddings(seq_len: int, head_dim: int, base: float = 10000.0):
    """
    Compute RoPE cos/sin tables.

    Returns:
        cos, sin: both shape (seq_len, head_dim // 2), float16
    """
    half_dim = head_dim // 2
    i = jnp.arange(half_dim, dtype=jnp.float32)
    theta = 1.0 / (base ** (2.0 * i / head_dim))   # (half_dim,)
    positions = jnp.arange(seq_len, dtype=jnp.float32)  # (seq_len,)
    freqs = jnp.outer(positions, theta)             # (seq_len, half_dim)
    cos = jnp.cos(freqs).astype(jnp.float16)
    sin = jnp.sin(freqs).astype(jnp.float16)
    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply RoPE to x.

    Args:
        x:   (num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)

    Returns:
        rotated x, same shape as x
    """
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]          # (..., seq_len, half_dim)
    x2 = x[..., half_dim:]          # (..., seq_len, half_dim)
    cos = cos[None, :, :]            # (1, seq_len, half_dim) — broadcast over heads
    sin = sin[None, :, :]
    return jnp.concatenate([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], axis=-1)


# ---------------------------------------------------------------------------
# Mirage graph (attention only — Q and K are already rotated)
# ---------------------------------------------------------------------------

def build_graph(num_heads: int, num_kv_heads: int, seq_len: int, head_dim: int):
    """
    Build the GQA attention kernel graph.

    Inputs (pre-RoPE-rotated):
        Q:   (num_heads,    seq_len,  head_dim)
        K^T: (num_kv_heads, head_dim, seq_len)   <- transposed for matmul
        V:   (num_kv_heads, seq_len,  head_dim)

    Computes: softmax(Q @ K^T) @ V
    """
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(num_heads, seq_len, head_dim),    dtype=mi.float16)
    K = graph.new_input(dims=(num_kv_heads, head_dim, seq_len), dtype=mi.float16)
    V = graph.new_input(dims=(num_kv_heads, seq_len, head_dim), dtype=mi.float16)

    A = graph.matmul(Q, K)       # (num_heads, seq_len, seq_len)
    E = graph.exp(A)
    S = graph.reduction(E, 2)    # (num_heads, seq_len, 1)
    D = graph.div(E, S)          # attention weights
    O = graph.matmul(D, V)       # (num_heads, seq_len, head_dim)
    graph.mark_output(O)
    return graph


# ---------------------------------------------------------------------------
# Reference (pure JAX)
# ---------------------------------------------------------------------------

def reference(Q, K_T, V, cos, sin):
    """
    Reference GQA + RoPE in pure JAX.

    Args:
        Q:   (num_heads,    seq_len,  head_dim)
        K_T: (num_kv_heads, head_dim, seq_len)  <- transposed storage
        V:   (num_kv_heads, seq_len,  head_dim)
        cos, sin: (seq_len, head_dim // 2)
    """
    Q_rope = apply_rope(Q, cos, sin)

    # K is stored transposed; untranspose, apply RoPE, retranspose
    K = jnp.transpose(K_T, (0, 2, 1))       # (num_kv_heads, seq_len, head_dim)
    K_rope = apply_rope(K, cos, sin)
    K_rope_T = jnp.transpose(K_rope, (0, 2, 1))  # (num_kv_heads, head_dim, seq_len)

    A = jnp.matmul(Q_rope, K_rope_T)
    E = jnp.exp(A)
    S = jnp.sum(E, axis=-1, keepdims=True)
    D = E / S
    return jnp.matmul(D, V)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    num_heads    = 2
    num_kv_heads = 1
    seq_len      = 64
    head_dim     = 32

    # --- superoptimize the attention graph ---
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

    # --- random inputs ---
    key = jax.random.PRNGKey(42)
    scale = 0.1
    Q   = (jax.random.normal(key, (num_heads,    seq_len,  head_dim)) * scale).astype(jnp.float16)
    K_T = (jax.random.normal(key, (num_kv_heads, head_dim, seq_len )) * scale).astype(jnp.float16)
    V   = (jax.random.normal(key, (num_kv_heads, seq_len,  head_dim)) * scale).astype(jnp.float16)

    # --- RoPE embeddings ---
    cos, sin = get_rope_embeddings(seq_len, head_dim)

    # --- apply RoPE outside Mirage ---
    Q_rope   = apply_rope(Q, cos, sin)
    K        = jnp.transpose(K_T, (0, 2, 1))
    K_rope_T = jnp.transpose(apply_rope(K, cos, sin), (0, 2, 1))

    # --- run Mirage kernel ---
    out = optimized_graph(inputs=[Q_rope, K_rope_T, V], debug=True)[0].block_until_ready()

    # --- reference ---
    ref = reference(Q, K_T, V, cos, sin).block_until_ready()

    max_err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print("output_shape:", out.shape)
    print("max_abs_err: ", max_err)
    print("output_sum:  ", float(out.astype(jnp.float32).sum()))
