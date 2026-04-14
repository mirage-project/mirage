"""
Benchmark: RMSNorm + matmul, Mirage Pallas vs jax.jit vs plain JAX
"""

import time
import jax
import jax.numpy as jnp
import mirage as mi


# Reference implementations

def rms_norm_matmul(x, w):
    mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    y = x / jnp.sqrt(mean_sq)
    return jnp.matmul(y, w)


jit_rms_norm_matmul = jax.jit(rms_norm_matmul)


# Mirage graph builder

def build_mirage_graph(rows, hidden, out_features):
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(rows, hidden), dtype=mi.float16)
    w = graph.new_input(dims=(hidden, out_features), dtype=mi.float16)
    y = graph.rms_norm(x, normalized_shape=(hidden,))
    z = graph.matmul(y, w)
    graph.mark_output(z)
    return graph


# Timing helper

def benchmark(fn, *args, warmup=5, iters=50):
    """Returns median latency in milliseconds."""
    # warmup as first few calls would be very slow
    for _ in range(warmup):
        out = fn(*args)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, (list, tuple)) and hasattr(out[0], "block_until_ready"):
            out[0].block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        # JAX is async: block until TPU actually finishes before stopping the timer
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, (list, tuple)) and hasattr(out[0], "block_until_ready"):
            out[0].block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


# Main benchmark loop

SIZES = [
    (8,   16,  8),
    (16,  64,  32),
    (32,  256, 128),
    (64,  512, 256),
    (128, 1024, 512),
]

WARMUP = 5
ITERS  = 50

print(f"{'rows':>6} {'hidden':>8} {'out':>6}  {'plain_jax':>12} {'jax_jit':>12} {'mirage':>12}  {'jit_speedup':>14} {'mirage_speedup':>16}")
print("-" * 100)

device = jax.devices()[0]
print(f"Running on: {device}\n")

for rows, hidden, out_features in SIZES:
    x_np = jnp.ones((rows, hidden), dtype=jnp.float16)
    w_np = jnp.ones((hidden, out_features), dtype=jnp.float16)
    x = jax.device_put(x_np, device)
    w = jax.device_put(w_np, device)

    # plain JAX
    t_plain = benchmark(rms_norm_matmul, x, w, warmup=WARMUP, iters=ITERS)

    # jax.jit: trigger compilation first
    _ = jit_rms_norm_matmul(x, w).block_until_ready()
    t_jit = benchmark(jit_rms_norm_matmul, x, w, warmup=WARMUP, iters=ITERS)

    # Mirage Pallas
    try:
        graph = build_mirage_graph(rows, hidden, out_features)
        optimized = graph.superoptimize(
            config="mlp",
            backend="pallas",
            franges=[1],
            use_graph_dataset=False,
            use_cached_graphs=False,
        )

        def mirage_fn(x, w):
            return optimized(inputs=[x, w])[0]

        t_mirage = benchmark(mirage_fn, x, w, warmup=WARMUP, iters=ITERS)
        mirage_str    = f"{t_mirage:>12.3f}"
        mirage_su_str = f"{t_plain / t_mirage:>16.2f}x"
    except Exception as e:
        mirage_str    = f"{'ERROR':>12}"
        mirage_su_str = f"{'N/A':>16}"

    jit_speedup = t_plain / t_jit

    print(
        f"{rows:>6} {hidden:>8} {out_features:>6}  "
        f"{t_plain:>12.3f} {t_jit:>12.3f} {mirage_str}  "
        f"{jit_speedup:>14.2f}x {mirage_su_str}"
    )

print()
print("All times in milliseconds (median over 50 iterations).")
print("Speedup columns show how much faster than plain JAX (no JIT).")
