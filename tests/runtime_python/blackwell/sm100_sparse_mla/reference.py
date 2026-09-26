"""Independent dense-softmax references; no Mirage/CUDA extension required.

The NumPy path runs on a CPU. The PyTorch path computes in FP32 on the GPU,
using exactly the BF16 inputs supplied to the device kernel.
"""

import numpy as np


def selected_rows(indices, counts, qo_indptr, kv_indptr, pages, last_page_len,
                  page_size, num_pages, num_tokens):
    """Validate metadata and yield (query, physical cache rows).

    Valid selected tokens must be unique. Padding, out-of-range and future
    positions are ignored. Page tables and prefix lengths are caller-owned.
    """
    indices = np.asarray(indices)
    counts = np.asarray(counts)
    qo = np.asarray(qo_indptr)
    ki = np.asarray(kv_indptr)
    pages = np.asarray(pages)
    last = np.asarray(last_page_len)
    if indices.ndim != 2 or indices.shape[0] != num_tokens:
        raise ValueError("indices must have shape [T, K]")
    if counts.shape != (num_tokens,) or np.any(counts < 0) or np.any(counts > indices.shape[1]):
        raise ValueError("counts must be valid index prefix lengths")
    if len(qo) != len(ki) or len(last) != len(qo) - 1:
        raise ValueError("request metadata lengths disagree")
    if qo[0] != 0 or ki[0] != 0 or qo[-1] > num_tokens or ki[-1] > len(pages):
        raise ValueError("invalid indptr bounds")
    if np.any(np.diff(qo) < 0) or np.any(np.diff(ki) < 0):
        raise ValueError("indptr must be nondecreasing")
    for request in range(len(last)):
        n_pages = int(ki[request + 1] - ki[request])
        if n_pages and not 1 <= last[request] <= page_size:
            raise ValueError("invalid last page length")
        seq_len = (n_pages - 1) * page_size + int(last[request]) if n_pages else 0
        history = seq_len - int(qo[request + 1] - qo[request])
        if history < 0:
            raise ValueError("cache must include the entire query chunk")
        for t in range(int(qo[request]), int(qo[request + 1])):
            pos = history + t - int(qo[request])
            selected = indices[t, :int(counts[t])]
            selected = selected[(selected >= 0) & (selected < seq_len) & (selected <= pos)]
            if len(np.unique(selected)) != len(selected):
                raise ValueError("valid selected token indices must be unique")
            physical_pages = pages[ki[request] + selected // page_size]
            if np.any(physical_pages < 0) or np.any(physical_pages >= num_pages):
                raise ValueError("invalid physical page")
            yield t, physical_pages.astype(np.int64) * page_size + selected % page_size


def numpy_reference(q, cache, indices, counts, qo, ki, pages, last, scale):
    q = np.asarray(q, dtype=np.float32)
    cache = np.asarray(cache, dtype=np.float32)
    out = np.zeros((*q.shape[:2], 512), dtype=np.float32)
    kv = cache.reshape(-1, cache.shape[-1])
    for t, rows in selected_rows(indices, counts, qo, ki, pages, last,
                                 cache.shape[1], cache.shape[0], q.shape[0]):
        if not len(rows):
            continue
        selected = kv[rows]
        scores = (q[t] @ selected.T) * scale
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out[t] = probs @ selected[:, :512]
    return out


def torch_reference(q, cache, indices, counts, qo, ki, pages, last, scale):
    import torch

    metadata = [x.detach().cpu().numpy() for x in (indices, counts, qo, ki, pages, last)]
    out = torch.zeros((*q.shape[:2], 512), dtype=torch.float32, device=q.device)
    kv = cache.reshape(-1, cache.shape[-1]).float()
    for t, rows in selected_rows(*metadata, cache.shape[1], cache.shape[0], q.shape[0]):
        if not len(rows):
            continue
        selected = kv[torch.as_tensor(rows, device=q.device, dtype=torch.long)]
        scores = (q[t].float() @ selected.T) * scale
        out[t] = torch.softmax(scores, dim=-1) @ selected[:, :512]
    return out


def make_case(rope_dim=64, heads=8, query_lengths=(1, 3), seq_lengths=(129, 131),
              page_size=64, capacity=2048, seed=7):
    """Small reproducible paged inputs with shuffled physical pages."""
    rng = np.random.default_rng(seed)
    qo = np.array([0, *np.cumsum(query_lengths)], dtype=np.int32)
    page_counts = [(s + page_size - 1) // page_size for s in seq_lengths]
    ki = np.array([0, *np.cumsum(page_counts)], dtype=np.int32)
    pages = rng.permutation(int(ki[-1])).astype(np.int32)
    last = np.array([(s - 1) % page_size + 1 for s in seq_lengths], dtype=np.int32)
    q = (rng.standard_normal((int(qo[-1]), heads, 512 + rope_dim)) * 0.25).astype(np.float32)
    cache = (rng.standard_normal((int(ki[-1]), page_size, 512 + rope_dim)) * 0.5).astype(np.float32)
    indices = np.full((len(q), capacity), -1, dtype=np.int32)
    counts = np.zeros(len(q), dtype=np.int32)
    for b, n in enumerate(query_lengths):
        for local in range(n):
            t = int(qo[b]) + local
            pos = seq_lengths[b] - n + local
            selected = rng.permutation(pos + 1)[:capacity]
            indices[t, :len(selected)] = selected
            counts[t] = len(selected)
    return q, cache, indices, counts, qo, ki, pages, last
