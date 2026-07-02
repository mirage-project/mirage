"""Batched top-k filter: keep each row's top-k scores, set the rest to -inf.

The reference implementation below is correct but slow (pure Python, sorts every
row). The task is to make it fast with numpy without changing its behavior.
"""


def batched_topk_filter(scores, k):
    out = []
    for row in scores:
        pairs = [(i, x) for i, x in enumerate(row)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        keep = set(i for i, _ in pairs[:k])
        out.append([x if i in keep else float("-inf") for i, x in enumerate(row)])
    return out
