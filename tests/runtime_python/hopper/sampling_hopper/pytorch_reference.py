import torch


def chunk_topk_reference(logits, num_tasks, topk_max, vocab, temperature):
    """Per-chunk top-topk_max values (descending), chunk max and exp-sum of
    logits/temperature, with lm_head padding (positions >= vocab) excluded."""
    batch, padded = logits.shape
    chunk = padded // num_tasks
    inv_t = 1.0 / temperature if temperature > 0 else 1.0
    x = logits.float() * inv_t
    x[:, vocab:] = float("-inf")
    x = x.view(batch, num_tasks, chunk)
    top = x.topk(topk_max, dim=-1).values
    cmax = x.max(dim=-1).values
    csum = torch.exp(x - cmax.unsqueeze(-1)).nan_to_num(0.0).sum(-1)
    return top, cmax, csum


def allowed_sets(logits, vocab, top_k, top_p):
    """Tokens a correct top-k / top-p sampler may return for each row."""
    probs = torch.softmax(logits[:, :vocab].float(), dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True)
    sets = []
    for row in range(logits.shape[0]):
        keep = order[row]
        if top_k > 0:
            keep = keep[:top_k]
        if top_p < 1.0:
            cum = torch.cumsum(probs[row, keep], 0)
            n = int((cum >= top_p).nonzero()[0]) + 1 if bool(
                (cum >= top_p).any()) else keep.numel()
            keep = keep[:n]
        sets.append(set(keep.tolist()))
    return sets
