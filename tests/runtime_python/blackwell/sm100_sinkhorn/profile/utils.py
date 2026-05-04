import torch


def sinkhorn_knopp_torch(x, repeat=20, eps=1e-9):
    """
    mHC Sinkhorn-Knopp forward projection for Hres.
    x: (..., hidden_size, hidden_size) unconstrained comb_res_mix logits
    """
    comb = torch.softmax(x.float(), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, repeat):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb
