import torch


def elementwise_add_ref(a, b):
    """Elementwise add: out = a + b."""
    return (a.to(torch.float32) + b.to(torch.float32)).to(a.dtype)
