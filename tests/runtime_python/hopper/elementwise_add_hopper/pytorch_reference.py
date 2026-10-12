
def elementwise_add_reference(a, b):
    """bf16 + bf16 summed in fp32 and rounded once, like the kernel."""
    return (a.float() + b.float()).to(a.dtype)
