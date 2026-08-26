import torch


def tensor_init_ref(linear_output, init_val=0.0):
    """PyTorch reference for ``tensor_init_layer``.

    The MPK kernel (``kernel::tensor_init_zero_sm100_task_impl`` in
    ``include/mirage/persistent_kernel/tasks/blackwell/tensor_init.cuh``)
    zero-fills its ``linear_output`` argument via vectorized 16-byte stores.
    The kernel is hard-wired to ``init_val = 0`` -- the parameter on this
    reference exists only to make assertions slightly more flexible.

    The ``linear_input`` argument of ``tensor_init_layer`` is a graph-edge
    placeholder (it appears as both a read input and as the dummy second
    output for dependency wiring), and is neither read nor written by the
    kernel.
    """
    return torch.full_like(linear_output, init_val)
