import torch


def tensor_init_ref(input_tensor, init_val=0.0):
    """Tensor init: fill ``input_tensor`` with ``init_val`` (default 0.0).

    Mirrors ``kernel::tensor_init_sm100_task_impl`` in
    ``include/mirage/persistent_kernel/tasks/blackwell/tensor_init.cuh``,
    which writes ``T(init_val)`` to every element of the input tensor
    (loop over ``BATCH_SIZE`` rows x ``OUTPUT_SIZE`` cols, with
    ``OUTPUT_STRIDE``).  The MPK code generator currently hard-codes
    ``init_val = 0`` (see ``register_tensor_init_task`` in
    ``src/kernel/task_register.cc``), so this is effectively a zero-fill.

    The ``dummy_input`` / ``dummy_output`` arguments of
    ``tensor_init_layer`` are graph-edge placeholders only -- they are
    not read or written by the kernel.
    """
    out = torch.full_like(input_tensor, init_val)
    return out
