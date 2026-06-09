import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import elementwise_add_ref


def test_elementwise_add_testmode():
    device = "cuda"
    torch.manual_seed(42)
    batch_size = 16
    hidden = 4096

    a = torch.randn(batch_size, hidden, dtype=torch.bfloat16, device=device)
    b = torch.randn(batch_size, hidden, dtype=torch.bfloat16, device=device)
    out = torch.zeros(batch_size, hidden, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size

    pk = PersistentKernel(**params)
    a_dt = pk.attach_input(a, name="a")
    b_dt = pk.attach_input(b, name="b")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    pk.elementwise_add_layer(
        input_a=a_dt,
        input_b=b_dt,
        output=out_dt,
        grid_dim=(batch_size, 1, 1),
        block_dim=block_dim,
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref = elementwise_add_ref(a, b)
    max_diff = (out - ref).abs().max().item()
    print(f"Max diff: {max_diff}")
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_elementwise_add_testmode()
