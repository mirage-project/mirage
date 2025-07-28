import torch
import mirage as mi
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiling", action="store_true", help="Enable mirage profiling mode"
    )
    args = parser.parse_args()

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(512, 256), dtype=mi.float16)
    W = graph.new_input(dims=(256, 1024), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(4, 4, 1), block_dim=(256, 1, 1), forloop_range=8, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(0, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(-1, 1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tO = tb_graph.forloop_accum(tM)
    tb_graph.new_output(stensor=tO, output_map=(0, 1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [
        torch.ones(512, 256, dtype=torch.float16, device="cuda:0"),
        torch.ones(256, 1024, dtype=torch.float16, device="cuda:0"),
    ]

    outputs = graph(
        inputs=input_tensors,
        num_warp_groups=2,
        pipeline_stages=4,
        profiling=args.profiling,
        file_id=3,
        target_cc=100,
    )

    print("Verifying output correctness...")

    X_tensor = input_tensors[0]  # (512, 256)
    W_tensor = input_tensors[1]  # (256, 1024)

    reference_output = torch.matmul(X_tensor, W_tensor)  # (512, 1024)

    mirage_output = outputs[0]

    print(f"Input X shape: {X_tensor.shape}")
    print(f"Input W shape: {W_tensor.shape}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Mirage output shape: {mirage_output.shape}")

    if reference_output.shape == mirage_output.shape:
        abs_diff = torch.abs(reference_output - mirage_output)
        max_abs_error = torch.max(abs_diff).item()
        print(f"Max absolute error: {max_abs_error:.6e}")

        eps = 1e-8
        rel_diff = abs_diff / (torch.abs(reference_output) + eps)
        passed = max_abs_error < 1e-4
        print(f"Verification status: {'Passed' if passed else 'Failed'}")
        if not passed:
            print("Error exceeds expected range.")
        else:
            print("Output verification passed.")
    else:
        print(f"Output shape mismatch!")
        print(f"Expected: {reference_output.shape}")
        print(f"Actual: {mirage_output.shape}")
