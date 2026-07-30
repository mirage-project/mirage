import triton
import torch
import tempfile
import os
import subprocess
import sys
from .core import *
from typing import List, Tuple, Optional
from tqdm import tqdm


class TritonProfiler:
    def __init__(
        self,
        warmup_iters: int = 16,
        profile_iters: int = 1000,
        debug: bool = False,
        save_codes: bool = False,
    ):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.debug = debug
        self.success_num = 0
        self.fail_num = 0
        self.cnt = 0
        self.save_codes = save_codes

    def _generate_profile_code_file(self, graph, target_cc: int) -> Tuple[str, str]:
        """Generate triton code for a graph and save to temp file"""

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            result = generate_triton_program(graph.cygraph, target_cc=target_cc)
            code = result["code"]
            output_shapes = result["output_shapes"]

            code_lines = code.split("\n")
            main_start = -1
            kernel_start = -1

            for i, line in enumerate(code_lines):
                if 'if __name__ == "__main__":' in line:
                    main_start = i
                if "@triton.jit" in line and kernel_start == -1:
                    kernel_start = i

            if main_start != -1:
                main_code = code_lines[main_start + 1 :]
                kernel_code = code_lines[kernel_start:main_start]
                header_code = code_lines[:kernel_start]
                main_code = "\n".join(
                    "        " + line.strip() for line in main_code if line.strip()
                )
            else:
                main_code = ""

            run_kernel_code = f"""
def profile_kernel():
    # Warmup iterations
    for _ in range({self.warmup_iters}):
{main_code}
    
    # Profile iterations
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    for _ in range({self.profile_iters}):
{main_code}
    ender.record()
    torch.cuda.synchronize()
    elapsed_time = starter.elapsed_time(ender) / {self.profile_iters}
    print(f'Average time per iteration: {{elapsed_time:.3f}} ms')

if __name__ == "__main__":
    print("Running kernel...")
    profile_kernel()

"""
            final_code = (
                "\n".join(header_code) + "\n".join(kernel_code) + run_kernel_code
            )

            if self.save_codes:
                with open(f"generated_codes/generated_code_{self.cnt}.py", "w") as _f:
                    _f.write(final_code)
            f.write(final_code)
            return f.name, code, output_shapes

    def _generate_debug_code_file(self, graph, target_cc: int) -> Tuple[str, str]:
        """Generate triton code for a graph and save to temp file"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            header = """
import sys
import traceback
"""
            error_handling = """
def run_with_debug():
    # Enable CUDA debug mode
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    try:
"""
            result = generate_triton_program(graph.cygraph, target_cc=target_cc)
            code = result["code"]
            output_shapes = result["output_shapes"]

            code_lines = code.split("\n")
            main_start = -1
            kernel_start = -1
            for i, line in enumerate(code_lines):
                if 'if __name__ == "__main__":' in line:
                    main_start = i
                if "@triton.jit" in line and kernel_start == -1:
                    kernel_start = i

            if main_start != -1:
                main_code = code_lines[main_start + 1 :]
                kernel_code = code_lines[kernel_start:main_start]
                kernel_name = kernel_code[0].split("(")[0].split(" ")[-1]
                header_code = code_lines[:kernel_start]
                main_code = "\n".join(
                    "        " + line.strip() for line in main_code if line.strip()
                )
            else:
                main_code = ""

            debug_code = """
        print("Debug: Initializing CUDA device")
        device = torch.device('cuda')
        print(f"Debug: Using device: {device}")
        print(f"Debug: CUDA available: {torch.cuda.is_available()}")
        print(f"Debug: Current device: {torch.cuda.current_device()}")
        print(f"Debug: Device name: {torch.cuda.get_device_name(0)}")
"""

            error_handling_end = """
    except Exception as e:
        print("ERROR: " + str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_with_debug()
"""
            final_file = (
                header
                + "\n".join(header_code)
                + "\n".join(kernel_code)
                + error_handling
                + debug_code
                + main_code
                + error_handling_end
            )
            f.write(final_file)
            with open(f"generated_debug_code_{self.cnt}.py", "w") as _f:
                _f.write(final_file)
            return f.name, code, output_shapes

    def _run_profile(self, module_path: str) -> float:
        """Profile execution time by running the module multiple times with progress bars"""
        try:
            process = subprocess.run(
                [sys.executable, module_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the average time from the output
            for line in process.stdout.splitlines():
                if "Average time per iteration" in line:
                    self.success_num += 1
                    return float(line.split(":")[-1].strip().split(" ")[0])

            raise RuntimeError("Failed to extract timing from output")

        except subprocess.CalledProcessError as e:
            tqdm.write("\nError Details:")
            tqdm.write(f"Return code: {e.returncode}")
            tqdm.write(f"Output: {e.output}")
            tqdm.write(f"Stdout: {e.stdout}")
            tqdm.write(f"Stderr: {e.stderr}")
            self.fail_num += 1
            return -1

    def profile_graphs(
        self, graphs: List, target_cc: int = 10
    ) -> Tuple[object, float, List[Tuple[object, float, str]]]:
        """Profile multiple graphs and return best one with timing results

        Returns:
            Tuple containing:
            - Best performing graph
            - Best performance time
            - List of (graph, performance, code) tuples for all graphs
        """
        results = []
        best_graph = None
        best_perf = float("inf")
        best_file_path = None
        best_output_shapes = None

        print(f"Profiling {len(graphs)} candidate graphs...")

        with tqdm(total=len(graphs), desc="Processing graphs", position=0) as pbar:
            for idx, g in enumerate(graphs):
                try:
                    # Generate and save code
                    if self.debug:
                        code_path, code, output_shapes = self._generate_debug_code_file(
                            g, target_cc
                        )
                    else:
                        code_path, code, output_shapes = (
                            self._generate_profile_code_file(g, target_cc)
                        )

                    # Profile the graph with progress description
                    perf = self._run_profile(code_path)

                    tqdm.write(f"Graph {idx}: {perf:.3f} ms")

                    results.append((g, perf, code))

                    if perf < best_perf and perf > 0:
                        best_graph = g
                        best_perf = perf
                        best_file_path = code_path
                        best_output_shapes = output_shapes
                        tqdm.write(f"New best performance: {perf:.3f} ms")

                except Exception as e:
                    tqdm.write(f"Error profiling graph {idx}: {str(e)}")
                    continue

                finally:
                    self.cnt += 1
                    pbar.update(1)

        print(f"\nBest performance: {best_perf:.3f} ms")
        print(f"Successes: {self.success_num}, Failures: {self.fail_num}")
        return best_graph, best_file_path, best_output_shapes


def profile_and_select_best_graph(
    graphs: List,
    target_cc: int = 10,
    warmup_iters: int = 16,
    profile_iters: int = 1000,
    debug_mode: bool = False,
    save_codes: bool = False,
) -> object:
    """Helper function to profile graphs and select the best one"""
    profiler = TritonProfiler(warmup_iters, profile_iters, debug_mode, save_codes)
    best_graph, best_file_path, best_output_shapes = profiler.profile_graphs(
        graphs, target_cc
    )
    return best_graph, best_file_path, best_output_shapes
