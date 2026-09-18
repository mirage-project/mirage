"""
Negative tests for disallowed role combinations detected by the
AnnotatedGraph pre-pass.

Each scenario runs in a child process because the C++ compile error is a
std::runtime_error that propagates past the Python binding and calls
std::terminate(). Running in a subprocess lets us inspect its stderr and
exit code instead of crashing the test runner.

  - Case 2: a layer is both a join-consumer AND a fork-consumer (would need
    two trigger_events on one task). MUST abort compile with "case 2".
  - Case 3: a layer is both a fork-producer AND a join-producer (would need
    two dependent_events on one task). MUST abort compile with "case 3".
"""

import os
import subprocess
import sys
import textwrap


_COMMON_HEADER = """
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_pk():
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = 8
    params["max_num_batched_requests"] = 8
    return PersistentKernel(**params)
"""


_CASE2_CHILD = _COMMON_HEADER + """
device = "cuda"
dtype = torch.bfloat16
batch_size = 8
hidden = 4096
torch.manual_seed(0)

x_in  = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
w_l1  = torch.randn(hidden, dtype=dtype, device=device) * 0.1
w_m   = torch.randn(hidden, dtype=dtype, device=device) * 0.1
w_b   = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01
w_c   = torch.randn(hidden, dtype=dtype, device=device) * 0.1
x_l1  = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_m   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_b   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_c   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)

pk = _make_pk()
dt = {
  'x_in': pk.attach_input(x_in,  name='x_in'),
  'w_l1': pk.attach_input(w_l1,  name='w_l1'),
  'w_m':  pk.attach_input(w_m,   name='w_m'),
  'w_b':  pk.attach_input(w_b,   name='w_b'),
  'w_c':  pk.attach_input(w_c,   name='w_c'),
  'x_l1': pk.attach_input(x_l1,  name='x_l1'),
  'x_m':  pk.attach_input(x_m,   name='x_m'),
  'x_b':  pk.attach_input(x_b,   name='x_b'),
  'x_c':  pk.attach_input(x_c,   name='x_c'),
}
target_cc = pk.target_cc
block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)
linear_grid_x = hidden // 96 if hidden % 96 == 0 else hidden // 64

pk.rmsnorm_layer(input=dt['x_in'], weight=dt['w_l1'], output=dt['x_l1'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)
pk.rmsnorm_layer(input=dt['x_in'], weight=dt['w_m'], output=dt['x_m'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)
# B = linear_with_residual(x_l1, w_b, x_m) -> join consumer + fork consumer
pk.linear_with_residual_layer(
    input=dt['x_l1'], weight=dt['w_b'], residual=dt['x_m'], output=dt['x_b'],
    grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)
# C = rmsnorm(x_l1) -> the second fork branch from L1
pk.rmsnorm_layer(input=dt['x_l1'], weight=dt['w_c'], output=dt['x_c'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)

import os as _os
pk.compile(output_dir=_os.path.dirname(_os.path.abspath(__file__) if '__file__' in dir() else '.'))
"""


_CASE3_CHILD = _COMMON_HEADER + """
device = "cuda"
dtype = torch.bfloat16
batch_size = 8
hidden = 4096
torch.manual_seed(0)

x_in  = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
w_l   = torch.randn(hidden, dtype=dtype, device=device) * 0.1
w_z   = torch.randn(hidden, dtype=dtype, device=device) * 0.1
w_j   = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01
w_k   = torch.randn(hidden, dtype=dtype, device=device) * 0.1
x_l   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_z   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_j   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
x_k   = torch.zeros(batch_size, hidden, dtype=dtype, device=device)

pk = _make_pk()
dt = {
  'x_in': pk.attach_input(x_in,  name='x_in'),
  'w_l':  pk.attach_input(w_l,   name='w_l'),
  'w_z':  pk.attach_input(w_z,   name='w_z'),
  'w_j':  pk.attach_input(w_j,   name='w_j'),
  'w_k':  pk.attach_input(w_k,   name='w_k'),
  'x_l':  pk.attach_input(x_l,   name='x_l'),
  'x_z':  pk.attach_input(x_z,   name='x_z'),
  'x_j':  pk.attach_input(x_j,   name='x_j'),
  'x_k':  pk.attach_input(x_k,   name='x_k'),
}
target_cc = pk.target_cc
block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)
linear_grid_x = hidden // 96 if hidden % 96 == 0 else hidden // 64

# L is a fork producer (L -> J, L -> K).
pk.rmsnorm_layer(input=dt['x_in'], weight=dt['w_l'], output=dt['x_l'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)
# Z is another producer feeding the join at J.
pk.rmsnorm_layer(input=dt['x_in'], weight=dt['w_z'], output=dt['x_z'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)
# J is a join consumer: two distinct producers (L and Z).
pk.linear_with_residual_layer(
    input=dt['x_l'], weight=dt['w_j'], residual=dt['x_z'], output=dt['x_j'],
    grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)
# K is the other fork branch from L.
pk.rmsnorm_layer(input=dt['x_l'], weight=dt['w_k'], output=dt['x_k'],
                 grid_dim=(batch_size, 1, 1), block_dim=block_dim)

# L has out-edges to J (join-consumer) and K -> L is join-producer AND
# fork-producer -> Case 3.
import os as _os
pk.compile(output_dir=_os.path.dirname(_os.path.abspath(__file__) if '__file__' in dir() else '.'))
"""


def _run_child(source, case_name):
    """
    Cases 2 and 3 are mathematically coupled: whenever a layer X is a
    join-consumer AND a fork-consumer (case 2), one of X's producers is
    necessarily both a fork-producer and a join-producer (case 3). The two
    violations co-occur for any disallowed topology. We therefore accept
    either error text.
    """
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        timeout=120,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    saw_case2 = "case 2" in combined
    saw_case3 = "case 3" in combined
    if (saw_case2 or saw_case3) and result.returncode != 0:
        marker = "case 2" if saw_case2 else "case 3"
        first_error_line = next(
            (line for line in combined.splitlines() if marker in line),
            "<error line not found>")
        print(f"PASSED: {case_name}: child aborted with disallowed-pattern error")
        print(f"    {first_error_line.strip()}")
        return True
    print(f"FAILED: {case_name}")
    print(f"  returncode = {result.returncode}")
    print(f"  stdout:\n{result.stdout}")
    print(f"  stderr:\n{result.stderr}")
    return False


def test_case2_negative():
    ok = _run_child(_CASE2_CHILD,
                    "case 2 scenario (join-consumer + fork-consumer at B)")
    if not ok:
        sys.exit(1)


def test_case3_negative():
    ok = _run_child(_CASE3_CHILD,
                    "case 3 scenario (fork-producer + join-producer at L)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_case2_negative()
    test_case3_negative()
