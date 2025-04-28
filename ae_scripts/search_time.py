import numpy as np
import torch
import os
import subprocess


def run_impl(batch_size, num_thread, max_num_threadblock_graph_op):
  mirage_root = os.environ.get('MIRAGE_ROOT')
  if mirage_root is None:
      raise ValueError("Please set the MIRAGE_ROOT environment variable to the path of the Mirage repository.")
  script = os.path.join(mirage_root, 'benchmark', f'rmsnorm.py')
  if not os.path.exists(script):
    print(f"Script {script} does not exist.")
    return
  checkpoint = os.path.join(mirage_root, 'ae_scripts', 'search_time.json')
  if os.path.exists(checkpoint):
    os.remove(checkpoint)

  try:
    subprocess.run(
      ['python3', script, '-b', str(batch_size), '--file', checkpoint, '-t', str(num_thread), '--max_num_threadblock_graph_op', str(max_num_threadblock_graph_op)],
    )
      
  except Exception as e:
    print(f"Error running rmsnorm with num_thread {num_thread} and max_num_threadblock_graph_op {max_num_threadblock_graph_op}: {e}")


if __name__ == "__main__":
  for num_thread in [1, 8]:
    for max_num_threadblock_graph_op in [5, 6, 7, 8, 9, 10, 11]:
      print(f"========================== num_thread: {num_thread}, max_num_threadblock_graph_op: {max_num_threadblock_graph_op} ==========================")
      run_impl(1, num_thread, max_num_threadblock_graph_op)
