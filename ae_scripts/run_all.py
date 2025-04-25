# import mirage as mi
import numpy as np
import torch
import os
import subprocess


def run_impl(impl_name, case_name, batch_size):
  mirage_root = os.environ.get('MIRAGE_ROOT')
  if mirage_root is None:
      raise ValueError("Please set the MIRAGE_ROOT environment variable to the path of the Mirage repository.")
  if impl_name == 'mirage':
    script = os.path.join(mirage_root, 'benchmark', f'{case_name}.py')
    checkpoint = os.path.join(mirage_root, 'benchmark', 'saved_mugraphs', f'{case_name}_bs{batch_size}.json')
  elif impl_name == 'mirage_end2end':
    script = os.path.join(mirage_root, 'benchmark', 'end-to-end', f'{case_name}.py')
  elif impl_name == 'pytorch_end2end':
    script = os.path.join(mirage_root, 'demo', 'pytorch', f'{case_name}.py')
  else:
    script = os.path.join(mirage_root, 'mirage_baselines', impl_name, f'{case_name}.py')
  if not os.path.exists(script):
    print(f"Script {script} does not exist.")
    return
  stdout_file = os.path.join(mirage_root, 'ae_scripts', f'{case_name}_{impl_name}_bs{batch_size}.out')

  try:
    with open(stdout_file, 'w') as stdout:
      if impl_name == 'mirage':
        subprocess.run(
          ['python3', script, '-b', str(batch_size), '--file', checkpoint],
          stdout=stdout
        )
      else:
        subprocess.run(
          ['python3', script, '-b', str(batch_size)],
          stdout=stdout
        )
      
  except Exception as e:
    print(f"Error running baseline {impl_name} for {case_name} with batch size {batch_size}: {e}")


def parse_results(impl_name, case_name, batch_size):
  mirage_root = os.environ.get('MIRAGE_ROOT')
  if mirage_root is None:
      raise ValueError("Please set the MIRAGE_ROOT environment variable to the path of the Mirage repository.")
  stdout_file = os.path.join(mirage_root, 'ae_scripts', f'{case_name}_{impl_name}_bs{batch_size}.out')
  if not os.path.exists(stdout_file):
    return float('inf')

  with open(stdout_file, 'r') as f:
    lines = f.readlines()
    if impl_name in {'pytorch', 'flashattn', 'mirage_end2end'}:
      for line in lines:
        try:
          return float(line)
        except ValueError:
          pass
    elif impl_name == 'tensorrt':
      # Check for the line containing "Runtime = "
      for line in lines:
        if "Runtime = " in line:
          # Extract the runtime value
          parts = line.split("=")
          if len(parts) > 1:
            return float(parts[1].strip())
    elif impl_name == 'triton':
      for line in lines:
        if line.startswith("0"):
          # Extract the runtime value
          parts = line.split()
          if len(parts) > 2:
            return float(parts[2])
    elif impl_name == 'mirage':
      for line in lines:
        if line.startswith("Best muGraph run time (ms): "):
          # Extract the runtime value
          parts = line.split(":")
          if len(parts) > 1:
            return float(parts[1].strip())
    elif impl_name == 'pytorch_end2end':
      for line in lines:
        if line.startswith("Torch "):
          # Extract the runtime value
          parts = line.split(":")
          if len(parts) > 1:
            return float(parts[1].strip())
    elif impl_name == 'taso':
      best = float('inf')
      for line in lines:
        if "Cost metrics: " in line:
          # parse from format like Cost metrics: exe_time(1.7097) flops(0.2500) memory_access(0.5000) kernel_launches(4)
          parts = line.split(" ")
          for part in parts:
            if part.startswith("exe_time("):
              # Extract the runtime value
              time_part = part.split("(")[1].split(")")[0]
              time = float(time_part)
              if time < best:
                best = time
      return best
    
  return float('inf')


def benchmark_evaluation():
  # models = ['gated_mlp', 'gqa', 'lora', 'norm_transformer', 'qknorm_gqa', 'rmsnorm']
  models = ['gated_mlp']
  batch_sizes = [1, 8]
  results = dict()
  for model in models:
    for batch_size in batch_sizes:
      for impl_name in ['pytorch', 'flashattn', 'taso', 'tensorrt', 'triton', 'mirage']:
        run_impl(impl_name, model, batch_size)
        results[(model, batch_size, impl_name)] = parse_results(impl_name, model, batch_size)
        print(f"Model: {model}, Batch Size: {batch_size}, Impl: {impl_name}, Time: {results[(model, batch_size, impl_name)]}")

  for model in models:
    for batch_size in batch_sizes:
      best_impl = min(['pytorch', 'flashattn', 'taso', 'tensorrt', 'triton'], key=lambda impl: results[(model, batch_size, impl)])
      best_time = results[(model, batch_size, best_impl)]
      mirage_time = results[(model, batch_size, 'mirage')]
      speedup = best_time / mirage_time
      print(f"Model: {model}, Batch Size: {batch_size}, Best Impl: {best_impl}, Best Time: {best_time:.4f}, Mirage Time: {mirage_time:.4f}, Speedup: {speedup:.2f}")


def end2end_evaluation():
  models = ['chameleon-7b', 'llama-8b', 'lora', 'ngpt']
  batch_sizes = [1, 8]
  for model in models:
    for batch_size in batch_sizes:
      run_impl('pytorch_end2end', model, batch_size)
      run_impl('mirage_end2end', model, batch_size)
      pytorch_time = parse_results('pytorch_end2end', model, batch_size)
      mirage_time = parse_results('mirage_end2end', model, batch_size)
      print(f"Model: {model}, Batch Size: {batch_size}, PyTorch Time: {pytorch_time:.4f}, Mirage Time: {mirage_time:.4f}, Speedup: {pytorch_time / mirage_time:.2f}")


if __name__ == "__main__":
  benchmark_evaluation()
  end2end_evaluation()
