#!/usr/bin/env python3
"""Build a torch CUDAExtension whose setup.py pins nvcc 12.8 while this venv's
torch was built against CUDA 13.0. The shipped MPK JIT does exactly that -- it
compiles the megakernel with `shutil.which("nvcc")` (12.8) and dlopens it into
the same torch process -- so neutralising torch's advisory version check
reproduces the shipped toolchain rather than deviating from it.
usage: build_ext_nover.py <dir-containing-setup.py>
"""
import os, runpy, sys
import torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None
d = sys.argv[1]
os.chdir(d)
sys.argv = ["setup.py", "build_ext", "--inplace"]
runpy.run_path("setup.py", run_name="__main__")
