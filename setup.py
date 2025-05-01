# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import shutil
from os import path
from pathlib import Path
import sys
import sysconfig
from setuptools import find_packages, setup, Command
from contextlib import contextmanager
import subprocess

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                              
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

import z3
z3_path = path.dirname(z3.__file__)
print(f"Z3 path: {z3_path}", flush=True)

# Use version.py to get package version
version_file = os.path.join(os.path.dirname(__file__), "python/mirage/version.py")
with open(version_file, "r") as f:
    exec(f.read())  # This will define __version__

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        mirage_path = ''
        cython_path = path.join(mirage_path, "python/mirage/_cython")
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "mirage.%s" % fn[:-4],
                ["%s/%s" % (cython_path, fn)],
                include_dirs=[path.join(mirage_path, "include"),
                              path.join(mirage_path, "deps", "json", "include"),
                              path.join(mirage_path, "deps", "cutlass", "include"),
                              path.join(z3_path, "include"),
                              "/usr/local/cuda/include"],
                libraries=["mirage_runtime", "cudadevrt", "cudart_static", "cudnn", "cublas", "cudart", "cuda", "z3", "gomp", "rt"],
                library_dirs=[path.join(mirage_path, "build"),
                              path.join(z3_path, "lib"),
                              "/usr/local/cuda/lib",
                              "/usr/local/cuda/lib64",
                              "/usr/local/cuda/lib64/stubs"],
                extra_compile_args=["-std=c++17", "-fopenmp"],
                extra_link_args=["-fPIC", "-fopenmp"],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)

# build Mirage runtime library
try:
    nvcc_path = shutil.which('nvcc')
    os.environ['CUDACXX'] = nvcc_path if nvcc_path else '/usr/local/cuda/bin/nvcc'
    mirage_path = path.dirname(__file__)
    # z3_path = os.path.join(mirage_path, 'deps', 'z3', 'build')
    # os.environ['Z3_DIR'] = z3_path
    if mirage_path == '':
        mirage_path = '.'
    os.makedirs(mirage_path, exist_ok=True)
    os.chdir(mirage_path)
    build_dir = os.path.join(mirage_path, 'build')
    
    cc_path = shutil.which('gcc')
    os.environ['CC'] = cc_path if cc_path else '/usr/bin/gcc'
    cxx_path = shutil.which('g++')
    os.environ['CXX'] = cxx_path if cxx_path else '/usr/bin/g++'
    print(f"CC: {os.environ['CC']}, CXX: {os.environ['CXX']}", flush=True)
  
    # Create the build directory if it does not exist
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(['cmake', '..',
                           '-DZ3_CXX_INCLUDE_DIRS=' + z3_path + '/include/',
                           '-DZ3_LIBRARIES=' + path.join(z3_path, 'lib', 'libz3.so'),
                           '-DCMAKE_C_COMPILER=' + os.environ['CC'],
                           '-DCMAKE_CXX_COMPILER=' + os.environ['CXX'],
                          ], cwd=build_dir, env=os.environ.copy())
    subprocess.check_call(['make', '-j'], cwd=build_dir, env=os.environ.copy())
    print("Mirage runtime library built successfully.")
except subprocess.CalledProcessError as e:
    print("Failed to build runtime library.")
    raise SystemExit(e.returncode)

setup_args = {}

# Create requirements list from requirements.txt
with open(Path(__file__).parent / "requirements.txt", "r") as reqs_file:
    requirements = reqs_file.read().strip().split("\n")
print(f"Requirements: {requirements}")

INCLUDE_BASE = "python/mirage/include"
@contextmanager
def copy_include():
    if not path.exists(INCLUDE_BASE):
        src_dirs = ["deps/cutlass/include", "deps/json/include"]
        for src_dir in src_dirs:
            shutil.copytree(src_dir, path.join(INCLUDE_BASE, src_dir))
        # copy mirage/transpiler/runtime/* 
        # to python/mirage/include/mirage/transpiler/runtime/*
        # instead of python/mirage/include/include/mirage/transpiler/runtime/*
        include_mirage_dirs = ["include/mirage/transpiler/runtime", 
                               "include/mirage/triton_transpiler/runtime"]
        include_mirage_dsts = [path.join(INCLUDE_BASE, "mirage/transpiler/runtime"), 
                               path.join(INCLUDE_BASE, "mirage/triton_transpiler/runtime")]
        for include_mirage_dir, include_mirage_dst in zip(include_mirage_dirs, include_mirage_dsts):
            shutil.copytree(include_mirage_dir, include_mirage_dst)

        config_h_src = path.join(mirage_path, "include/mirage/config.h") # Needed by transpiler/runtime/threadblock/utils.h
        config_h_dst = path.join(INCLUDE_BASE, "mirage/config.h")
        shutil.copy(config_h_src, config_h_dst)
        yield True
    else:
        yield False
    shutil.rmtree(INCLUDE_BASE)

with copy_include() as copied:
    if not copied:
        print("WARNING: include directory already exists. Not copying again. "
              f"This may cause issues. Please remove {INCLUDE_BASE} and rerun setup.py", flush=True)
    
    setup(name='mirage-project',
          version=__version__,
          description="Mirage: A Multi-Level Superoptimizer for Tensor Algebra",
          zip_safe=False,
          install_requires=requirements,
          packages=find_packages(where='python'),
          package_dir={'': 'python'},
          url='https://github.com/mirage-project/mirage',
          ext_modules=config_cython(),
          include_package_data=True,
          #**setup_args,
          )