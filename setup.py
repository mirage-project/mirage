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
from os import path
from pathlib import Path
import sys
import sysconfig
from setuptools import find_packages, setup, Command
import subprocess

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                              
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "python/mirage/_cython")
        mirage_path = path.dirname(__file__)
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "mirage.%s" % fn[:-4],
                ["%s/%s" % (cython_path, fn)],
                include_dirs=[path.join(mirage_path, "include"),
                              path.join(mirage_path, "deps", "json", "include"),
                              path.join(mirage_path, "deps", "cutlass", "include"),
                              "/usr/local/cuda/include"],
                libraries=["mirage_runtime", "cudadevrt", "cudart_static", "cudnn", "cublas", "cudart", "cuda", "z3"],
                library_dirs=[path.join(mirage_path, "build"),
                              path.join(mirage_path, "deps", "z3", "build"),
                              "/usr/local/cuda/lib64",
                              "/usr/local/cuda/lib64/stubs"],
                extra_compile_args=["-std=c++17"],
                extra_link_args=["-fPIC"],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
        return []
    
try:
    print('Installing Z3 right now...')
    mirage_path = path.dirname(__file__)
    z3_path = os.path.join(mirage_path, 'deps', 'z3')
    build_dir = os.path.join(z3_path, 'build')
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)
    print(f"Changed to directory: {os.getcwd()}")
    print(f"running cmake command at {build_dir}")
    subprocess.check_call(['cmake', '..'], cwd=build_dir)
    print("finished running cmake command")
    print(f"running make command at {build_dir}")
    subprocess.check_call(['make', '-j'], cwd=build_dir)
    print("running make command")

    # update LD_LIBRARY_PATH
    print("here")
    print(f"{build_dir}:{os.environ.get('LD_LIBRARY_PATH','')}")
    os.environ['LD_LIBRARY_PATH'] = f"{build_dir}:{os.environ.get('LD_LIBRARY_PATH','LD_LIBRARY_PATH')}"
    print("Z3 installed successfully.")
except subprocess.CalledProcessError as e:
    print("Failed to install Z3.")
    raise SystemExit(e.returncode)

# build Mirage runtime library
try:
    os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
    mirage_path = path.dirname(__file__)
    z3_path = os.path.join(mirage_path, 'deps', 'z3', 'build')
    os.environ['Z3_DIR'] = z3_path
    os.makedirs(mirage_path, exist_ok=True)
    os.chdir(mirage_path)
    build_dir = os.path.join(mirage_path, 'build')
  
    # Create the build directory if it does not exist
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(['cmake', '..'], cwd=build_dir, env=os.environ.copy())
    subprocess.check_call(['make', '-j'], cwd=build_dir, env=os.environ.copy())
    print("Mirage runtime library built successfully.")
except subprocess.CalledProcessError as e:
    print("Failed to build runtime library.")
    raise SystemExit(e.returncode)

setup_args = {}

# Create requirements list from requirements.txt
with open(Path(__file__).parent / "requirements.txt", "r") as reqs_file:
    requirements = reqs_file.read().strip().split("\n")

setup(name='mirage',
      version="0.1.1",
      description="Mirage: A Multi-Level Superoptimizer for Tensor Algebra",
      zip_safe=False,
      install_requires=requirements,
      packages=find_packages(),
      url='https://github.com/mirage-project/mirage',
      ext_modules=config_cython(),
      #**setup_args,
      )
