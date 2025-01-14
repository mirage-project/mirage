# Mirage Installation

The quickest way to try Mirage is installing the latest stable release from Pypi:
```bash
pip install mirage-project
```

Mirage can also be built from source code using the following instructions.

## Intall from pre-built wheel
We provide some pre-built binary wheels in [Release Page](https://github.com/mirage-project/mirage/releases/latest). For example, to install mirage 0.2.2 compiled with CUDA 12.2 for python 3.10, using the following command:
```bash
pip install https://github.com/mirage-project/mirage/releases/download/v0.2.2/mirage_project-0.2.2+cu122-cp310-cp310-linux_x86_64.whl
```

## Install from source code

### Prerequisties

* CMAKE 3.24 or higher
* Cython 0.28 or higher
* CUDA 11.0 or higher and CUDNN 8.0 or higher

### Install the Mirage python package from source code
To get started, you can clone the Mirage source code from github.
```bash
git clone --recursive https://www.github.com/mirage-project/mirage
cd mirage
```

Then, you can simple build the Mirage runtime library from source code using the following command line
```bash
pip install -e . -v 
```
All dependenices will be automatically installed.

### Check your installation
Just try to import mirage in Python. If there is no output, then Mirage and all dependencies have been successfully installed.
```bash
python -c 'import mirage'
```

## Build Standalone C++ library
If you want to build standalone c++ library, you can follow the steps below.
Given that MIRAGE_ROOT points to top-level mirage project folder.
* Build the Z3 from source.
```bash
cd $MIRAGE_ROOT/deps/z3
mkdir build; cd build
cmake ..
make -j
```
* Export Z3 build directory.
```bash
export Z3_DIR=$MIRAGE_ROOT/deps/z3/build
```
* Build mirage from source.
```bash
cd $MIRAGE_ROOT
mkdir build; cd build
cmake ..
make -j
make install
```
By default, mirage build process will generate a static library. To install mirage in your directory of choice
specify -CMAKE_INSTALL_PREFIX=path/to/your/directory as a cmake option.

## Docker images

We require [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) to run the Mirage [docker images](https://hub.docker.com/r/mlso/mirage).

* First, clone the Mirage gitpub repository to obtain necessary scripts.
```bash
git clone --recursive https://www.github.com/mirage-project/mirage
```

* Second, use the following command to run a Mirage docker image. The default CUDA version is 12.4.
```bash
/path-to-mirage/docker/run_docker.sh mlso/mirage
```

* You are ready to use Mirage now. Try some of our demos to superoptimize DNNs.
```python
python demo/demo_group_query_attention_spec_decode.py --checkpoint demo/checkpoint_group_query_attn_spec_decode.json
```
