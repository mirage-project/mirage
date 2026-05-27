# Mirage Installation

The quickest way to try Mirage is installing the latest stable release from Pypi:
```bash
pip install mirage-project
```

Mirage can also be built from source code using the following instructions.

## Install from pre-built wheel
We provide some pre-built binary wheels in [Release Page](https://github.com/mirage-project/mirage/releases/latest). Check the release page for available wheels for your CUDA version and Python version.

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

We provide a Dockerfile to build a Mirage image from source. Requires [docker](https://docs.docker.com/engine/installation/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone --recursive https://www.github.com/mirage-project/mirage
cd mirage
docker build -t mirage docker/
docker run --gpus all --rm -it mirage
```

## Modal

[Modal](https://modal.com) provides GPUs to try MPK with minimal setup.

```
pip install modal
modal setup
modal run demo/qwen3/demo_modal.py
```

