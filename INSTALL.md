# Mirage Installation

The quickest way to try Mirage is through our prebuilt docker images with all dependencies preinstalled. Mirage can also be built from source code using the following instructions.

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

### Install the Mirage python package from Pypi
We also provide the latest version of Mirage on Pypi. Install by running:
```bash
pip install mirage
```

### Check your installation
Just try to import mirage in Python. If there is no output, then Mirage and all dependencies have been successfully installed.
```bash
python -c 'import mirage'
```