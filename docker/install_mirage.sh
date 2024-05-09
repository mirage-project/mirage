#!/bin/bash
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

set -e
set -u
set -o pipefail

cd /usr
git clone --recursive -b python https://github.com/jiazhihao/attention_superoptimizer.git mirage
# build z3
cd /usr/mirage/deps/z3
mkdir -p build
cd build
cmake ..
make -j
# build mirage runtime
cd /usr/mirage
mkdir -p build
cd build
export Z3_DIR=/usr/mirage/deps/z3/build
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake ..
make -j
# build mirage python package
cd /usr/mirage/python
python setup.py install
