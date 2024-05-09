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

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mv Miniconda3-latest-Linux-x86_64.sh ~/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install cmake make
RUN /opt/conda/bin/conda install -c conda-forge numpy
RUN /opt/conda/bin/conda install -c anaconda cython

ENV PATH /opt/conda/bin:$PATH
ENV MIRAGE_HOME /usr/mirage/
ENV LD_LIBRARY_PATH /usr/mirage/deps/z3/build/:$LD_LIBRARY_PATH

COPY install_mirage.sh /install/install_mirage.sh
RUN bash /install/install_mirage.sh

WORKDIR /usr/mirage
#RUN chmod -R a+w /usr/mirage
