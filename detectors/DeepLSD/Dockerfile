# Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ARG ALGORITHM_FOLDER=DeepLSD

ARG DEBIAN_FRONTEND=noninteractive

ARG WIREFRAME_WEIGHTS=https://www.polybox.ethz.ch/index.php/s/FQWGkH57UNTqlJZ/download

ARG MEGA_DEPTH_WIGHTS=https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys/download

ARG CERES_VERSION=2.1.0

ARG CMAKE_VERSION=3.19.7

ARG GLOG_VERSION=0.6.0

WORKDIR /detector

# install dependencies
RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    build-essential \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libopencv-dev \
    python3-opencv \
    libsuitesparse-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/usr/local

# install glog
RUN wget -O glog-${GLOG_VERSION}.tar.gz https://github.com/google/glog/archive/refs/tags/v${GLOG_VERSION}.tar.gz \
    && tar zxf glog-${GLOG_VERSION}.tar.gz \
    && mkdir glog-${GLOG_VERSION}/build  \
    && cd glog-${GLOG_VERSION}/build \
    && cmake .. \
    && make install -j4 \
    && rm -rf glog-${GLOG_VERSION}.tar.gz glog-${GLOG_VERSION}

# install ceres
RUN wget http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz \
    && tar zxf ceres-solver-${CERES_VERSION}.tar.gz \
    && mkdir ceres-solver-${CERES_VERSION}/build  \
    && cd ceres-solver-${CERES_VERSION}/build \
    && cmake .. \
    && make install -j4 \
    && rm -rf ceres-solver-${CERES_VERSION}.tar.gz ceres-solver-${CERES_VERSION}

# download model wights
RUN mkdir weights \
    && wget $WIREFRAME_WEIGHTS -O weights/deeplsd_wireframe.tar \
    && wget $MEGA_DEPTH_WIGHTS -O weights/deeplsd_md.tar

COPY $ALGORITHM_FOLDER .

# install python requirements
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir glog -r requirements.txt

# build project
RUN bash install.sh

COPY ./common ./common

ENTRYPOINT ["python3", "run.py"]