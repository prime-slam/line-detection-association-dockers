FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG OPENCV_VERSION=4.5.2

ARG CMAKE_VERSION=3.19.7

ARG ALGORITHM_FOLDER=LSDNet

# install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing -y \
    build-essential \
    ffmpeg \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libfontconfig1 \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libswscale-dev \
    libtiff-dev \
    libv4l-dev \
    libx264-dev \
    libxext6 \
    libxrender1 \
    libxvidcore-dev \
    locales \
    python3-pip \
    unzip \
    wget \
    yasm \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen "en_US.UTF-8"

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/usr/local

# install opencv
RUN wget -O opencv-${OPENCV_VERSION}.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib-${OPENCV_VERSION}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv-${OPENCV_VERSION}.zip \
    && unzip opencv_contrib-${OPENCV_VERSION}.zip \
    && mkdir opencv-${OPENCV_VERSION}/build  \
    && cd opencv-${OPENCV_VERSION}/build \
    && cmake  \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_OPENJPEG=OFF \
    -D WITH_JASPER=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_V4L=OFF \
    -D WITH_FFMPEG=OFF \
    -D WITH_GSTREAMER=OFF \
    -D BUILD_JAVA=OFF \
    -D WITH_CUDA=OFF \
    -D BUILD_TESTS=OFF \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
    -D HAVE_opencv_python3=ON \
    -D BUILD_EXAMPLES=OFF .. \
    && make install -j12 \
    && ldconfig \
    && rm -rf opencv-${OPENCV_VERSION}.zip opencv_contrib-${OPENCV_VERSION}.zip \
    opencv-${OPENCV_VERSION} opencv_contrib-${OPENCV_VERSION}

WORKDIR /detector

COPY $ALGORITHM_FOLDER/requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt --no-cache-dir

COPY $ALGORITHM_FOLDER .

COPY ./common ./common

# Build cpp project
RUN mkdir line_extracter/cmake-build-release \
    && cd line_extracter/cmake-build-release \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j4

ENTRYPOINT ["python3", "run.py"]