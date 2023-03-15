FROM python:3.9-slim

ENV PYTHON_VERSION=3.9

ENV PYTHONPATH="$PYTHONPATH:/usr/local/lib/kht/python/${PYTHON_VERSION}"

WORKDIR /detector

# install dependencies
RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    cmake \
    gcc\
    g++ \
    libboost-numpy-dev \
    libboost-python-dev \
    libgl1 \
    libglib2.0-0 \
    make \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY ./KHT/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY ./KHT .

COPY ./common ./common

# build cpp
RUN cd cpp \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . --config Release --target install

# build python bindings
RUN cd python  \
    && mkdir build  \
    && cd build \
    && cmake .. \
    && cmake --build . --config Release --target install \
    && export

ENTRYPOINT ["python3", "run.py"]