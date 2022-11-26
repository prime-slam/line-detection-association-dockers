FROM python:3.9-slim

ENV PROJECT_DIR=/detector

ENV PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/build/bindings"

WORKDIR $PROJECT_DIR

# install dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    cmake \
    gcc\
    g++ \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir "pybind11[global]" -r requirements.txt

COPY . .

# build project
RUN mkdir build \
    && cd build \
    && cmake .. \
    && make -j4

ENTRYPOINT ["python3", "run.py"]