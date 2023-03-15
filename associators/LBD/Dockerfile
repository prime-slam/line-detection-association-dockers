# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

FROM python:3.9-slim

ENV PROJECT_DIR=/associator

WORKDIR $PROJECT_DIR

# install dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    cmake \
    gcc\
    g++ \
    libgl1 \
    libglib2.0-0 \
    make \
    libopencv-dev \
    libopencv-contrib-dev \
    libarpack++2-dev \
    libarpack2-dev \
    libsuperlu-dev \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY ./LBD/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# install pytlbd
COPY ./LBD .

COPY ./common ./common

RUN python3 -m pip install .

ENTRYPOINT ["python3", "run.py"]