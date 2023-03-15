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

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /detector

# install dependencies
RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY ./L-CNN/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir gdown -r requirements.txt

# download pretrained models
RUN mkdir pretrained \
    && gdown "1NvZkEqWNUBAfuhFPNGiCItjy4iU0UOy2" -O pretrained/checkpoint.tar

COPY ./L-CNN .

COPY ./common ./common

ENTRYPOINT ["python3", "run.py"]