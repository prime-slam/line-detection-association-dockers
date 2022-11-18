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

ARG MODEL_PATH=https://vcl.ucsd.edu/letr/checkpoints/res101/res101_stage2_focal.zip

WORKDIR /detector

# install dependencies
RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install python requirements
COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# download pretrained models
RUN mkdir pretrained \
    && wget $MODEL_PATH -O checkpoint.zip \
    && unzip checkpoint.zip \
    && mv exp/res101_stage2_focal/checkpoints/checkpoint0024.pth pretrained/checkpoint.pth \
    && rm -rf exp checkpoint.zip

COPY . .

ENTRYPOINT ["python3", "run.py"]