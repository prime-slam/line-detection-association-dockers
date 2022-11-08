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

# TODO: possible solutions: 1. change daemon.json; 2.manually install in docker contained and then commit
# https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0

# install python requirements
COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cd modeling/DCNv2 && python3 setup.py build develop

ENTRYPOINT ["python3", "run.py"]