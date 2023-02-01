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

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict

from modeling.Hourglass import HourglassNet
from modeling.TP_Net import Res320, Res160


@dataclass
class ModelConfig:
    model: type
    input_resolution: Tuple[int, int]
    output_resolution: Tuple[int, int]
    head: Dict = field(default_factory=lambda: {"center": 1, "dis": 4, "line": 1})


class Model(Enum):
    TPLSD = ModelConfig(
        model=Res320, input_resolution=(320, 320), output_resolution=(320, 320)
    )
    TPLSD_Lite = ModelConfig(
        model=Res160, input_resolution=(320, 320), output_resolution=(320, 320)
    )
    TPLSD_512 = ModelConfig(
        model=Res320, input_resolution=(512, 512), output_resolution=(512, 512)
    )
    Hourglass = ModelConfig(
        model=HourglassNet, input_resolution=(512, 512), output_resolution=(128, 128)
    )
