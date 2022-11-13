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

import numpy as np
import torch

from typing import Dict, List


def unwrap_results(
    wrapped_results: Dict[str, torch.Tensor], batch_size: int
) -> List[Dict[str, np.ndarray]]:
    """Split stacked images into list of images"""
    return [
        dict(
            (prediction_name, predictions[i].cpu().numpy())
            for prediction_name, predictions in wrapped_results.items()
        )
        for i in range(batch_size)
    ]
