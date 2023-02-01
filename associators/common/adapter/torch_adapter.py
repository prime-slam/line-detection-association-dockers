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

import numpy as np
import random
import torch

from abc import ABC
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from common.adapter.adapter_base import AdapterBase
from common.device import Device


class TorchAdapter(AdapterBase, ABC):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        association_scores_dir: Optional[str],
        output_path: Path,
        device: Device,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            association_scores_dir,
            output_path,
        )

        device_name = "cpu"
        if device == Device.gpu:
            if torch.cuda.is_available():
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                device_name = "cuda"
            else:
                print("No cuda device available! Fall back on cpu.")

        self.device = device_name

    def run(self) -> None:
        image_loader = self._create_frame_pairs_loader()
        model = self._build_model()

        with torch.no_grad():
            for frame_pairs_batch, metadata_batch in tqdm(image_loader):
                for frame_pairs, metadata in zip(frame_pairs_batch, metadata_batch):
                    raw_predictions = self._predict(model, frame_pairs)
                    prediction = self._postprocess_prediction(raw_predictions, metadata)
                    self._save_prediction(prediction)