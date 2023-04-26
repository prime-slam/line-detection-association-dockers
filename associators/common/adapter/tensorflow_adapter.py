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

import tensorflow as tf

from abc import ABC
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from common.adapter.adapter_base import AdapterBase
from common.device import Device


class TensorflowAdapter(AdapterBase, ABC):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        output_path: Path,
        device: Device,
        association_scores_dir: Optional[str] = None,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            output_path,
            association_scores_dir,
        )
        device_number = 0

        device_name = f"/CPU:{device_number}"
        if device == Device.gpu:
            if tf.config.list_physical_devices("GPU"):
                device_name = f"/GPU:{device_number}"
            else:
                print("No gpu device available! Fall back on cpu.")

        self.device = device_name

    def run(self) -> None:
        frame_pairs_loader = self._create_frame_pairs_loader()
        model = self._build_model()

        with tf.device(self.device):
            for frame_pair in tqdm(frame_pairs_loader):
                raw_predictions = self._predict(model, frame_pair)
                prediction = self._postprocess_prediction(raw_predictions, frame_pair.images_metadata_pair)
                self._save_prediction(prediction)
