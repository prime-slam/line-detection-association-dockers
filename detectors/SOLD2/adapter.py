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

from pathlib import Path
from typing import List, Any, Optional

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from common.adapter.torch_adapter import TorchAdapter
from common.dataset.collate import collate
from common.dataset.line_dataset import LineDataset
from common.device import Device
from common.image_metadata import ImageMetadata
from common.prediction import Prediction


class Adapter(TorchAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Optional[Path],
        device: Device,
        batch_size: int,
    ):
        super().__init__(
            image_path,
            output_path,
            lines_output_directory,
            scores_output_directory,
            device,
        )
        # authors recommend resizing the images to a resolution in the range 400~800 pixels
        self.model_resolution = (600, 600)
        self.batch_size = batch_size

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model(image)

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = cv2.resize(image, self.model_resolution)
        transformed = K.image_to_tensor(transformed).float() / 255.0
        transformed = K.color.rgb_to_grayscale(transformed)

        return transformed

    def _build_model(self) -> torch.nn.Module:
        # config=None => use default config
        model = KF.SOLD2_detector(pretrained=True, config=None)
        model.to(self.device)
        model.eval()
        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        predictions = raw_predictions["line_segments"]

        postprocessed_predictions = []
        for lines, meta in zip(predictions, metadata):
            width = meta.width
            height = meta.height

            model_height, model_width = self.model_resolution

            lines = lines.cpu().numpy()[:, :, ::-1].flatten().reshape(-1, 4)

            x_scale = width / model_width
            y_scale = height / model_height

            x_index = [0, 2]
            y_index = [1, 3]

            lines[:, x_index] *= x_scale
            lines[:, y_index] *= y_scale

            postprocessed_predictions.append(
                Prediction(lines=lines, scores=None, metadata=meta)
            )

        return postprocessed_predictions
