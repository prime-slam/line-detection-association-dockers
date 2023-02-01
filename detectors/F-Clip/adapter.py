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

import cv2
import numpy as np
import torch

from pathlib import Path
from typing import List, Any

from common.adapter.torch_adapter import TorchAdapter
from common.device import Device
from common.image_metadata import ImageMetadata
from common.dataset.collate import collate
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction
from common.transform import unwrap_results
from FClip.config import C, M
from test import build_model


class Adapter(TorchAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        base_config_path: Path,
        model_config_path: Path,
        pretrained_model_path: Path,
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
        self.base_config_path = base_config_path
        self.model_config_path = model_config_path
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size
        self.__update_configuration()
        self.heatmap_size = M.resolution
        self.model_input_size = (512, 512)

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model(
            {
                "image": image.to(self.device),
            },
            isTest=True,
        )["heatmaps"]

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=1,
            collate_fn=collate,
            num_workers=C.io.num_workers,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed_image = cv2.resize(image, self.model_input_size)
        transformed_image = transformed_image.astype(float)[:, :, :3]

        # normalize image
        transformed_image = (transformed_image - M.image.mean) / M.image.stddev
        transformed_image = np.rollaxis(transformed_image, 2)

        return torch.from_numpy(transformed_image).float()

    def _build_model(self) -> torch.nn.Module:
        model = build_model(self.device == "cpu")
        model.to(self.device)
        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        batch_size = raw_predictions["lines"].shape[0]
        predictions = unwrap_results(raw_predictions, batch_size)

        postprocessed_predictions = []
        for prediction, meta in zip(predictions, metadata):
            lines = prediction["lines"]
            scores = prediction["score"]

            width = meta.width
            height = meta.height

            # reformat: [[y1, x1], [y2, x2]] -> [x1, y1, x2, y2]
            lines = lines[:, :, ::-1].flatten().reshape((-1, 4))

            # rescale: it was predicted on a 128 x 128 heatmap
            x_scale = width / self.heatmap_size
            y_scale = height / self.heatmap_size

            x_index = [0, 2]
            y_index = [1, 3]

            lines[:, x_index] *= x_scale
            lines[:, y_index] *= y_scale

            postprocessed_predictions.append(
                Prediction(lines=lines, scores=scores, metadata=meta)
            )

        return postprocessed_predictions

    def __update_configuration(self) -> None:
        C.update(C.from_yaml(filename=self.base_config_path))
        C.update(C.from_yaml(filename=self.model_config_path))
        M.update(C.model)
        C.io.model_initialize_file = self.pretrained_model_path
