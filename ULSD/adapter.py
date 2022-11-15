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
from torchvision.transforms.transforms import ToTensor, Normalize
from yacs.config import CfgNode

from common.adapter_base import DLAdapterBase
from common.device import Device
from common.image_metadata import ImageMetadata
from common.line_dataset import LineDataset, collate
from common.prediction import Prediction
from network.lcnn import LCNN


class Adapter(DLAdapterBase):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
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
        self.model_config = self.__load_model_config(model_config_path)
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size
        self.heatmap_size = self.model_config.heatmap_size
        self.model_input_size = self.model_config.image_size

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        prediction = model(image)
        return prediction

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, self.model_input_size)
        image = ToTensor()(image)
        image = Normalize(self.model_config.mean, self.model_config.std)(image)

        return image

    def _build_model(self) -> torch.nn.Module:
        model = LCNN(self.model_config)
        model.load_state_dict(
            torch.load(str(self.pretrained_model_path), map_location=self.device)
        )
        model.to(self.device)

        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        _, _, lines, scores = raw_predictions

        predictions = [
            Prediction(lines.cpu().numpy(), scores.cpu().numpy(), meta)
            for lines, scores, meta in zip(lines, scores, metadata)
        ]

        for prediction in predictions:
            width = prediction.metadata.width
            height = prediction.metadata.height

            heatmap_width = self.heatmap_size[1]
            heatmap_height = self.heatmap_size[0]

            # rescale
            x_scale = width / heatmap_width
            y_scale = height / heatmap_height

            prediction.lines = prediction.lines.flatten().reshape((-1, 4))

            x_index = [0, 2]
            y_index = [1, 3]

            prediction.lines[:, x_index] *= x_scale
            prediction.lines[:, y_index] *= y_scale

        return predictions

    @staticmethod
    def __load_model_config(model_config_path):
        with open(model_config_path) as config_file:
            cfg = CfgNode.load_cfg(config_file)
        return cfg
