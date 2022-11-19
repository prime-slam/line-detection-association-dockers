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
from model import ModelConfig
from utils.reconstruct import TPS_line
from utils.utils import load_model


class Adapter(TorchAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        pretrained_model_path: Path,
        model_config: ModelConfig,
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
        self.batch_size = batch_size
        self.pretrained_model_path = pretrained_model_path
        self.model_config = model_config

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model(image)[-1]

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(self.image_path, self._transform_image),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = cv2.resize(image, self.model_config.input_resolution)
        height, width, channels = transformed.shape
        hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        imgv0 = hsv[..., 2]
        imgv = cv2.resize(
            imgv0, (0, 0), fx=1.0 / 4, fy=1.0 / 4, interpolation=cv2.INTER_LINEAR
        )
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv2.resize(imgv, (width, height), interpolation=cv2.INTER_LINEAR)
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        transformed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        transformed = transformed.astype(np.float32) / 255.0
        transformed = transformed.transpose(2, 0, 1)

        return transformed

    def _build_model(self) -> torch.nn.Module:
        model = load_model(
            self.model_config.model(self.model_config.head),
            self.pretrained_model_path,
            resume=False,
            selftrain=False,
        )

        model = model.to(self.device)
        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        predictions = self.__unwrap_predictions(raw_predictions)

        postprocessed_predictions = []
        for prediction, meta in zip(predictions, metadata):
            lines, scores = self.__get_predictions(prediction, meta)
            postprocessed_predictions.append(
                Prediction(lines=lines, scores=scores, metadata=meta)
            )

        return postprocessed_predictions

    def __get_predictions(self, model_output, meta):
        heatmap_height, heatmap_width = self.model_config.output_resolution
        lines, _, _, pos, _ = TPS_line(
            model_output, 0.0, 0.5, heatmap_height, heatmap_width
        )
        center = model_output["center"][0][0].detach().cpu().numpy()
        pos_mat = pos.astype(int)
        scores = center[pos_mat[:, 1], pos_mat[:, 0]].tolist() if pos_mat.any() else []

        x_scale = meta.width / heatmap_height
        y_scale = meta.height / heatmap_width

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return lines, scores

    @staticmethod
    def __unwrap_predictions(wrapped_results):
        batch_size = wrapped_results["line"].shape[0]
        return [
            dict((k, v[i][None, :]) for k, v in wrapped_results.items())
            for i in range(batch_size)
        ]
