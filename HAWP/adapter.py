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
from torchvision.transforms import functional as F
from typing import List, Any

from common.adapter.torch_adapter import TorchAdapter
from common.device import Device
from common.image_metadata import ImageMetadata
from common.dataset.collate import collate
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction
from hawp.fsl.config import cfg
from hawp.fsl.model import build_model


class Adapter(TorchAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        model_config_path: Path,
        pretrained_model_path: Path,
        device: Device,
    ):
        super().__init__(
            image_path,
            output_path,
            lines_output_directory,
            scores_output_directory,
            device,
        )
        self.model_config_path = model_config_path
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = 1  # model returns result for batch_size = 1
        self.__update_configuration()

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        predictions, _ = model(*self.__create_model_input(image))
        return predictions

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = image.astype("float32")[:, :, :3]
        transformed = cv2.resize(
            transformed,
            (
                cfg.DATASETS.IMAGE.HEIGHT,
                cfg.DATASETS.IMAGE.WIDTH,
            ),
        )
        transformed = F.to_tensor(transformed)

        if not cfg.DATASETS.IMAGE.TO_255:
            transformed /= 255.0

        transformed = F.normalize(
            transformed,
            mean=cfg.DATASETS.IMAGE.PIXEL_MEAN,
            std=cfg.DATASETS.IMAGE.PIXEL_STD,
        )

        return transformed

    def _build_model(self) -> torch.nn.Module:
        model = build_model(cfg)
        model.to(self.device)

        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        metadata = metadata[0]

        lines = raw_predictions["lines_pred"].cpu().numpy()
        scores = raw_predictions["lines_score"].cpu().numpy()

        width = metadata.width
        height = metadata.height

        # rescale
        x_scale = width / cfg.DATASETS.IMAGE.WIDTH
        y_scale = height / cfg.DATASETS.IMAGE.HEIGHT

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return [Prediction(lines=lines, scores=scores, metadata=metadata)]

    def __update_configuration(self) -> None:
        cfg.merge_from_file(self.model_config_path)

    def __create_model_input(self, image: torch.Tensor):
        return image.to(self.device), [
            {
                "width": cfg.DATASETS.IMAGE.WIDTH,
                "height": cfg.DATASETS.IMAGE.HEIGHT,
                "filename": "",
            }
        ]
