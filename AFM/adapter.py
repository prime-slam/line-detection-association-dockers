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
from torch.autograd import Variable
from typing import List, Any

from common.adapter.torch_adapter import TorchAdapter
from common.device import Device
from common.image_metadata import ImageMetadata
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction
from config import cfg
from lib.squeeze_to_lsg import lsgenerator
from modeling.net import build_network


def collate_tensor_fn(batch):
    return torch.stack(batch, 0)


def collate(batch):
    return collate_tensor_fn([elem[0] for elem in batch]), [elem[1] for elem in batch]


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
        cfg.merge_from_file(model_config_path)
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = 1
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.model_input_size = (cfg.INPUT.IN_RES, cfg.INPUT.IN_RES)
        self.model_output_size = (cfg.INPUT.OUT_RES, cfg.INPUT.OUT_RES)

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model(Variable(image))

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = cv2.resize(image, self.model_input_size) / 255.0
        transformed = (transformed - self.image_mean) / self.image_std
        transformed = np.transpose(transformed, (2, 0, 1))
        return torch.from_numpy(transformed).float()

    def _build_model(self) -> torch.nn.Module:
        model = build_network(cfg)
        model.load_state_dict(
            torch.load(self.pretrained_model_path, map_location=self.device)
        )

        model = model.to(self.device)
        model.eval()
        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        metadata = metadata[0]
        predictions, _, _ = lsgenerator(raw_predictions[0].cpu().data.numpy())
        lines, scores = predictions[:, :-1], predictions[:, -1]
        line_lengths = np.sqrt(((lines[:, :2] - lines[:, 2:]) ** 2).sum(-1))
        scores = np.reciprocal(scores / line_lengths)
        scores /= scores.max()

        output_height, output_width = self.model_output_size

        x_scale = metadata.width / output_width
        y_scale = metadata.height / output_height

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return [Prediction(lines=lines, scores=scores, metadata=metadata)]
