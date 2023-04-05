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

import cv2
import numpy as np
import torch
import yaml

from pathlib import Path
from typing import List, Any, Optional

from common.adapter.torch_adapter import TorchAdapter
from common.device import Device
from common.image_metadata import ImageMetadata
from common.dataset.collate import collate
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction
from deeplsd.models.deeplsd import DeepLSD


class Adapter(TorchAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Optional[Path],
        config_path: Path,
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
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size
        with open(config_path, "r") as conf:
            self.config = yaml.safe_load(conf)

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model({"image": image})

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            collate_fn=collate,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0

        return torch.tensor(transformed_image, dtype=torch.float, device=self.device)[
            None
        ]

    def _build_model(self) -> torch.nn.Module:
        ckpt = torch.load(self.pretrained_model_path, map_location="cpu")
        model = DeepLSD(self.config)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        lines_batch = raw_predictions["lines"]

        postprocessed_predictions = [
            Prediction(lines=lines.reshape(-1, 4), scores=None, metadata=meta)
            for lines, meta in zip(lines_batch, metadata)
        ]

        return postprocessed_predictions
