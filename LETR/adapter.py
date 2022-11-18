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

from pathlib import Path
from torchvision.transforms import functional as F
from torch.nn.functional import softmax
from typing import List, Any

from common.adapter_base import DLAdapterBase
from common.device import Device
from common.image_metadata import ImageMetadata
from common.line_dataset import LineDataset, collate
from common.prediction import Prediction
from LETR.src.models import build_model
from LETR.src.util.misc import nested_tensor_from_tensor_list


class Adapter(DLAdapterBase):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
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
        self.pretrained_model_path = pretrained_model_path
        self.model_input_size = 512
        self.image_mean = [0.538, 0.494, 0.453]
        self.image_std = [0.257, 0.263, 0.273]
        self.batch_size = 1

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        inputs = nested_tensor_from_tensor_list([torch.squeeze(image)])
        return model(inputs)[0]

    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path), self._transform_image),
            batch_size=self.batch_size,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = F.to_tensor(image)
        transformed = F.resize(transformed, [self.model_input_size])
        transformed = F.normalize(transformed, self.image_mean, self.image_std)

        return transformed

    def _build_model(self) -> torch.nn.Module:
        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)

        args = checkpoint["args"]
        model, _, postprocessors = build_model(args)
        model.load_state_dict(checkpoint["model"])

        model = model.to(self.device)
        model.eval()

        return model

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: List[ImageMetadata]
    ) -> List[Prediction]:
        metadata = metadata[0]

        logits, lines = raw_predictions["pred_logits"], raw_predictions["pred_lines"]
        prob = softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        scores = scores.cpu().numpy()
        lines = lines.cpu().numpy().flatten().reshape(-1, 4)

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= metadata.width
        lines[:, y_index] *= metadata.height

        return [Prediction(lines=lines, scores=scores, metadata=metadata)]
