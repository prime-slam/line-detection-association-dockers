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
from lcnn.config import C, M
from lcnn.models import hg
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner


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
        self.heatmap_size = 128
        self.model_input_size = (512, 512)
        self.__update_configuration()

    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        return model(self.__create_model_input(image))["preds"]

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
        model = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )

        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        model = MultitaskLearner(model)
        model = LineVectorizer(model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
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

            # rescale: it was predicted on a 128 x 128 heatmap
            x_scale = width / self.heatmap_size
            y_scale = height / self.heatmap_size

            x_index = 1
            y_index = 0

            # [[y1, x1], [y2, x2]]
            lines[:, :, x_index] *= x_scale
            lines[:, :, y_index] *= y_scale

            # slice unique lines
            for i in range(1, len(lines)):
                if (lines[i] == lines[0]).all():
                    lines = lines[:i]
                    scores = scores[:i]
                    break

            # reformat: [[y1, x1], [y2, x2]] -> [x1, y1, x2, y2]
            lines = lines[:, :, ::-1].flatten().reshape((-1, 4))
            postprocessed_predictions.append(
                Prediction(lines=lines, scores=scores, metadata=meta)
            )

        return postprocessed_predictions

    def __update_configuration(self) -> None:
        C.update(C.from_yaml(filename=self.model_config_path))
        M.update(C.model)

    def __create_model_input(self, image: torch.Tensor):
        return {
            "image": image.to(self.device),
            "mode": "testing",
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(self.device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
            },
        }
