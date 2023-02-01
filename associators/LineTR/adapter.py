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
from typing import Any, Dict, Optional, Tuple
from torch.utils.data.dataloader import default_collate

from common.adapter.torch_adapter import TorchAdapter
from common.dataset.collate import collate
from common.device import Device
from common.dataset.frame_pairs_dataset import FramePairsDataset
from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata
from common.prediction import Prediction
from models.line_transformer import LineTransformer
from models.matching import Matching
from models.superpoint import SuperPoint


class Adapter(TorchAdapter):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        association_scores_dir: Optional[str],
        output_path: Path,
        pairs_file: Path,
        frames_step: int,
        config_path: Path,
        device: Device,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            association_scores_dir,
            output_path,
            device,
        )
        with open(config_path, "r") as conf:
            self.config = yaml.safe_load(conf)
        self.superpoint = SuperPoint(self.config["superpoint"]).eval().to(self.device)
        self.frames_step = frames_step
        self.line_transformer = (
            LineTransformer(self.config["linetransformer"]).eval().to(self.device)
        )
        self.pairs_file = pairs_file

    def _create_frame_pairs_loader(self):
        return torch.utils.data.DataLoader(
            FramePairsDataset(
                self.images_path,
                self.lines_path,
                transform_frames_pair=self._transform_frames_pair,
                frames_step=self.frames_step,
                pairs_file=self.pairs_file,
            ),
            batch_size=1,
            collate_fn=collate,
            pin_memory=True,
        )

    def _transform_frames_pair(self, pair: FramesPair):
        return pair.transform(self.__transform_image, self.__transform_lines)

    def _build_model(self):
        return Matching(self.config).eval().to(self.device)

    def _postprocess_prediction(
        self, raw_predictions: Any, metadata: Tuple[ImageMetadata, ImageMetadata]
    ) -> Prediction:
        raw_matches = raw_predictions["matches_l"].cpu().numpy().squeeze()
        associations = np.column_stack(np.where(raw_matches > 0))
        scores = (
            raw_predictions["matching_scores_l"]
            .cpu()
            .numpy()
            .squeeze()[raw_matches != 0]
        )
        return Prediction(
            associations=associations, pair_metadata=metadata, scores=scores
        )

    def _predict(self, model, frames_pair: FramesPair):
        input_data = {}

        for frame in [0, 1]:
            lines = frames_pair.lines_pair[frame]
            image = frames_pair.images_pair[frame]
            input_data.update(
                self.__create_frame_data(lines, image, frame_number=frame)
            )

        return model(input_data)

    def __transform_image(self, image: np.ndarray):
        transformed_image = cv2.resize(
            image, (self.config["resize"]["width"], self.config["resize"]["height"])
        )
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(transformed_image).float()[None, None].to(self.device)

    def __transform_lines(self, lines: np.ndarray):
        return lines

    def __create_frame_data(self, lines, image, frame_number):
        def add_key_suffix(data: Dict, suffix: str):
            return {key + suffix: value for key, value in data.items()}

        suffix = str(frame_number)
        data = {}
        support_points_data = self.superpoint({"image": image})
        data.update(add_key_suffix(support_points_data, suffix=suffix))
        valid_mask = torch.ones_like(image).to(image)
        data["valid_mask" + suffix] = valid_mask

        image_shape = image.shape
        lines_data = self.line_transformer.preprocess(
            lines, image_shape, support_points_data, valid_mask
        )
        lines_data = self.line_transformer(lines_data)
        data.update(add_key_suffix(lines_data, suffix=suffix))
        data["image" + suffix] = image

        return data
