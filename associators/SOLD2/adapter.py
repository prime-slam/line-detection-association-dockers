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

import kornia as K
import kornia.feature as KF
from pathlib import Path
from typing import Any, Optional, Tuple
from torch.utils.data.dataloader import default_collate

from common.adapter.torch_adapter import TorchAdapter
from common.dataset.collate import collate
from common.device import Device
from common.dataset.frame_pairs_dataset import FramePairsDataset
from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata
from common.prediction import Prediction


class Adapter(TorchAdapter):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        output_path: Path,
        pairs_file: Path,
        frames_step: int,
        device: Device,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            output_path,
            device,
        )
        self.frames_step = frames_step
        self.pairs_file = pairs_file
        self.model_resolution = (600, 600)

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
        return KF.SOLD2().eval().to(self.device)

    def _postprocess_prediction(
        self, raw_predictions: Any, metadata: Tuple[ImageMetadata, ImageMetadata]
    ) -> Prediction:
        second_indices = raw_predictions.cpu().numpy()
        first_indices = np.arange(len(second_indices))
        matched = second_indices != -1
        associations = np.column_stack(
            (first_indices[matched], second_indices[matched])
        )

        return Prediction(associations=associations, pair_metadata=metadata)

    def _predict(self, model, frames_pair: FramesPair):
        outputs = model(frames_pair.images_pair)
        first_image_descriptor, second_image_descriptor = outputs["dense_desc"]
        first_lines, second_lines = frames_pair.lines_pair
        return model.match(
            first_lines,
            second_lines,
            first_image_descriptor[None],
            second_image_descriptor[None],
        )

    def __transform_image(self, image: np.ndarray):
        transformed = cv2.resize(image, self.model_resolution)
        transformed = K.image_to_tensor(transformed).float() / 255.0
        transformed = K.color.rgb_to_grayscale(transformed)
        return transformed.to(self.device)

    def __transform_lines(self, lines: np.ndarray):
        return torch.from_numpy(lines.reshape(-1, 2, 2).astype(np.float32)).to(
            self.device
        )
