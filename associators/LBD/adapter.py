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
import pytlbd

from pathlib import Path
from typing import Any, Tuple

from skimage.feature import match_descriptors

from common.adapter.common_adapter import CommonAdapter
from common.dataset.frame_pairs_dataset import FramePairsDataset
from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata
from common.prediction import Prediction


class Adapter(CommonAdapter):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        output_path: Path,
        pairs_file: Path,
        frames_step: int,
    ):
        super().__init__(images_path, lines_path, associations_dir, output_path)
        self.frames_step = frames_step
        self.pairs_file = pairs_file

        # values recommended by the authors
        self.bands_number = 9
        self.band_width = 7

    def _predict(self, model, frames_pair: FramesPair):
        first_image, second_image = frames_pair.images_pair
        first_lines, second_lines = frames_pair.lines_pair

        first_descriptors = pytlbd.lbd_single_scale(
            first_image, first_lines, self.bands_number, self.band_width
        )
        second_descriptors = pytlbd.lbd_single_scale(
            second_image, second_lines, self.bands_number, self.band_width
        )

        return match_descriptors(first_descriptors, second_descriptors)

    def _create_frame_pairs_loader(self):
        return FramePairsDataset(
            self.images_path,
            self.lines_path,
            transform_frames_pair=self._transform_frames_pair,
            frames_step=self.frames_step,
            pairs_file=self.pairs_file,
        )

    def _transform_frames_pair(self, pair: FramesPair):
        return pair.transform(
            lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), lambda lines: lines
        )

    def _build_model(self):
        pass

    def _postprocess_prediction(
        self, raw_predictions: Any, metadata: Tuple[ImageMetadata, ImageMetadata]
    ) -> Prediction:
        associations = raw_predictions
        return Prediction(associations=associations, pair_metadata=metadata)
