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

from pathlib import Path
from typing import Any, List

from common.adapter.common_adapter import CommonAdapter
from common.image_metadata import ImageMetadata
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction


class Adapter(CommonAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
    ):
        super().__init__(
            image_path,
            output_path,
            lines_output_directory,
            scores_output_directory,
        )
        self.canny_first_threshold = 50
        self.canny_second_threshold = 200
        self.canny_kernel_size = 3

        self.rho = 1
        self.theta = np.pi / 180
        self.hough_threshold = 80
        self.min_line_length = 10
        self.max_line_gap = 10

    def _predict(self, model, image: np.ndarray):
        return cv2.HoughLinesP(
            image,
            rho=self.rho,
            theta=self.theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

    def _create_imageloader(self) -> LineDataset:
        return LineDataset(Path(self.image_path), self._transform_image)

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        transformed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        transformed = cv2.Canny(
            transformed,
            self.canny_first_threshold,
            self.canny_second_threshold,
            self.canny_kernel_size,
        )
        return transformed

    def _build_model(self):
        pass

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: ImageMetadata
    ) -> List[Prediction]:
        lines = raw_predictions.flatten().reshape(-1, 4)

        return [Prediction(lines=lines, scores=None, metadata=metadata)]
