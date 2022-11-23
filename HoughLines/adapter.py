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
        self.hough_threshold = 150

    def _predict(self, model, image: np.ndarray):
        return cv2.HoughLines(image, self.rho, self.theta, self.hough_threshold)

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
        lines = []
        for rho, theta in raw_predictions.reshape(-1, 2):
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            x0 = rho * cos_theta
            y0 = rho * sin_theta
            x1 = int(x0 - 1000 * sin_theta)
            y1 = int(y0 + 1000 * cos_theta)
            x2 = int(x0 + 1000 * sin_theta)
            y2 = int(y0 - 1000 * cos_theta)
            lines.append([x1, y1, x2, y2])

        return [Prediction(lines=np.array(lines), scores=None, metadata=metadata)]
