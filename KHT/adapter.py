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
from kht import kht


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

        # same threshold as in python/example/main.py
        self.canny_first_threshold = 80
        self.canny_second_threshold = 200

    def _predict(self, model, image: np.ndarray):
        return kht(image)

    def _create_imageloader(self) -> LineDataset:
        return LineDataset(self.image_path, self._transform_image)

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        transformed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        transformed = cv2.Canny(
            transformed, self.canny_first_threshold, self.canny_second_threshold
        )

        return transformed

    def _build_model(self):
        pass

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: ImageMetadata
    ) -> List[Prediction]:
        # lines are sorted in order of their importance
        scores = np.linspace(1, 0, len(raw_predictions))

        width = metadata.width
        height = metadata.height
        lines = []

        for (rho, theta) in raw_predictions:
            theta = np.radians(theta)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)

            # Convert from KHT to Matplotlib's coordinate system conventions.
            # The KHT implementation assumes row-major memory alignment for
            # images. Also, it assumes that the origin of the image coordinate
            # system is at the center of the image, with the x-axis growing to
            # the right and the y-axis growing down.
            if sin_theta != 0:
                x = (-width / 2, width / 2 - 1)
                y = (
                    (rho - x[0] * cos_theta) / sin_theta,
                    (rho - x[1] * cos_theta) / sin_theta,
                )
            else:
                x = (rho, rho)
                y = (-height / 2, height / 2 - 1)
            x1 = x[0] + width / 2
            x2 = x[1] + width / 2
            y1 = y[0] + height / 2
            y2 = y[1] + height / 2
            lines.append([x1, y1, x2, y2])

        lines = np.array(lines)

        return [Prediction(lines=lines, scores=scores, metadata=metadata)]
