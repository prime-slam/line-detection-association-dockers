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
from typing import Any, List, Optional

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
        scores_output_directory: Optional[Path],
    ):
        super().__init__(
            image_path,
            output_path,
            lines_output_directory,
            scores_output_directory,
        )

    def _predict(self, model, image: np.ndarray):
        return model.detect(image)

    def _create_imageloader(self) -> LineDataset:
        return LineDataset(Path(self.image_path), self._transform_image)

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        transformed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return transformed

    def _build_model(self):
        return cv2.ximgproc.createFastLineDetector()

    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: ImageMetadata
    ) -> List[Prediction]:
        lines = raw_predictions.squeeze()

        return [Prediction(lines=np.array(lines), scores=None, metadata=metadata)]
