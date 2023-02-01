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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from common.prediction import Prediction


class AdapterBase(ABC):
    """Base adapter for deep learning line detector"""

    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Optional[Path],
    ):
        self.image_path = image_path
        self.lines_path = output_path / lines_output_directory
        self.scores_path = (
            output_path / scores_output_directory if scores_output_directory else None
        )
        self.prediction_file_suffix = ".csv"

    def _save_prediction(self, prediction: Prediction) -> None:
        self.lines_path.mkdir(parents=True, exist_ok=True)
        file_name = prediction.metadata.image_name + self.prediction_file_suffix
        np.savetxt(
            self.lines_path / file_name,
            prediction.lines,
            delimiter=",",
        )
        if self.scores_path:
            self.scores_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                self.scores_path / file_name,
                prediction.scores,
                delimiter=",",
            )

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def _predict(self, model, image):
        pass

    @abstractmethod
    def _create_imageloader(self):
        pass

    @abstractmethod
    def _transform_image(self, image: np.ndarray):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: Dict
    ) -> List[Prediction]:
        pass
