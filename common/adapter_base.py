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
import random
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

from common.device import Device
from common.prediction import Prediction


class DLAdapterBase(ABC):
    """Base adapter for deep learning line detector"""

    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        device: Device,
    ):
        self.image_path = image_path
        self.lines_path = output_path / lines_output_directory
        self.scores_path = output_path / scores_output_directory
        self.prediction_file_suffix = ".csv"

        if device == Device.cuda:
            if torch.cuda.is_available():
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
            else:
                print("No available cuda device! Fall back on cpu.")
                device = Device.cpu

        self.device = device.name

    def run(self) -> None:
        self.lines_path.mkdir(exist_ok=True)
        self.scores_path.mkdir(exist_ok=True)

        image_loader = self._create_imageloader()
        model = self._build_model()

        with torch.no_grad():
            for image, metadata in tqdm(image_loader):
                raw_predictions = self._predict(model, image.to(self.device))
                predictions = self._postprocess_predictions(raw_predictions, metadata)

                for prediction in predictions:
                    self.__save_prediction(prediction)

    def __save_prediction(self, prediction: Prediction) -> None:
        file_name = prediction.metadata.image_name
        np.savetxt(
            self.lines_path.joinpath(file_name).with_suffix(
                self.prediction_file_suffix
            ),
            prediction.lines,
            delimiter=",",
        )
        np.savetxt(
            self.scores_path.joinpath(file_name).with_suffix(
                self.prediction_file_suffix
            ),
            prediction.scores,
            delimiter=",",
        )

    @abstractmethod
    def _predict(self, model: torch.nn.Module, image: torch.Tensor):
        pass

    @abstractmethod
    def _create_imageloader(self) -> torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def _transform_image(self, image: np.ndarray) -> torch.Tensor:
        pass

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def _postprocess_predictions(
        self, raw_predictions: Any, metadata: Dict
    ) -> List[Prediction]:
        pass
