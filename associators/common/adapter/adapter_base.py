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

import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata
from common.prediction import Prediction


class AdapterBase(ABC):
    """Base adapter for line associator"""

    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        association_scores_dir: Optional[str],
        output_path: Path,
    ):
        self.images_path = images_path
        self.lines_path = lines_path
        self.associations_output_path = output_path / associations_dir
        self.association_scores_output_path = (
            output_path / association_scores_dir if association_scores_dir else None
        )
        self.output_path = output_path
        self.prediction_file_suffix = ".csv"

    def _save_prediction(self, prediction: Prediction) -> None:
        self.associations_output_path.mkdir(parents=True, exist_ok=True)
        file_name = (
            "_".join(map(lambda frame: frame.image_name, prediction.pair_metadata))
            + self.prediction_file_suffix
        )
        np.savetxt(
            self.associations_output_path / file_name,
            prediction.associations,
            delimiter=",",
            fmt="%d",
        )
        if self.association_scores_output_path is not None:
            self.association_scores_output_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                self.association_scores_output_path / file_name,
                prediction.scores,
                delimiter=",",
            )

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def _predict(self, model, frames_pair: FramesPair):
        pass

    @abstractmethod
    def _create_frame_pairs_loader(self):
        pass

    @abstractmethod
    def _transform_frames_pair(self, pair: FramesPair):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _postprocess_prediction(
        self, raw_predictions: Any, metadata: Tuple[ImageMetadata, ImageMetadata]
    ) -> Prediction:
        pass
