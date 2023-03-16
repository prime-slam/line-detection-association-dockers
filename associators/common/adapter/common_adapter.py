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

from abc import ABC
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from common.adapter.adapter_base import AdapterBase


class CommonAdapter(AdapterBase, ABC):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        output_path: Path,
        association_scores_dir: Optional[str] = None,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            output_path,
            association_scores_dir
        )

    def run(self) -> None:
        image_loader = self._create_frame_pairs_loader()
        model = self._build_model()

        for image, metadata in tqdm(image_loader):
            raw_predictions = self._predict(model, image)
            prediction = self._postprocess_prediction(raw_predictions, metadata)
            self._save_prediction(prediction)
