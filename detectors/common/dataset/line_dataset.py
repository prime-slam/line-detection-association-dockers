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

from pathlib import Path
from skimage import io
from typing import Callable

from common.image_metadata import ImageMetadata


class LineDataset:
    def __init__(
        self,
        data_path: Path,
        transform_image: Callable,
    ):
        files = sorted(data_path.iterdir())
        self.files = files
        self.transform_image = transform_image

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        image_name = path.stem
        image = io.imread(path)

        height, width = image.shape[0], image.shape[1]
        metadata = ImageMetadata(width=width, height=height, image_name=image_name)

        return self.transform_image(image), metadata
