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
import torch

from pathlib import Path
from skimage import io
from torch.utils.data import default_collate, Dataset
from typing import Tuple

from lcnn.config import M


def collate(batch):
    return (default_collate([elem[0] for elem in batch]), [elem[1] for elem in batch])


class LineDataset(Dataset):
    def __init__(self, data_path: Path, image_rescale: Tuple[int, int] = (512, 512)):
        files = sorted(data_path.iterdir())
        self.files = files
        self.image_rescale = image_rescale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        path = self.files[idx]
        image_name = path.stem
        image = io.imread(path)

        height, width = image.shape[0], image.shape[1]
        metadata = {"width": width, "height": height, "image_name": image_name}

        transformed_image = self.__transform_image(image)
        return torch.from_numpy(transformed_image).float(), metadata

    def __transform_image(self, image: np.ndarray) -> np.ndarray:
        transformed_image = cv2.resize(image, self.image_rescale)
        transformed_image = transformed_image.astype(float)[:, :, :3]

        # normalize image
        transformed_image = (transformed_image - M.image.mean) / M.image.stddev
        transformed_image = np.rollaxis(transformed_image, 2)

        return transformed_image
