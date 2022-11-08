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
from typing import Tuple
from torch.utils.data import default_collate, Dataset
from torchvision.transforms import functional as F

from hawp.fsl.config import cfg


def collate(batch):
    return (default_collate([elem[0] for elem in batch]), [elem[1] for elem in batch])


class LineDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        image_rescale: Tuple[int, int] = (
            cfg.DATASETS.IMAGE.HEIGHT,
            cfg.DATASETS.IMAGE.WIDTH,
        ),
    ):
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
        return transformed_image, metadata

    def __transform_image(self, image: np.ndarray) -> torch.Tensor:
        transformed = image.astype("float32")[:, :, :3]
        transformed = cv2.resize(transformed, self.image_rescale)
        transformed = F.to_tensor(transformed)

        if not cfg.DATASETS.IMAGE.TO_255:
            transformed /= 255.0

        transformed = F.normalize(
            transformed,
            mean=cfg.DATASETS.IMAGE.PIXEL_MEAN,
            std=cfg.DATASETS.IMAGE.PIXEL_STD,
        )

        return transformed
