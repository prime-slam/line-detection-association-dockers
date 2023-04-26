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
from skimage import io
from typing import Callable, Optional

from pathlib import Path

from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata


class FramePairsDataset:
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        transform_frames_pair: Callable,
        frames_step: int = 1,
        pairs_file: Optional[Path] = None,
    ):
        self.image_files = sorted(images_path.iterdir())
        self.lines_files = sorted(lines_path.iterdir())
        images_number = len(self.image_files)

        if len(self.image_files) != len(self.lines_files):
            raise ValueError(
                "The number of image files must be equal to the number of line files"
            )
        self.pairs = (
            np.genfromtxt(pairs_file)
            if pairs_file
            else list(zip(range(images_number), range(frames_step, images_number)))
        )
        self.size = len(self.pairs)
        self.transform_frames_pair = transform_frames_pair
        self.__csv_delimiter = ","

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        first_frame, second_frame = self.pairs[idx]

        first_image_file = self.image_files[first_frame]
        first_image = io.imread(first_image_file)
        second_image_file = self.image_files[second_frame]
        second_image = io.imread(second_image_file)

        first_lines_file = self.lines_files[first_frame]
        second_lines_file = self.lines_files[second_frame]

        first_lines = np.genfromtxt(first_lines_file, delimiter=self.__csv_delimiter)
        second_lines = np.genfromtxt(second_lines_file, delimiter=self.__csv_delimiter)

        first_image_metadata = ImageMetadata(
            width=first_image.shape[1],
            height=first_image.shape[0],
            image_name=first_image_file.stem,
        )

        second_image_metadata = ImageMetadata(
            width=second_image.shape[1],
            height=second_image.shape[0],
            image_name=second_image_file.stem,
        )

        frame_pair = FramesPair(
            images_pair=(first_image, second_image),
            images_metadata_pair=(first_image_metadata, second_image_metadata),
            lines_pair=(first_lines, second_lines),
        )

        return self.transform_frames_pair(frame_pair)
