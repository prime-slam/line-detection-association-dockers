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

import cv2
import numpy as np
import yaml

from pathlib import Path
from skimage.feature import match_descriptors
from tensorpack.tfutils.varmanip import get_checkpoint_path
from typing import Any, Tuple

from cnn import config
from cnn.main import Model
from cnn.misc import get_predictor
from cnn.modules.geometry import Segment2D
from cnn.modules.lineprocessor import LineProcessor
from common.adapter.tensorflow_adapter import TensorflowAdapter
from common.dataset.frame_pairs_dataset import FramePairsDataset
from common.device import Device
from common.frames_pair import FramesPair
from common.image_metadata import ImageMetadata
from common.prediction import Prediction


class Adapter(TensorflowAdapter):
    def __init__(
        self,
        images_path: Path,
        lines_path: Path,
        associations_dir: str,
        output_path: Path,
        pairs_file: Path,
        frames_step: int,
        config_path: Path,
        model_path: Path,
        device: Device,
    ):
        super().__init__(
            images_path,
            lines_path,
            associations_dir,
            output_path,
            device,
        )

        with open(config_path, "r") as conf:
            config.set(**yaml.safe_load(conf))
        self.frames_step = frames_step
        self.pairs_file = pairs_file
        self.model_path = model_path

    def _create_frame_pairs_loader(self):
        return FramePairsDataset(
                self.images_path,
                self.lines_path,
                transform_frames_pair=self._transform_frames_pair,
                frames_step=self.frames_step,
                pairs_file=self.pairs_file,
            )

    def _transform_frames_pair(self, pair: FramesPair):
        return pair

    def _build_model(self):
        model = Model(config.depth)
        checkpoint_path = get_checkpoint_path(str(self.model_path))
        model = get_predictor(checkpoint_path, model)
        return model

    def _postprocess_prediction(
        self, raw_predictions: Any, metadata: Tuple[ImageMetadata, ImageMetadata]
    ) -> Prediction:
        associations = raw_predictions
        return Prediction(associations=associations, pair_metadata=metadata)

    def _predict(self, model, frames_pair: FramesPair):
        first_image, second_image = frames_pair.images_pair
        first_lines, second_lines = frames_pair.lines_pair

        first_descriptors = self.__create_descriptors(model, first_lines, first_image)
        second_descriptors = self.__create_descriptors(model, second_lines, second_image)

        return match_descriptors(first_descriptors, second_descriptors)

    def __create_descriptors(self, model, lines, image):
        (batch_cutouts, batch_wavelets, batch_heights) = self.__create_frame_data(lines, image)
        descriptors, _ = model(batch_cutouts, batch_heights, batch_wavelets)
        return descriptors

    def __create_frame_data(self, lines, image, cutout_width: int = 27, cutout_height: int = 100):
        keylines = []
        img_max_size = max(image.shape)
        processor = LineProcessor()
        for class_id, (x1, y1, x2, y2) in enumerate(lines):
            seg = Segment2D([x1, y1], [x2, y2])
            kl = seg.to_keyline(class_id, img_max_size)
            keylines.append(kl)
        cutouts, wavelet_cutouts = processor.process(image, keylines)

        # convert cutouts from RGB to RGBA
        cutouts = [cv2.cvtColor(cutout, cv2.COLOR_RGB2RGBA) for cutout in cutouts]

        # convert cutouts to float32
        cutouts = [np.float32(cutout) for cutout in cutouts]

        # set alpha channel to specific value
        for cutout in cutouts:
            cutout[:, :, 3] = 127.5

        # merge wavelet cutouts into a single image with multiple channels for each wavelet
        wavelets = []
        for wavelet_cutout in wavelet_cutouts:
            result = np.empty((cutout_height, cutout_width, 0), dtype=np.float32)
            for n in range(len(wavelet_cutout)):
                wavelet = np.float32(wavelet_cutout[n])
                wavelet = wavelet[:, :, np.newaxis]
                result = np.append(result, wavelet, axis=2)
            wavelets.append(result)
        num_cutouts = len(cutouts)
        height_list = [cutout_height] * num_cutouts
        return cutouts, wavelets, height_list
