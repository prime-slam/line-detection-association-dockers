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
import torch

from enum import Enum
from pathlib import Path
from tqdm import tqdm

from dataset.line_dataset import LineDataset, collate
from model import ModelConfig
from utils.reconstruct import TPS_line
from utils.utils import load_model


class Device(Enum):
    cuda = 0
    cpu = 1


class Adapter:
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        pretrained_model_path: Path,
        model_config: ModelConfig,
        device: Device,
        batch_size: int,
    ):
        self.image_path = image_path
        self.lines_path = output_path.joinpath(lines_output_directory)
        self.scores_path = output_path.joinpath(scores_output_directory)
        self.batch_size = batch_size
        self.pretrained_model_path = pretrained_model_path
        self.prediction_file_suffix = ".csv"
        self.model_config = model_config

        if device == Device.cuda:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(0)
            else:
                print("No available cuda device! Fall back on cpu.")
                device = Device.cpu

        self.device = device.name

    def run(self) -> None:
        self.lines_path.mkdir(exist_ok=True)
        self.scores_path.mkdir(exist_ok=True)

        model = load_model(
            self.model_config.model(self.model_config.head),
            self.pretrained_model_path,
            resume=False,
            selftrain=False,
        )

        model = model.cuda()
        model.eval()

        image_loader = self.__create_imageloader()

        with torch.no_grad():
            for image, metadata in tqdm(image_loader):
                wrapped_results = model(image.cuda())[-1]
                results = self.__unwrap_results(wrapped_results)
                for result, meta in zip(results, metadata):
                    lines, scores = self.__get_predictions(result, meta)
                    self.__save_results(
                        file_name=f"{meta['image_name']}.csv",
                        lines=lines,
                        scores=scores,
                    )

    def __get_predictions(self, model_output, meta):
        heatmap_height, heatmap_width = self.model_config.output_resolution
        lines, _, _, pos, _ = TPS_line(
            model_output, 0.0, 0.5, heatmap_height, heatmap_width
        )
        center = model_output["center"][0][0].detach().cpu().numpy()
        pos_mat = pos.astype(int)
        scores = center[pos_mat[:, 1], pos_mat[:, 0]].tolist() if pos_mat.any() else []

        x_scale = meta["width"] / heatmap_height
        y_scale = meta["height"] / heatmap_width

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return lines, scores

    def __save_results(
        self, file_name: str, lines: np.ndarray, scores: np.ndarray
    ) -> None:
        np.savetxt(
            self.lines_path.joinpath(file_name).with_suffix(
                self.prediction_file_suffix
            ),
            lines,
            delimiter=",",
        )
        np.savetxt(
            self.scores_path.joinpath(file_name).with_suffix(
                self.prediction_file_suffix
            ),
            scores,
            delimiter=",",
        )

    def __create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(self.image_path, self.model_config.input_resolution),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate,
        )

    @staticmethod
    def __unwrap_results(wrapped_results):
        batch_size = wrapped_results["line"].shape[0]
        return [
            dict((k, v[i][None, :]) for k, v in wrapped_results.items())
            for i in range(batch_size)
        ]
