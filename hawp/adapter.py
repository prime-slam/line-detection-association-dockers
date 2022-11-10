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
import random
import warnings

from enum import Enum
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from hawp.fsl.config import cfg
from hawp.fsl.model import build_model
from line_dataset import LineDataset, collate

warnings.filterwarnings("ignore")


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
        model_config_path: Path,
        pretrained_model_path: Path,
        device: Device,
    ):
        self.image_path = image_path
        self.lines_path = output_path / lines_output_directory
        self.scores_path = output_path / scores_output_directory
        self.model_config_path = model_config_path
        self.pretrained_model_path = pretrained_model_path
        self.prediction_file_suffix = ".csv"
        self.batch_size = 1  # model returns result for batch_size = 1

        if device == Device.cuda:
            if torch.cuda.is_available():
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
            else:
                print("No available cuda device! Fall back on cpu.")
                device = Device.cpu

        self.device = device.name
        self.__update_configuration()

    def run(self) -> None:
        self.lines_path.mkdir(exist_ok=True)
        self.scores_path.mkdir(exist_ok=True)

        image_loader = self.__create_imageloader()

        model = self.__build_model()
        model.eval()

        with torch.no_grad():
            for image, metadata in tqdm(image_loader):
                meta = metadata[0]
                result, _ = model(*self.__create_model_input(image, meta))

                lines = result["lines_pred"].cpu().numpy()
                scores = result["lines_score"].cpu().numpy()

                lines = self.__postprocess_lines(lines, meta)

                self.__save_results(
                    file_name=meta["image_name"],
                    lines=lines,
                    scores=scores,
                )

    def __update_configuration(self) -> None:
        cfg.merge_from_file(self.model_config_path)

    def __create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path)),
            batch_size=self.batch_size,
            collate_fn=collate,
            pin_memory=True,
        )

    def __save_results(
        self, file_name: str, lines: np.ndarray, scores: np.ndarray
    ) -> None:
        np.savetxt(
            (self.lines_path / file_name).with_suffix(self.prediction_file_suffix),
            lines,
            delimiter=",",
        )
        np.savetxt(
            (self.scores_path / file_name).with_suffix(self.prediction_file_suffix),
            scores,
            delimiter=",",
        )

    def __build_model(self) -> torch.nn.Module:
        model = build_model(cfg)
        model.to(self.device)

        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])

        return model

    def __create_model_input(self, image: torch.Tensor, meta: Dict):
        return image.to(self.device), [
            {
                "width": cfg.DATASETS.IMAGE.WIDTH,
                "height": cfg.DATASETS.IMAGE.HEIGHT,
                "filename": meta["image_name"],
            }
        ]

    def __postprocess_lines(self, lines: np.ndarray, metadata: Dict) -> np.ndarray:
        width = metadata["width"]
        height = metadata["height"]

        # rescale
        x_scale = width / cfg.DATASETS.IMAGE.WIDTH
        y_scale = height / cfg.DATASETS.IMAGE.HEIGHT

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return lines
