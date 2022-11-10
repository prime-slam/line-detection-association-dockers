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
import warnings

from enum import Enum
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from lcnn.config import C, M
from lcnn.line_dataset import LineDataset, collate
from lcnn.models import hg
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner

# ignore pytorch internal warnings
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
        self.heatmap_size = 128

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

                results = self.__unwrap_results(
                    model(self.__create_model_input(image))["preds"]
                )

                for result, meta in zip(results, metadata):
                    lines = result["lines"]
                    scores = result["score"]

                    lines, scores = self.__postprocess_predictions(lines, scores, meta)

                    self.__save_results(
                        file_name=meta["image_name"],
                        lines=lines,
                        scores=scores,
                    )

    def __update_configuration(self) -> None:
        C.update(C.from_yaml(filename=self.model_config_path))
        M.update(C.model)

    def __create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(Path(self.image_path)),
            batch_size=1,
            collate_fn=collate,
            num_workers=C.io.num_workers,
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
        model = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )

        checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
        model = MultitaskLearner(model)
        model = LineVectorizer(model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        return model

    def __create_model_input(self, image: torch.Tensor):
        return {
            "image": image.to(self.device),
            "mode": "testing",
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(self.device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
            },
        }

    def __postprocess_predictions(
        self, lines: np.ndarray, scores: np.ndarray, metadata: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        width = metadata["width"]
        height = metadata["height"]

        # rescale: it was predicted on a 128 x 128 heatmap
        x_scale = width / self.heatmap_size
        y_scale = height / self.heatmap_size

        x_index = 1
        y_index = 0

        # [[y1, x1], [y2, x2]]
        lines[:, :, x_index] *= x_scale
        lines[:, :, y_index] *= y_scale

        # slice unique lines
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # reformat: [[y1, x1], [y2, x2]] -> [x1, y1, x2, y2]
        lines = lines[:, :, ::-1].flatten().reshape((-1, 4))

        return lines, scores

    @staticmethod
    def __unwrap_results(
        wrapped_results: Dict[str, torch.Tensor]
    ) -> List[Dict[str, np.ndarray]]:
        batch_size = wrapped_results["lines"].shape[0]
        return [
            dict(
                (prediction_name, predictions[i].cpu().numpy())
                for prediction_name, predictions in wrapped_results.items()
            )
            for i in range(batch_size)
        ]
