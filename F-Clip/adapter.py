import warnings
import numpy as np
import torch

from os import path, makedirs
from enum import Enum
from FClip.line_dataset import LineDataset, collate
from FClip.config import C, M
from test import build_model

warnings.filterwarnings("ignore")


class Device(Enum):
    cuda = 0
    cpu = 1


class Adapter:
    def __init__(
        self,
        image_path: str,
        output_path: str,
        lines_output_directory: str,
        scores_output_directory: str,
        base_config_path: str,
        model_config_path: str,
        pretrained_model_path: str,
        device: Device,
        batch_size: int,
    ):
        self.image_path = image_path
        self.lines_path = path.join(output_path, lines_output_directory)
        self.scores_path = path.join(output_path, scores_output_directory)
        self.base_config_path = base_config_path
        self.model_config_path = model_config_path
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size

        if device == Device.cuda:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(0)
            else:
                print("No available cuda device! Fall back on cpu.")
                device = Device.cpu

        self.device = device.name
        self.__update_configuration()

    def run(self) -> None:
        makedirs(self.lines_path, exist_ok=True)
        makedirs(self.scores_path, exist_ok=True)

        image_loader = self.__create_imageloader()

        model = build_model(self.device == "cpu")
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for image, metadata in image_loader:
                heatmap_size = M.resolution
                result = model(
                    {
                        "image": image.to(self.device),
                    },
                    isTest=True,
                )

                wrapped_results = result["heatmaps"]
                results = self.__unwrap_results(wrapped_results)

                for result, meta in zip(results, metadata):
                    predicted_lines = result["lines"]
                    scores = result["score"]

                    # reformat: [[y1, x1], [y2, x2]] -> [x1, y1, x2, y2]
                    predicted_lines = (
                        predicted_lines[:, :, ::-1].flatten().reshape((-1, 4))
                    )

                    # rescale: it was predicted on a 128 x 128 heatmap
                    x_scale = meta["width"] / heatmap_size
                    y_scale = meta["height"] / heatmap_size

                    x_index = [0, 2]
                    y_index = [1, 3]

                    predicted_lines[:, x_index] *= x_scale
                    predicted_lines[:, y_index] *= y_scale

                    self.__save_results(
                        file_name=f"{meta['image_name']}.csv",
                        lines=predicted_lines,
                        scores=scores,
                    )

    def __update_configuration(self) -> None:
        C.update(C.from_yaml(filename=self.base_config_path))
        C.update(C.from_yaml(filename=self.model_config_path))
        M.update(C.model)
        C.io.model_initialize_file = self.pretrained_model_path
        C.io.datadir = self.image_path

    def __create_imageloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            LineDataset(C.io.datadir),
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=C.io.num_workers,
            pin_memory=True,
        )

    def __save_results(
        self, file_name: str, lines: np.ndarray, scores: np.ndarray
    ) -> None:
        np.savetxt(path.join(self.lines_path, file_name), lines, delimiter=",")
        np.savetxt(path.join(self.scores_path, file_name), scores, delimiter=",")

    @staticmethod
    def __unwrap_results(wrapped_results):
        batch_size = wrapped_results["lines"].shape[0]
        return [
            dict(
                (prediction_name, predictions[i].cpu().numpy())
                for prediction_name, predictions in wrapped_results.items()
            )
            for i in range(batch_size)
        ]
