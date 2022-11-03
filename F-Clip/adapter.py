import warnings
import numpy as np
import torch

from os import path, makedirs
from FClip.line_dataset import LineDataset, collate
from FClip.config import C, M
from test import build_model

warnings.filterwarnings("ignore")


class Adapter:
    def __init__(
        self,
        image_path: str,
        output_path: str,
        base_config_path: str = path.join(path.dirname(__file__), "config/base.yaml"),
        model_config_path: str = path.join(
            path.dirname(__file__), "config/fclip_HR.yaml"
        ),
        pretrained_model_path: str = path.join(
            path.dirname(__file__), "pretrained/HR/checkpoint.pth.tar"
        ),
        batch_size: int = 2,
    ):
        self.image_path = image_path
        self.lines_path = f"{output_path}/lines"
        self.scores_path = f"{output_path}/scores"
        self.base_config_path = base_config_path
        self.model_config_path = model_config_path
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size
        self.__update_configuration()

    def run(self) -> None:
        makedirs(self.lines_path, exist_ok=True)
        makedirs(self.scores_path, exist_ok=True)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(0)

        image_loader = self.__create_imageloader()

        model = build_model()
        model.cuda()
        model.eval()

        with torch.no_grad():
            for image, metadata in image_loader:
                heatmap_size = M.resolution
                result = model(
                    {
                        "image": image.cuda(),
                    },
                    isTest=True,
                )

                heatmaps = result["heatmaps"]
                for i, meta in enumerate(metadata):

                    results = {}
                    for k, v in heatmaps.items():
                        if v is not None:
                            results[k] = v[i].cpu().numpy()

                    predicted_lines = results["lines"]
                    scores = results["score"]

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
