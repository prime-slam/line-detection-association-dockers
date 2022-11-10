import cv2
import numpy as np
import torch

from pathlib import Path
from skimage import io
from torch.utils.data import default_collate, Dataset
from typing import Tuple


def collate(batch):
    return (default_collate([b[0] for b in batch]), [b[1] for b in batch])


class LineDataset(Dataset):
    def __init__(self, data_path: Path, image_rescale: Tuple[int, int] = (512, 512)):
        files = sorted(data_path.iterdir())
        self.files = files
        self.image_rescale = image_rescale

    def __len__(self):
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
        transformed = cv2.resize(image, self.image_rescale)
        height, width, channels = transformed.shape
        hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        imgv0 = hsv[..., 2]
        imgv = cv2.resize(
            imgv0, (0, 0), fx=1.0 / 4, fy=1.0 / 4, interpolation=cv2.INTER_LINEAR
        )
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv2.resize(imgv, (width, height), interpolation=cv2.INTER_LINEAR)
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        transformed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        transformed = transformed.astype(np.float32) / 255.0
        transformed = transformed.transpose(2, 0, 1)

        return transformed
