import glob

import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset, default_collate

from FClip.config import M


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch]
    )


class LineDataset(Dataset):
    def __init__(self, rootdir, image_rescale=(512, 512)):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/*")
        filelist.sort()
        self.filelist = filelist
        self.image_rescale = image_rescale

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        path = self.filelist[idx]
        image_name = path.split('/')[-1].split('.')[0]
        image_ = io.imread(path)
        height, width = image_.shape[0], image_.shape[1]
        image_ = cv2.resize(image_, self.image_rescale)

        image_ = image_.astype(float)[:, :, :3]

        meta = {
            'width': width,
            'height': height,
            'image_name': image_name
        }

        image = (image_ - M.image.mean) / M.image.stddev
        image = np.rollaxis(image, 2).copy()

        return torch.from_numpy(image).float(), meta
