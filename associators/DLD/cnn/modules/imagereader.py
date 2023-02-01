from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorpack as tp

import misc.logger as logger
from .geometry import Segment2D
from .linedetector import LineDetector
from .lineprocessor import LineProcessor
from .processingdata import ProcessingData

_L = logger.getLogger("DataFlow")


class Column(Enum):
    CUTOUT = 'cutout'
    HEIGHT = 'height'
    LEFT = 'left'
    LABEL = 'label'
    KEYLINE = 'keyline'
    BD_DESC = 'bd_desc'
    CNN_DESC = 'cnn_desc'


class LineData(tp.dataflow.RNGDataFlow):
    def reset_state(self) -> None:
        super(LineData, self).reset_state()

    def save_results(self, cnn_desc_list: list, label_list: list, left_list: list) -> None:
        # create data dictionary from lists for dataframe
        data = {
            Column.LEFT.value: left_list,
            Column.LABEL.value: label_list,
            Column.CNN_DESC.value: cnn_desc_list
        }

        # join results to instance dataframe
        left_df = self.__dataframe.copy()
        right_df = pd.DataFrame(data)
        merged = pd.merge(left_df, right_df, how='left', left_on=[Column.LEFT.value, Column.LABEL.value],
                          right_on=[Column.LEFT.value, Column.LABEL.value])

        # filter dataframe by left side
        is_left = merged[Column.LEFT.value]
        use_left = True
        merged = merged[is_left == use_left].copy()
        merged = merged.sort_index(ascending=True, inplace=False)
        merged = merged.reset_index(drop=True, inplace=False)

        # convert keylines into points
        keylines = merged[Column.KEYLINE.value].to_list()
        keylines = [np.array([[kl.startPointX, kl.startPointY], [kl.endPointX, kl.endPointY]], np.float32) for kl in
                    keylines]

        # extract cnn descriptors from dataframe
        cnn_desc_list = merged[Column.CNN_DESC.value].to_list()

        # save NPZ file next to image file
        name = "{stem}.npz".format(stem=str(self.__image_path.stem))
        file = str(self.__image_path.with_name(name))

        # save lines and descriptors into a single file in NPZ format
        np.savez_compressed(file, keylines=keylines, cnn_descs=cnn_desc_list)

    def __init__(self, image_path: str = '', cutout_width: int = 27, cutout_height: int = 100, min_length: int = 15,
                 keylines_path: str = '', use_right: bool = False) -> None:
        super().__init__()

        # set instance variables
        self.__image_path = str(image_path)
        self.__cutout_width = int(cutout_width)
        self.__cutout_height = int(cutout_height)
        self.__min_length = int(min_length)
        self.__keylines_path = str(keylines_path)
        self.__use_right = bool(use_right)

        # check if precomputed keylines are used
        self.__use_precomputed = False
        if self.__keylines_path and (self.__keylines_path) != 'None':
            self.__use_precomputed = True

        # check if strings are not 'falsy'
        if self.__use_precomputed:
            assert self.__image_path and self.__keylines_path
            self.__image_path = Path(self.__image_path)
            self.__keylines_path = Path(self.__keylines_path)
        else:
            assert self.__image_path
            self.__image_path = Path(self.__image_path)

        # read image from given path
        image = cv2.imread(str(self.__image_path), cv2.IMREAD_COLOR)

        # initialize LineDetector and LineProcessor
        ld = LineDetector()
        lp = LineProcessor()

        # set width of band based on cutout width
        ld.set_width_of_band(self.__cutout_width)

        # initialize keyline list
        keylines = []

        # load keylines
        if self.__use_precomputed:
            npz = np.load(str(self.__keylines_path))
            data = ProcessingData(npz, self.__use_right)

            start_points = data.get_start_points()
            end_points = data.get_end_points()
            img_max_size = max(image.shape)

            for class_id in range(len(data)):
                p1 = start_points[class_id]
                p2 = end_points[class_id]
                seg = Segment2D(p1, p2)
                kl = seg.to_keyline(class_id, img_max_size)
                keylines.append(kl)
        # compute keylines
        else:
            keylines = ld.detect(image, self.__min_length)

        # compute cutouts from image and keylines
        cutouts = lp.process(image, keylines)

        # convert cutouts (should be in BGR) to RGBA
        cutouts = [cv2.cvtColor(cutout, cv2.COLOR_BGR2RGBA) for cutout in cutouts]
        # set alpha channel to zero
        for cutout in cutouts:
            cutout[:, :, 3] = 0


        # compute descriptors from image and keylines
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keylines, descriptors = ld.compute(image, keylines)

        # check if size of lists match up
        assert len(keylines) == len(cutouts) == len(descriptors)
        num_cutouts = len(cutouts)

        # create lists from data
        cutout_list = cutouts
        height_list = [self.__cutout_height] * num_cutouts
        left_list = [True] * num_cutouts
        label_list = list(range(num_cutouts))
        keyline_list = keylines
        descriptor_list = descriptors

        # create data for both sides
        cutout_list.extend(cutout_list)
        height_list.extend(height_list)
        left_list.extend([False] * num_cutouts)
        label_list.extend(label_list)
        keyline_list.extend(keyline_list)
        descriptor_list.extend(descriptor_list)

        # create data dictionary from lists for DataFrame
        data = {
            Column.CUTOUT.value: cutout_list,
            Column.HEIGHT.value: height_list,
            Column.LEFT.value: left_list,
            Column.LABEL.value: label_list,
            Column.KEYLINE.value: keyline_list,
            Column.BD_DESC.value: descriptor_list
        }

        # create DataFrame from data dictionary
        dataframe = pd.DataFrame(data)
        dataframe = dataframe.sort_values(by=[Column.LABEL.value, Column.LEFT.value], ascending=True, inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)

        self.__num_cutouts = num_cutouts
        self.__dataframe = dataframe

    def __iter__(self) -> list:
        for __, row in self.__dataframe.iterrows():
            # set alpha of cutout to zero
            cutout = row[Column.CUTOUT.value].copy()
            cutout[:, :, 3] = 0

            # create datapoint using dataframe
            dp = [cutout, row[Column.HEIGHT.value], row[Column.LEFT.value], row[Column.LABEL.value],
                  row[Column.BD_DESC.value]].copy()
            yield dp

    def __len__(self) -> int:
        return self.__num_cutouts
