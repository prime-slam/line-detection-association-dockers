from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd


class Column(Enum):
    CUTOUT = 'cutout'
    DESC = 'desc'
    LABEL = 'label'
    LEFT = 'left'
    TARGET_HEIGHT = 'target_height'
    OFFSET = 'offset'
    P1 = 'p1'
    P2 = 'p2'
    WAVELET_CUTOUT = 'wavelet_cutout'


class ProcessingData:
    def get_cutouts(self) -> List[np.ndarray]:
        return self.__dataframe[Column.CUTOUT].to_list()

    def get_descriptors(self) -> List[np.ndarray]:
        return self.__dataframe[Column.DESC].to_list()

    def get_start_points(self) -> List[np.ndarray]:
        return self.__dataframe[Column.P1].to_list()

    def get_end_points(self) -> List[np.ndarray]:
        return self.__dataframe[Column.P2].to_list()

    def get_wavelet_cutouts(self) -> List[np.ndarray]:
        return self.__dataframe[Column.WAVELET_CUTOUT].to_list()

    def __init__(self, npz: np.lib.npyio.NpzFile, use_right: bool = False) -> None:
        # extract data from given NPZ file
        num_cutouts, dataframe = ProcessingData.__extract_data(npz)

        # filter dataframe by side
        is_left = dataframe[Column.LEFT]
        use_left = (not bool(use_right))
        dataframe = dataframe[is_left == use_left].copy()
        dataframe = dataframe.sort_index(ascending=True, inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)

        # assign instance variables
        self.__use_left = use_left
        self.__num_cutouts = len(dataframe)
        self.__dataframe = dataframe

    def __len__(self) -> int:
        return self.__num_cutouts

    @staticmethod
    def __extract_data(npz: np.lib.npyio.NpzFile) -> Tuple[int, pd.DataFrame]:
        # extract number of cutouts
        num_cutouts = int(npz['n_cutouts'].item())

        # initialize dictionary keys
        cutout_key = '{}_cutout'
        desc_key = '{}_desc'
        label_key = '{}_label'
        left_key = '{}_left'
        target_height_key = '{}_target_height'
        offset_key = '{}_offset'
        p1_key = '{}_p1'
        p2_key = '{}_p2'
        wavelet_cutout_key = '{}_{}_wavelet_cutout'

        # initialize data lists
        cutout_list = []
        desc_list = []
        label_list = []
        left_list = []
        target_height_list = []
        offset_list = []
        p1_list = []
        p2_list = []
        wavelet_cutout_list = []

        # loop through cutouts and extract data from NPZ dictionary
        for key in range(num_cutouts):
            cutout_list.append(npz[cutout_key.format(key)])
            desc_list.append(npz[desc_key.format(key)])
            label_list.append(npz[label_key.format(key)].item())
            left_list.append(npz[left_key.format(key)].item())
            target_height_list.append(npz[target_height_key.format(key)].item())
            offset_list.append(npz[offset_key.format(key)].item())
            p1_list.append(npz[p1_key.format(key)])
            p2_list.append(npz[p2_key.format(key)])
            # TODO: do we always have 8 wavelet cutouts?
            wavelet_cutouts = []
            try:  # if we use a npz from dld it won't necessarily have wavelet cutouts
                for level in range(8):
                    wavelet_cutouts.append(npz[wavelet_cutout_key.format(key, level)])
            except KeyError:  # this is not the best way to handle this case but it should work for now
                wavelet_cutouts.clear()
            wavelet_cutout_list.append(wavelet_cutouts)

        # create data dictionary from lists for DataFrame
        data = {
            Column.CUTOUT: cutout_list,
            Column.DESC: desc_list,
            Column.LABEL: label_list,
            Column.LEFT: left_list,
            Column.TARGET_HEIGHT: target_height_list,
            Column.OFFSET: offset_list,
            Column.P1: p1_list,
            Column.P2: p2_list,
            Column.WAVELET_CUTOUT: wavelet_cutout_list
        }

        # create DataFrame from data dictionary
        dataframe = pd.DataFrame(data)

        return num_cutouts, dataframe
