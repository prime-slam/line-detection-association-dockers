# -*- coding: utf-8 -*-
"""python reimplementation of
opencv_contrib/blob/master/modules/line_descriptor/src/bitops.hpp
"""


def popcnt(x):
    """Count set bits in x

    Magic from https://www.expobrain.net/2013/07/29/hamming-weights-python-implementation/""" # noqa
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    return (((x + (x >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def pairwise_lbd_distance(vec):
    import numpy as np
    return np.sum(popcnt(np.bitwise_xor(vec, vec[:, None])), axis=2)
