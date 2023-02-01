#!/usr/bin/python3

import sys
import math

from os import mkfifo

from pathlib import Path
from struct import pack
import numpy as np
from time import sleep


def cout(t):
    sys.stdout.write(t)


def flush():
    sys.stdout.flush()


c2s_pipe = "~/c2s"


def write_pipe(pipe_dest, data):
    if type(data) == np.matrix or type(data) == np.ndarray:
        dat = data.reshape([-1, 16]).tolist()[0]
    elif type(data) == list:
        dat = data

    with open(pipe_dest, 'wb') as fifo:
        for d in dat:
            fifo.write(pack("f", d))
        # data = fifo.read(10)
        # if not len(data):
            # print("\nWriter closed", flush=True)
    # print("read {}".format(pipe_dest))
    # print("wrote ` {} `".format(", ".join(map(str, data))))


def length(vec):
    from math import sqrt
    return sqrt(np.dot(vec, vec.T))


def normal(vec):
    return vec / length(vec)


def translate(mat, vec):
    m = np.eye(4)
    m[3, :3] = vec[:3]
    return np.dot(mat, m)


def rotate(mat, angle, direction, p=None):
    m = np.eye(4)
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direc = direction[:3]
    direc = normal(direc)
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direc, direc) * (1.0 - cosa)
    direc *= sina
    R += np.array([[0.0, -direc[2], direc[1]],
                   [direc[2], 0.0, -direc[0]],
                   [-direc[1], direc[0], 0.0]])
    m[:3, :3] = R
    if p is not None:
        p = np.array(p[:3], dtype=np.float64, copy=False)
        m[:3, 3] = p - np.dot(R, p)
    return np.dot(mat, m.T)


def angle_between(vec1, vec2):
    from math import atan2
    return atan2(length(np.cross(vec1, vec2)),
                 np.dot(vec1, vec2))


def look_at(mat, target):
    l = mat[3, :3]
    # new x-axis (front)
    x = normal(target - l)
    # new y-axis (right) (perpendicular on X (front) and Z (Up))
    y = np.cross(x, np.array([0, 0, 1]))
    # we have to look straight down or up (=> object directly below/above)
    if not y.any():
        # use old y axis
        # (this is like standing and looking between your feet
        #             without turning around)
        y = mat[1, :3]
    # new z-axis (up) (perpendicular on X (front) and Y (right))
    z = np.cross(x, y)
    # if new z-axis points downwards
    # we are looking upside-down
    if z[2] < 0:
        # flip right axis (because it did actually point to the left)
        y = -y
        # and calculate z again. This should now point upwards
        z = np.cross(x, y)
    return np.array([[x[0], x[1], x[2], 0],
                     [y[0], y[1], y[2], 0],
                     [z[0], z[1], z[2], 0],
                     [l[0], l[1], l[2], 1]])


if __name__ == "__main__":
    my_file = Path(c2s_pipe)
    if not my_file.exists():
        mkfifo(c2s_pipe)
    if not my_file.exists():
        print("could not generate {}".format(c2s_pipe))
        exit(1)

    # origin_data = [
    #     -0.515676, 0.623356,  -0.587798, 0,  # 0   3
    #     -0.770518, -0.637418, 0,         0,  # 4   7
    #     -0.374673, 0.452909,  0.809008,  0,  # 8  11
    #     210,       -370,      280,       1   # 12 15
    # ]
    origin_data = [
        1, 0, 0, 0,  # 0   3
        0, 1, 0, 0,  # 4   7
        0, 0, 1, 0,  # 8  11
        0, 0, 90, 1   # 12 15
    ]
    kissen = ([-111, -35, -1],
              [-41, -115, -1],
              [-31, 85, 39])

    kissen = list(map(np.array, kissen))
    origin_data = np.array(origin_data).reshape([4,4])
    data = origin_data.copy()

    ######

    # reset
    # print("resetting")
    # write_pipe(c2s_pipe, data)
    dx = -np.linspace(31, 71, 40, endpoint=True)
    dx = dx.tolist() + dx.tolist()[::-1]
    dy = np.linspace(85, 105, 20, endpoint=True)
    dy2 = dy - (dy.max() - dy.min())
    dy2 = dy2.tolist()[::-1] + dy2.tolist()
    dy = dy.tolist() + dy.tolist()[::-1]
    dy = dy + dy2
    data = np.eye(4)
    for i in range(len(dx)):
        new_location = np.array([dx[i], dy[i], 90])
        data[3, :3] = new_location
        # data = translate(m, np.array([dx[i], dy[i], 90]))
        data = look_at(data, kissen[2])
        # data = look_at(data, np.array([0,0,0]))
        write_pipe(c2s_pipe, data)
        # pass
