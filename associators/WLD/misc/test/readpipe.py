#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from os import mkfifo
import time
import multiprocessing
from pathlib import Path
from struct import unpack, pack
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

import readpipe3 as wp

depth_pipe = "/home/username/depthOut"
color_pipe = "/home/username/colorOut"

DEBUG_OUTPUT = 0

OVERRIDE_WIDTH = 0
OVERRIDE_HEIGHT = 0

# OVERRIDE_WIDTH = 1920
# OVERRIDE_HEIGHT = 1200

# TIME_LIMIT = 10  # sekunden
TIME_LIMIT = 20  # sekunden

img_as_list = list()
def cout(t):
    sys.stdout.write(t)


def flush():
    sys.stdout.flush()


def move(iter):

    c2s_pipe = "~/c2s"
    my_file = Path(c2s_pipe)
    if not my_file.exists():
        mkfifo(c2s_pipe)
    if not my_file.exists():
        print("could not generate {}".format(c2s_pipe))
        return

    if iter <= 0:
        print(" ITER 0!!!! ORIGIN")
        origin_data = [
            -0.515676, 0.623356,  -0.587798, 0,  # 0   3
            -0.770518, -0.637418, 0,         0,  # 4   7
            -0.374673, 0.452909,  0.809008,  0,  # 8  11
            210,       -370,      280,       1   # 12 15
        ]
        origin_data = np.array(origin_data).reshape([4, 4])
        wp.write_pipe(c2s_pipe, origin_data)
        return

    kissen = ([-111, -35, -1],
              [-41, -115, -1],
              [-31, 85, 39])

    kissen = list(map(np.array, kissen))
    data = np.eye(4)
    dx = [0, -20, -20]
    dy = [0, 100, -100]
    di, mo = divmod(iter-1, 3)
    new_location = np.array([dx[mo], dy[mo], 90])
    data[3, :3] = new_location
    data = wp.look_at(data, kissen[di])
    wp.write_pipe(c2s_pipe, data)


def read_pipe(color_depth, pipe_dest, q, rep):
    my_name = "  {}".format(multiprocessing.current_process().name)
    data_read = 0
    to_wait = TIME_LIMIT
    my_file = Path(pipe_dest)
    read_color = color_depth == "COUT"
    read_depth = not read_color
    if read_color:
        print("{}: reading color".format(my_name))
        ndimg = np.ndarray((100, 100, 4), dtype=np.uint8)
    else:
        print("{}: reading color".format(my_name))
        ndimg = np.zeros((100, 100), dtype=np.float32)

    # waiting..
    print("{}: waiting for pipe at `{}`".format(my_name, pipe_dest))
    while not my_file.exists():
        to_wait -= 0.1
        if to_wait <= 0:
            break
        time.sleep(0.1)
        cout('\r[')
        left = (100 / TIME_LIMIT) * to_wait
        cout('.' * int(left))
        cout(' ' * int((100 / TIME_LIMIT) * (TIME_LIMIT - to_wait)))
        cout('] ')
        cout("{:.1f}".format(to_wait))
        flush()

    # move(0)
    # move(0)
    # move(0)
    # for iteration in range(rep):
        # move(iteration)

    # sleep(0.2)
    for iteration in range(-3, rep):
        if read_depth:
            move(iteration)
        my_name = "  {}({})".format(multiprocessing.current_process().name,
                                    iteration)
        orig_data = []
        orig_data_list = []
        # reading ..
        if my_file.exists():
            read_str = ""
            begin = False
            width = -1
            height = -1
            img = list()
            row = 0
            col = 0
            # biggest_num = -1
            # smallest_num = 20
            print("\n{}: found pipe.. start listening".format(my_name))
            with open(pipe_dest, 'rb') as fifo:
                # for _ in range(1024):
                while True:
                    if not begin:
                        read_str = ""
                        data = fifo.read(1)
                        if not len(data):
                            print("\n{}: Writer closed".format(my_name), flush=True)
                            break
                        try:
                            if (data.decode("UTF-8") == color_depth[0]):
                                read_str += data.decode("UTF-8")
                                for _ in range(3):
                                    data = fifo.read(1)
                                    read_str += data.decode("UTF-8")
                                    if (read_str == color_depth):
                                        orig_data.append(bytes(read_str, "utf-8"))
                                        print("{}: beginning..".format(my_name), flush=True)
                                        begin = True
                                    if read_str not in color_depth:
                                        break
                        except:
                            pass
                    else:
                        if width < 0:
                            data = fifo.read(4)
                            if not len(data):
                                print("\n{}: Writer closed".format(my_name), flush=True)
                                break
                            orig_data.append(data)
                            width = unpack("I", data)[0]
                            print("{}:    Width: {} ".format(my_name, width),
                                flush=True)
                        elif height < 0:
                            data = fifo.read(4)
                            if not len(data):
                                print("\n{}: Writer closed".format(my_name), flush=True)
                                break
                            orig_data.append(data)
                            height = unpack("I", data)[0]
                            print("{}:    Height: {} ".format(my_name, height),
                                flush=True)
                        else:
                            if col >= width:
                                row += 1
                                col = 0
                            if row >= height:
                                break
                                # img.append([])
                            if col == 0:
                                img.append(list())
                                orig_data_list.append(list())
                            col += 1

                            data = fifo.read(4)
                            if not len(data):
                                print("\n{}: Writer closed".format(my_name), flush=True)
                                break
                            orig_data.append(data)
                            if read_depth:
                                # orig_data.append(data)
                                orig_data_list[row].append(data)
                                # if (data[3] == 1):
                                # img[row].append(unpack("<f", data))
                                # img[row].append(unpack(">f", data))
                                # else:
                                # img[row].append((0.0,))
                                to_unpack = [
                                    # ("f", data[:3] + b'\x00'),
                                    ##### THIS IS WORKING!!!
                                    # ("f", data[1:] + b'\x00'),
                                    # ("f", b'\x00' + data[1:])
                                    ###########
                                    ("f", (data[1:] + b'\x00') if data[3] < 30 else b'\x00\x00\x00\x00'),
                                    # ("f", (b'\x00' + data[1:]) if data[3] < 30 else b'\x00\x00\x00\x00'),
                                    # ("B", bytes([data[0] != 0])),
                                    # ("B", data[:1]),
                                    # ("B", bytes([data[3] != 0])),
                                    # ("B", data[3:] if data[3] > 30 else b'\x00')
                                    # ("B", data[3:])
                                ]
                                unpackf = list(zip(*to_unpack))
                                img[row].append(unpack("".join(unpackf[0]),
                                                    b''.join(unpackf[1])))
                                img_as_list.append(unpack("".join(unpackf[0]),
                                                    b''.join(unpackf[1]))[0])
                                # img[row].append(unpack("fB", data[:3] + b'\x00' + data[3:]))
                                # img[row].append(unpack("I", b'\x00' + data[:3]))
                                # img[row].append(unpack("f", b'\x00' + data[:3]))
                                # img[row].append(unpack("<f", b'\x00' + data[1:]))
                                # img[row].append(unpack("<f", data))
                                # biggest_num = max(biggest_num, img[row][-1][0])
                                # smallest_num = min(smallest_num, img[row][-1][0])
                            elif read_color:
                                img[row].append(unpack("BBBB", data))

            if width > 0 and height > 0 and iteration >= 0:
                if (len(orig_data)):
                    out_file = "~/pout"
                    if read_depth:
                        out_file = "~/pdepthOut2"
                    elif read_color:
                        out_file = "~/pcolorOut2"

                    with open(out_file, "ab") as dfile:
                        for da in orig_data:
                            dfile.write(da)

                    # mkfifo("~/pdepthOut")
                    # with open("~/Desktop/out/python_orig_depth.pfm", "wb") as dfile:
                    if DEBUG_OUTPUT:
                        with open("~/Desktop/out/python_orig_read.pfm", "w") as dfiler:
                            # for dat in orig_data:
                            for da in orig_data_list:
                                for dat in da:
                                # dfiler.write("{:.5e} ".format(float(unpack("f", dat)[0])))
                                # for ddat in dat:
                                    # dfiler.write(str(int(ddat)))
                                    dfiler.write("|".join(["{:0>8}".format(bin(int(ddat))[2:])
                                                        for ddat in dat]))
                                    # dfiler.write("|" if (ddat != dat[-1]) else "")
                                    dfiler.write("\n")
                                dfiler.write("\n")
                        with open("~/Desktop/out/python_shift_read.pfm", "w") as dfile:
                            # for dat in img_as_list:
                            for da in img:
                                for dat in da:
                                    # dfile.write("{:.5e} ".format(float(dat)))
                                    # for ddat in pack("f", dat):
                                    # dfile.write(str(int(ddat)))
                                    dfile.write(
                                        "|".join(["{:0>8}".format(
                                            bin(int(ddat))[2:])
                                                for ddat in pack("f", dat[0])]))
                                    dfile.write("\n")
                                dfile.write("\n")
                                # dfile.write(pack("f", dat))
                h = height
                w = width
                if read_color:
                    ndimg = np.ndarray((h, w, 4), dtype=np.uint8)
                if read_depth:
                    # cout("{}: range: {} - {}\n".format(my_name, smallest_num,
                    #                                    biggest_num))
                    # ndimg = np.ndarray((h, w), dtype=np.float32)
                    # ndimg2 = np.ndarray((h, w), dtype=np.float32)
                    ndimg = [np.ndarray((h, w), dtype=np.float32)
                            for _ in range(len(img[0][0]))]
                for i, row in enumerate(img):
                    for j, col in enumerate(row):
                        if read_color:
                            for c in range(3):
                                ndimg[i, j, c] = col[3 - c]
                            ndimg[i, j, 3] = 255
                        elif read_depth:
                            # ndimg[i, j] = (1/(biggest_num - smallest_num)
                            # * (col[0] - smallest_num))
                            for x in range(len(col)):
                                ndimg[x][i, j] = col[x]
                            # ndimg[i, j] = col[0]
                            # ndimg2[i, j] = col[1]
                            # imgplot = plt.imshow(ndimg)
                # plt.show()
                # print("{}: Read {} bytes, ({}kb, {}mb, {}gb)".format(
                #     my_name,
                #     data_read,
                #     data_read/1024,
                #     data_read/1024/1024,
                #     data_read/1024/1024/1024))
        else:
            print("\n\n{}: no pipe detected..".format(my_name), flush=True)
        # print("{}: putting..".format(my_name))
        q.put(ndimg)
        print("{}: done".format(my_name))
    return


if __name__ == "__main__":
    from multiprocessing import Process, SimpleQueue
    rep = 10
    q = SimpleQueue()
    processes = [Process(name=a[0], target=read_pipe, args=a)
                    for a in [("DOUT", depth_pipe, q, rep),
                           ("COUT", color_pipe, q, rep)]]
# start and wait for processes
    for p in processes:
        print("posting process {}".format(p.name))
        p.start()

    for _ in range(len(processes)*rep):
        tmp_img = q.get()
    # for _ in range(2):
    #     tmp_img = q.get()
    #     if type(tmp_img) == list:

    #         for x in range(len(tmp_img)):
    #             ximg = tmp_img[x]
    #             oh = ximg.shape[0] if not OVERRIDE_HEIGHT else OVERRIDE_HEIGHT
    #             ow = ximg.shape[1] if not OVERRIDE_WIDTH else OVERRIDE_WIDTH
    #             ximg = ximg[:oh, :ow]
    #             tmp_max = ximg.max()
    #             tmp_min = ximg.min()
    #             if tmp_max:
    #                 print("DepthStencil {}, [{}, {}]".format(
    #                     x, tmp_min, tmp_max))
    #                 plt.figure()
    #                 plt.title("DepthStencil {}".format(x))
    #                 # ximg = (1 / (tmp_max - tmp_min)) * (ximg - tmp_min)
    #                 print("DepthStencil {}, [{}, {}]".format(
    #                     x, ximg.min(), ximg.max()))
    #                 if DEBUG_OUTPUT:
    #                     with open("~/Desktop/out/python_scale_read.pfm",
    #                               "w") as dfiler:
    #                         for dy in ximg:
    #                             for dx in dy:
    #                                 # dfiler.write("{:.5e} ".format(float(dx)))
    #                                 # dfiler.write("{:.7f} ".format(float(dx)))
    #                                 # for ddat in pack("f", dx):
    #                                     # dfiler.write(str(int(ddat)))
    #                                 dfiler.write(
    #                                     "|".join(["{:0>8}".format(
    #                                         bin(int(ddat))[2:])
    #                                               for ddat in pack("f", dx)]))
    #                                 dfiler.write("\n")
    #                             dfiler.write("\n")
    #                 # plt.imshow(1 / (tmp_max - tmp_min) * (ximg - tmp_min))
    #                 # plt.imshow(ximg, vmin=ximg.min(), vmax=ximg.max(),
    #                 #            cmap="gist_rainbow")
    #                 plt.imshow(ximg, vmin=ximg.min(),
    #                            vmax=ximg.max(), cmap="gray")
    #                 # plt.colorbar()
    #                 plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    #     else:
    #         plt.figure()
    #         plt.title("Color Image")
    #         oh = tmp_img.shape[0] if not OVERRIDE_HEIGHT else OVERRIDE_HEIGHT
    #         ow = tmp_img.shape[1] if not OVERRIDE_WIDTH else OVERRIDE_WIDTH
    #         tmp_img = tmp_img[:oh, :ow]
    #         plt.imshow(tmp_img)

    plt.show()
