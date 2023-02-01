import sys
import math

from os import mkfifo

from pathlib import Path
from struct import pack
import numpy as np


def write_pipe(pipe_dest, data):
    if type(data) == np.matrix or type(data) == np.ndarray:
        dat = data.reshape([-1, 16]).tolist()[0]
    elif type(data) == list:
        dat = data

    with open(pipe_dest, 'wb') as fifo:
        for d in dat:
            fifo.write(pack("B", d))


# mkfifo("~/fifo")

data = list(range(16))
write_pipe("~/fifo", data)
