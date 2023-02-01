#!/usr/bin/python3

import sys

from pathlib import Path
# from struct import unpack


def cout(t):
    sys.stdout.write(t)


def flush():
    sys.stdout.flush()


depth_pipe = "~/depthOut"
color_pipe = "~/colorOut"

pipes = [depth_pipe, color_pipe]


def read_pipe(pipe_dest):
    print("trying to read {}".format(pipe_dest))
    my_file = Path(pipe_dest)
    if my_file.exists():
        print("{} exists".format(pipe_dest))
        with open(pipe_dest, 'rb') as fifo:
            data = fifo.read(10)
            if not len(data):
                print("\nWriter closed", flush=True)
        print("read {}".format(pipe_dest))
    else:
        print("file doesn't exist: {}".format(pipe_dest))


if __name__ == "__main__":
    from multiprocessing import Process
    processes = [Process(target=read_pipe, args=(arg,)) for arg in pipes]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
