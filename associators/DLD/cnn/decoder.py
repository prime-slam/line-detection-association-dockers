from io import BytesIO
"""Contains functions for en/decoding purposes"""

__header_template = ("{{'descr': '|u1', "
                     "'fortran_order': False, "
                     "'shape': ({}, {}, {}), }}")


def data_to_npy(data, shape):
    import numpy as np
    from struct import pack

    width, height, channel = shape

    # MAGIC + Format Version (1.0)
    preamble = b"\x93NUMPY\x01\x00"
    header = bytes(__header_template
                   .format(height, width, channel),
                   "utf-8")
    # preamble + header should have % 16 length
    # preamble has 10 bytes
    # last byte should be \n
    remainder = 16 - (10 + len(header)) % 16
    header += (b" " * (remainder - 1))
    header += b"\n"

    img_b = BytesIO()
    img_b.write(preamble)
    img_b.write(pack("<H", len(header)))
    img_b.write(header)
    img_b.write(data)
    img_b.seek(0)

    arr = np.load(img_b)
    arr = arr.astype(dtype="float32")  # tf needs float32
    return arr


# takes 1114.0µs for 20x100x4
def stream_to_npy(data: BytesIO, shape):
    """Extracts image data from readable stream dataobject
    and generates numpy image matrix

    see https://docs.scipy.org/doc/numpy/neps/npy-format.html#format-specification-version-1-0""" # noqa

    width, height, channel = shape
    return data_to_npy(data.read(height * width * channel), shape)
