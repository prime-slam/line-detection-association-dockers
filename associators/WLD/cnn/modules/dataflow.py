# -*- coding: utf-8 -*-
import tensorpack as tp

import sys
from os import listdir, path

from cnn.modules.fio import FBytesIO
from cnn import config

import misc.logger as logger
_L = logger.getLogger("DataFlow")


def np_convert():
    """Create list -> np.array converter.

    This is to cache the conversion dictionary
    while keeping global namespace clean"""
    import numpy as np
    conv = {
        int: np.int32,
        bool: np.int32,
        float: np.float32,
    }

    def asarray(el):
        if not el:
            raise AttributeError("Empty list not supported")
        dt = conv.get(type(el[0]))
        if not dt:
            dt = object
            try:
                dt = el[0].dtype
            except AttributeError:
                pass
        return np.asarray(el, dtype=dt)
    return asarray
asarray = np_convert()  # noqa


def extract_data_wavelets(data):
    """extract learning/testing data from one ProcessingData"""
    import numpy as np
    from cnn.decoder import data_to_npy
    from numpy import frombuffer

    (max_height, idx, left, raw_img,
        width, height, raw_wavelets, raw_desc) = data
    desc = frombuffer(raw_desc, dtype="uint8")

    learn_data = []
    img = data_to_npy(raw_img, (width, height, 4))
    if False:  # True: idx % 2
        img[:, :, :] = 127.5
    else:
        img[:, :, 3] = 127.5

    if (height != max_height):
        img = np.resize(img, (max_height, *img.shape[1:]))

    from matplotlib import pyplot as plt

    wavelets = np.empty((100, 27, 0), dtype=np.float32)
    for n in range(len(raw_wavelets)):
        gw = data_to_npy(raw_wavelets[n],(width, height, 1))
        wavelets = np.append(wavelets, gw, axis=2)

        # lum_img = gw[:, :, 0]
        # plt.imshow(lum_img, interpolation="nearest")
        # plt.show()

    learn_data.append(img)
    learn_data.append(height)
    learn_data.append(left)
    learn_data.append(idx)
    learn_data.append(wavelets)
    learn_data.append(desc)

    return learn_data


def extract_data7(data):
    """extract learning/testing data from one ProcessingData"""
    import numpy as np
    from cnn.decoder import data_to_npy
    from numpy import frombuffer

    (max_height, idx, left, raw_img,
        width, height, raw_desc) = data
    desc = frombuffer(raw_desc, dtype="uint8")

    learn_data = []
    img = data_to_npy(raw_img, (width, height, 4))
    img[:, :, 3] = 0
    if (height != max_height):
        img = np.resize(img, (max_height, *img.shape[1:]))

    learn_data.append(img)
    learn_data.append(height)
    learn_data.append(left)
    learn_data.append(idx)
    learn_data.append(desc)

    return learn_data

# TODO: deprecated?
def extract_data8(data):
    import numpy as np
    from cnn.decoder import data_to_npy
    from numpy import frombuffer

    (max_height, idx, left, raw_imgs,
     widths, heights, raw_imgs2, raw_desc) = data
    desc = frombuffer(raw_desc, dtype="uint8")

    learn_data = [[], [], [], [], [], []]
    for n in range(len(raw_imgs)):
        img = data_to_npy(raw_imgs[n],
                          (widths[n], heights[n], 4))
        if (heights[n] != img.shape[0]):
            img = np.resize(img,
                            (max_height, *img.shape[1:]))

        img2 = data_to_npy(raw_imgs2[n],
                           (widths[n], heights[n], 4))
        if (heights[n] != img2.shape[0]):
            img2 = np.resize(img2,
                             (max_height, *img2.shape[1:]))

        learn_data[0].append(img)
        learn_data[1].append(heights[n])
        learn_data[2].append(img2)
        learn_data[-3].append(left)
        learn_data[-2].append(idx)
        learn_data[-1].append(desc)

    tmp = [asarray(el) for el in learn_data]
    # learn_data = np.array(tmp, dtype=object)
    learn_data = tmp

    return learn_data


def collect_input_files(input_folder, slices):
    """Collect the names of all files in given range"""
    nums = [int(el) for el in listdir(input_folder) if str.isnumeric(el)]
    nums.sort()
    file_range = range(nums[0], nums[-1] + 1)

    # collect files
    files = []
    for s in slices:
        files += file_range[s]
    return files


class LineData(tp.dataflow.RNGDataFlow):
    """Extract Data from Server or Files.

    Returns a datapoint as `[<current dataset size><datapoint>]`"""

    def __init__(self, client_or_folder, slices=None, rnd=None):
        from cnn.client import Client

        self.client = None
        self.input_folder = None
        self.slices = slices or [slice(None)]
        self.headerFmt = "!I"  # <numElements>
        self.random = rnd

        # fetch from client or filesystem?
        if type(client_or_folder) == Client:
            self.client = client_or_folder
        else:
            self.input_folder = client_or_folder

        self.match_data = []
        self.max_height = -1

        if self.client:
            self.fetch_func = self.fetch_tcp
        else:
            self.fetch_func = self.fetch_file
            self.file_idx = 0
            self.files = collect_input_files(self.input_folder, self.slices)

    def __len__(self):
        return len(self.match_data)

    def reset_state(self):
        super(LineData, self).reset_state()
        # start with random file
        if self.random:
            self.file_idx = self.rng.randint(0, len(self.files))

        # fetch at the beginning
        self.fetch()

    def fetch(self):
        if self.match_data:
            return

        # _L.debug("new data")
        data = None
        # read until something came through (max 3 times)
        for _ in range(3):
            data = self.fetch_func()
            if data:
                break

        if not data:
            _L.critical("No data after 3 tries..")
            sys.exit("No data..")

        b = FBytesIO(data)
        header = b.unpack(self.headerFmt)
        self.max_height, *self.match_data = LineData.extract(header, b)

    def fetch_tcp(self):
        """Fetch new data from TCP Server"""

        answer = self.client.send("n")  # request doesn't need content
        # stop training if disconnected as we can't query more data
        if answer.error:
            if "disconnected" in answer.error:
                _L.error("remote disconnected..")
                # raise tp.StopTraining()
                raise StopIteration
                return

        data = answer.data

        s = data.find(b":")
        if s < 0:
            return

        return data[s + 1:]

    def fetch_file(self):
        import gzip

        file_id = str(self.files[self.file_idx])
        the_file = path.join(self.input_folder, file_id)

        # set next file index
        if self.random:
            self.file_idx = self.rng.randint(0, len(self.files))
            # print('Random is true')
        else:
            self.file_idx = (self.file_idx + 1) % len(self.files)
            # print('Random is false')

        # the file was not found..
        if not path.exists(the_file):
            _L.error("File {} does not exist.\
            Will try the next one..".format(the_file))
            return

        # read data from file
        with gzip.open(the_file, "rb") as f:
            data = f.read()

        s = 8 + len(file_id) + 1  # magic + payload_size + msgid + :
        return data[s:]

    @staticmethod
    def extract(hdr, b):
        """Extract all data from datastream"""
        num = hdr
        max_height = 0
        data = [-1]

        for i in range(num):
            idx = b.unpack("!I")
            left_nCutouts = b.unpack("!B")
            left = bool(left_nCutouts & 0x80)
            n_cutouts = left_nCutouts & 0x7F
            width, height = b.unpack("!II")

            imgs = []
            imgs2 = []
            heights = []
            widths = []

            for n in range(n_cutouts):
                cut = b.read(width * height * 4)
                cut2 = None
                if config.debug_cutout:
                    cut2 = b.read(width * height * 4)

                if height >= config.min_len:
                    imgs.append(cut)
                    imgs2.append(cut2)
                    heights.append(height)
                    widths.append(width)
                    max_height = max(max_height, height)

            # read Wavelets
            wavelets = []
            num_wavelets = b.unpack("!I")
            for n in range(num_wavelets):
                wavelets.append(b.read(width * height * 1))

            desc = b.read(32)

            if imgs:
                for n in range(len(imgs)):
                    if config.debug_cutout:
                        data.append([idx, left, imgs[n], widths[n],
                                    heights[n], imgs2[n], wavelets, desc])
                    else:
                        data.append([idx, left, imgs[n], widths[n],
                                     heights[n], wavelets, desc])

        data[0] = max_height
        return data

    def __iter__(self):
        """Deliver traning/validation examples afap.
        Must yield shape used in `_get_inputs`."""
        self.fetch()

        for data in self.match_data:
            # always return the current length
            # as the `PrefetchDataZMQ` can't handle dynamic sized Dataflows
            len_data = len(self.match_data)
            # extracted = extract_data7([self.max_height] + data)
            extracted = extract_data_wavelets([self.max_height] + data)
            datapoint = [len_data] + extracted
            yield datapoint

        # delete everything so that we fetch again next time
        del self.match_data[:]


class MyPrefetchDataZMQ(tp.dataflow.PrefetchDataZMQ):
    """PrefetchDataZMQ for dynamic dataflow sizes"""

    def _recv(self):
        """Override `_recv` for changing current dataflow size

        WARNING: do not use this with `nr_proc > 1`"""
        from tensorpack.utils.serialize import loads
        d = loads(self.socket.recv(copy=False))

        self._size = d[0]
        return d


class StripData(tp.dataflow.ProxyDataFlow):
    """Throws away the first value (LineData appends its own size to data)"""
    def __iter__(self):
        for d in self.ds:
            yield d[1:]


class MyTestDataSpeed(tp.dataflow.TestDataSpeed):
    """For testing finite Dataflows"""
    class ItrWrapper(tp.dataflow.ProxyDataFlow):
        def __init__(self, i):
            super().__init__(i)
            self.i = i

        def __iter__(self):
            while True:
                itr = self.i.__iter__()
                for el in itr:
                    yield el

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = MyTestDataSpeed.ItrWrapper(self.ds)

    # def start(self):
    #     return self.start_test()

    # def start_test(self):
    #     import tqdm
    #     from tensorpack.utils.utils import get_tqdm, get_tqdm_kwargs
    #     """
    #     Start testing with a progress bar.
    #     """
    #     self.ds.reset_state()
    #     itr = self.ds.__iter__()
    #     if self.warmup:
    #         for _ in tqdm.trange(self.warmup, **get_tqdm_kwargs()):
    #             next(itr)
    #     queried = []
    #     # add smoothing for speed benchmark
    #     with get_tqdm(total=self.test_size,
    #                   leave=True, smoothing=0.2) as pbar:
    #         for idx, dp in enumerate(itr):
    #             queried.append(dp)
    #             pbar.update()
    #             if idx == self.test_size - 1:
    #                 break
    #     return queried
