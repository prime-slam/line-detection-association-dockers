import tensorpack as tp
import numpy as np
import gzip
from glob import glob
from os.path import join, basename
from os import listdir
from io import BufferedReader, BytesIO


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
    '''Extract Data from .npy Files and yield random datapoints.'''

    ''' NOTE so far cutouts are saved raw inside .npz files, i.e.
        we need to split it into 1-3 sub-cutouts. Currently a random sub-cutout is chosen.
        TODO later only one final cutout will be stored in .npz file, then
        we need to fix line __iter__()'''

    def __init__(self, dataset_dir, rnd=False, slices=None):
        super().__init__()
        # setup list of all dataset-files in dataset_dir
        self.dataset_dir = dataset_dir
        self.data_paths = sorted([ path for path in glob(join(dataset_dir, '*.npz.gz')) ], key=lambda x: (int(basename(x).split('.npz.gz')[0])))
        #self.path_to_id = { path : i for i, path in enumerate(self.data_paths) }
        self.id_to_path = { (int(basename(path).split('.npz.gz')[0])) : path for i, path in enumerate(self.data_paths)}
        self.random = rnd
        self.currentPaths = {}

        # get slices (/range) - only wanted files
        if slices and slices[0].stop != None:
            self.id_to_path = { id : self.id_to_path[id] for s in slices for id in list(self.id_to_path) if id in range(s.start, s.stop) }

        # collect initial information
        self.size = 0
        self.data_lens = {}
        for path_id in self.id_to_path:
            with np.load(gzip.open(self.id_to_path[path_id], 'rb')) as f:
                _n = int(f['n_cutouts'][0])
                self.data_lens[self.id_to_path[path_id]] = _n
                self.size += _n
        # init rng
        self.rng = np.random.RandomState()


    def reset_state(self):
        super(LineData, self).reset_state()
        # start with random file
        self.currentPaths = {}


    def __iter__(self):
        '''yields single [datapoint] aka [cutout, height, left, idx, lbd-desc]'''
        # init data-ids
        from copy import deepcopy

        # load list of filePaths if current list is empty
        if not self.currentPaths:
            self.currentPaths = deepcopy(self.id_to_path)

        dp_array = []
        for _ in range(len(list(self.currentPaths))):
            # randomize if requested
            files = list(self.currentPaths)
            if self.random:
                self.rng.shuffle(files)

            file = files[0]
            del self.currentPaths[file]     # remove from current list to avoid using it twice

            '''
            ids = [ (self.path_to_id[path], data_id) for path, size in self.data_lens.items() for data_id in range(size) ]
            # shuffle data
            if self.random:
                self.rng.shuffle(ids)
            '''
            #import time
            #start_time = time.time()

            gz_file = gzip.open(self.id_to_path[file], 'rb')
            buffered_file = BufferedReader(gz_file)
            buffered_file_data = buffered_file.read(-1)
            file_like_data = BytesIO(buffered_file_data)

            with np.load(file_like_data) as data:
                data_dict = dict(data)

            for data_id in range(self.data_lens[self.id_to_path[file]]):
                # load datapoint
                datapoint = []

                # extract cutout from npz
                pre = str(data_id) + '_'
                _height = data_dict[pre+'target_height'][0]
                _offsets = data_dict[pre+'offset']
                _cutout_raw = data_dict[pre+'cutout']
                cutouts = [ _cutout_raw[int(o):int(o)+_height, :, :] for o in _offsets ]
                # compose and yield datapoint
                _cid = self.rng.randint(len(cutouts))
                #if (_height != max_height):
                #    img = np.resize(img, (max_height, *cutouts.shape[1:]))
                cutout_zero_alpha = cutouts[_cid]
                cutout_zero_alpha[:, :, 3] = 0
                datapoint.append(cutout_zero_alpha)
                datapoint.append(_height)
                datapoint.append(data_dict[pre+'left'][0])
                datapoint.append(data_dict[pre+'label'][0])
                desc = data_dict[pre+'desc']
                datapoint.append(desc)
                len_data = self.data_lens[self.id_to_path[file]]
                dp = [len_data] + datapoint
                dp_array.append(dp)

        #print("--- %s seconds for part 2---" % (time.time() - start_time))
        for dp in dp_array:
            yield dp


    def __len__(self):
        return self.size


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
