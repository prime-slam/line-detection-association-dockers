# -*- coding: utf-8 -*-
# Author: Fabian Schweinfurth <fabian@schweinfurth.de>

# modified boilerplate.py from
#    https://github.com/ppwwyyxx/tensorpack/blob/master/examples/boilerplate.py

import signal, os

import tensorflow as tf
from tensorflow_addons.losses.metric_learning \
    import pairwise_distance
import tensorpack as tp

from tensorpack.utils.gpu import change_gpu, get_num_gpu # noqa

# from matplotlib import offsetbox
# import matplotlib.pyplot as plt

from cnn import config

from cnn.resnet import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
from cnn.modules.dataflow import LineData, MyPrefetchDataZMQ, StripData, MyTestDataSpeed # noqa
from cnn.modules.npy_dataflow import LineData as LineDataNPZ
from cnn.modules.imagereader import LineData as LineDataImageReader

from cnn.client import Client
import cnn.misc as cm
from misc import logger

# for line profiling
# from cnn import profile

_L = logger.getLogger("CNN")
# _L.setLevel(logger.DEBUG)
# from cnn.misc import start_timer, pause_timer, print_times

abort_after_epoch = False


class AbortCallback(tp.Callback):
    def _trigger_epoch(self):
        if abort_after_epoch:
            _L.info("Aborted by user.")
            raise tp.StopTraining()

class ROCValidationCallback(tp.Callback):
    '''
       Adds self.validation_predictor (tp.OnlinePredictor) and self.validation_dataset (tp.DataFlow)
       to tp.SimpleTrainer and uses both for validation after each epoch (inside _trigger_epoch()).

       ROC curves will be used for validation and whenever ROC index decreases the model will be saved.
    '''

    def __init__(self, validation_dataset, validation_steps=None):
        self.validation_dataset = validation_dataset
        self.min_roc_idx = 100.0

    def _setup_graph(self):
        input_names = ['input', 'heights', 'left', 'label', 'lbd.descriptor']
        output_names = ['emb'] # get descriptors from model to compute ROC curves
        self.validation_predictor = self.trainer.get_predictor(input_names, output_names)
        self.saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2, save_relative_paths=True)
        self.save_path = os.path.join(tp.logger.get_logger_dir(), 'min-roc')


    def _trigger_epoch(self):
        from cnn.modules.lbdmatcher import pairwise_lbd_distance
        from cnn.misc import fake_data

        # get predicted descriptors and labels
        roc_data = []
        data = fake_data(self.validation_dataset)

        for batch_n in range(config.batch_num):
            batch = next(data)
            descs = self.validation_predictor(batch[0], batch[1], batch[2], batch[3], batch[4])[0]

            # calculate lbd distance matrix
            lbd_dists = pairwise_lbd_distance(batch[4])
            cnn_dists = self.compute_cnn_dists(descs)

            roc_data.append((batch[3], batch[2], cnn_dists, lbd_dists))

        # compute ROC index on all predictions
        roc_idx = self.compute_roc_idx(roc_data)
        # track roc_idx in tensorboard
        self.trainer.monitors.put_scalar('roc-validation/roc_idx', roc_idx)  # roc_idx)
        # save model when roc_idx decreases
        if roc_idx < self.min_roc_idx:
            self.min_roc_idx = roc_idx
            try:
                self.saver.save(tf.get_default_session(), self.save_path, global_step=tf.train.get_global_step(), write_meta_graph=False)
                tp.logger.info("Better Model saved to {}.".format(
                    tf.train.get_checkpoint_state(tp.logger.get_logger_dir()).model_checkpoint_path))
            except (IOError, tf.errors.PermissionDeniedError, tf.errors.ResourceExhaustedError):
                tp.logger.exception("Exception in ROCValidation-Saver")


    def compute_cnn_dists(self, descs):
        import numpy as np
        # from pairwise_distance in metric_loss_ops.py in tensorflow
        # math background: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        """Computes the -squared- pairwise distance matrix with numerical stability.

          output[i, j] = || feature[i, :] - feature[j, :] ||_2

          Args:
            feature: 2-D Tensor of size [number of data, feature dimension].

          Returns:
            pairwise_distances: 2-D Tensor of size [number of data, number of data].
          """

        sq = np.square(descs)
        reduced_sum_1 = np.add.reduce(np.transpose(sq))[:, None]
        reduced_sum_2 = np.transpose(np.add.reduce(np.transpose(sq))[:, None])
        reduced_added_sums = reduced_sum_1 + reduced_sum_2

        multiplied_featureMat = np.matmul(descs, np.transpose(descs))

        pairwise_distances_squared = reduced_added_sums - 2.0 * multiplied_featureMat
        pairwise_distances_squared = pairwise_distances_squared.clip(min=0)

        error_mask = np.less_equal(pairwise_distances_squared, 0.0)

        # This is probably not necessary, as we use squared distances here
        pairwise_distances_squared = np.multiply(pairwise_distances_squared, np.logical_not(error_mask).astype(np.float32))

        num_data = pairwise_distances_squared.shape[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = np.ones_like(pairwise_distances_squared) - np.diag(np.ones([num_data]))

        pairwise_distances_squared = np.multiply(pairwise_distances_squared, mask_offdiagonals)

        return pairwise_distances_squared


    def compute_roc_idx(self, roc_data):
        '''
        Calculate the FPR Rate at a specific TPR Rate
        roc_data: list of tuples like (batch_of_descriptors, batch_of_labels)
        '''
        import numpy as np
        from cnn.misc import split_dists
        import cnn.plotter as plotter

        # prepare ROC
        tap_dists, tan_dists = split_dists(roc_data)

        TPR_list, FPR_list = plotter.get_roc_data(tap_dists[0], tan_dists[0], 200)
        # Calculate the FPR at x percent TPR
        x95 = np.interp(0.95, TPR_list, FPR_list)
        roc_idx = x95 * 100

        print('#DEBUG len(roc_data): {}, roc_data[0][0].shape: {}, roc_data[0][1].shape: {}, roc_idx: {}'.format(
            len(roc_data), roc_data[0][0].shape, roc_data[0][0].shape, roc_idx))

        return roc_idx


def handle_sigusr(sig, frame):
    """Enter ipdb debugger after receiving SIGUSR1"""
    import ipdb
    ipdb.set_trace(frame)


def handle_sigint(sig, frame):
    global abort_after_epoch
    if abort_after_epoch:
        _L.info("Force abort now.")
        raise tp.StopTraining()
        exit(1)
    else:
        abort_after_epoch = True
        _L.info("Aborting after current Epoch."
                " If you want to force-abort, press C-c again")


signal.signal(signal.SIGINT, handle_sigint)
# Windows doesn't support SIGUSR1
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, handle_sigusr)


def _get_pairwise_distances(labels, embeddings, squared):
    # get pairwise distance
    p_dist = pairwise_distance(embeddings, squared=squared)

    # generate 2D mask where mat[i][j] = 1 iff label[i] == label[j]
    mask_a_positive = tf.cast(
        tf.equal(tf.expand_dims(labels, 0),
                 tf.expand_dims(labels, 1)), tf.float32)

    # positive distances
    # filter pairwise distance to leave only positive pairs
    # p_dist is 0 in the diagonal (i == j), so we don't need to filter that
    a_pos_dist = tf.multiply(mask_a_positive, p_dist)

    # negative distanes
    # add max float per anchor to every invalid negative
    #  (every distance that isn't actually a "negative")
    a_neg_dist = p_dist + tf.float32.max * mask_a_positive

    return a_pos_dist, a_neg_dist


def triplet_loss(labels, img_sides, embeddings, margin, squared=False,
                 scope="triplet_loss_online"):

    with tf.name_scope(scope):
        a_pos_dist, a_neg_dist = _get_pairwise_distances(labels, embeddings,
                                                         squared=squared)

        hardest_a_positive_dist = tf.reduce_max(a_pos_dist, axis=1,
                                                keepdims=True)

        hardest_a_negative_dist = tf.reduce_min(a_neg_dist, axis=1,
                                                keepdims=True)

        # combine max dist(a, p) and min dist(a, n) for every anchor
        dist = hardest_a_positive_dist - hardest_a_negative_dist
        t_loss = tf.maximum(dist + margin, 0.0)

        # final loss over all anchors
        t_loss = tf.reduce_mean(t_loss)

        # this is not completely true but it's only for the user, so...
        neg_dist = tf.reduce_mean(hardest_a_negative_dist, name="Nd")
        pos_dist = tf.add(t_loss - margin, neg_dist, name="Pd")

        num_lines = tf.size(labels, name="NLines")

        return t_loss, pos_dist, neg_dist, num_lines


class Model(tp.ModelDesc):
    """Contains the code for the CNN"""

    def __init__(self, depth, mode="preact"):
        """From tensorpack_examples/imagenet-resnet.py"""
        self.mode = mode
        _L.info("Creating Model with Depth of {} and Mode `{}`"
                .format(depth, mode))

        if config.use_dense_net:
            self.num_blocks = {
                10: ([2, 2, 2]),
                16: ([4, 4, 4]),
                22: ([6, 6, 6])
            }[depth]

        else:

            basicblock = (preresnet_basicblock
                          if mode == 'preact'
                          else resnet_basicblock)
            bottleneck = {
                'resnet': resnet_bottleneck,
                'preact': preresnet_bottleneck,
                'se': se_resnet_bottleneck
            }[mode]

            self.num_blocks, self.block_func = {
                10: ([1, 1, 1, 1], basicblock),
                18: ([2, 2, 2, 2], basicblock),
                34: ([3, 4, 6, 3], basicblock),
                50: ([3, 4, 6, 3], bottleneck),
                101: ([3, 4, 23, 3], bottleneck),
                152: ([3, 8, 36, 3], bottleneck)
            }[depth]

    def get_logits(self, image, nfeatures):
        """From tensorpack_examples/imagenet-resnet.py"""
        with tp.argscope([tp.Conv2D, tp.MaxPooling,
                          tp.GlobalAvgPooling, tp.BatchNorm],
                         data_format="NHWC"):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group,
                self.block_func, nfeatures)

    def embed(self, x, heights, nfeatures=8, fixed_length = False,
              l_change_prob=0.25, l_change_delta=1.5):
        """Embed all given tensors into an nfeatures-dim space."""
        # delta will be applied after scaling to [0; 1]
        l_change_delta = l_change_delta / 255.0

        def get_n_logits(el, h):
            delta = tf.random.uniform([], -l_change_delta, l_change_delta)
            prob = tf.cast(tf.random.uniform([]) > l_change_prob, tf.int32)
            prob = tf.cast(prob, tf.float32)
            # # el = tf.image.random_brightness(el, l_change_delta)
            # el = tf.image.adjust_brightness(el, prob * delta)
            # el = tf.clip_by_value(el, 0.0, 1.0)

            # scale values from [0; 1] to [-1; 1]
            el = el * 2 - 1

            el = tf.expand_dims(el, 0)
            # el = tf.image.crop_to_bounding_box(el, 0, 0, h, config.C_WIDTH)

            return self.get_logits(el, nfeatures)

        list_split = 0

        # concat the list of tensors to a tensor
        if isinstance(x, list):
            list_split = len(x)
            x = tf.concat(x, 0)

        # scale values from [0; 255] to [0; 1]
        x /= 255

        if config.use_dense_net:
            from cnn.densenet import create_dense_net
            # dense net
            embeddings = create_dense_net(x, nfeatures)

        else:

            if not fixed_length:
                # the embedding network
                # x = tf.Print(x, [tf.shape(x), x], "x1:", summarize=100)
                # heights = tf.Print(heights, [tf.shape(heights), heights], "heights1:", summarize=300)
                embeddings = tf.map_fn(lambda el: get_n_logits(el[0], el[1]),
                                       (x, heights), dtype=tf.float32)
                # embeddings = tf.Print(embeddings, [tf.shape(embeddings), embeddings], "E1:", summarize=100)

                embeddings = tf.reshape(embeddings, [-1, embeddings.shape[2]])
                # embeddings = tf.Print(embeddings, [tf.shape(embeddings), embeddings], "E2var:", summarize=100000)
            else:
                # scale values from [0; 1] to [-1; 1]
                x = x * 2 - 1
                # x = tf.Print(x, [tf.shape(x), x], "xFix:", summarize=100000)
                embeddings = self.get_logits(x, nfeatures)
                # embeddings = tf.Print(embeddings, [tf.shape(embeddings), embeddings], "E2fix:", summarize=100000)

        # if "x" was a list of tensors, then split the embeddings
        if list_split > 0:
            embeddings = tf.split(embeddings, list_split, 0)

        return embeddings

    def inputs(self):
        """Define inputs for training/validation phase."""
        inputs = [tf.compat.v1.placeholder(tf.float32,
                                           (None, config.C_HEIGHT, config.C_WIDTH, 4),
                                           "input")]
        inputs += [tf.compat.v1.placeholder(tf.int32,
                                            (None,), "heights")]

        if (config.debug_cutout):
            inputs.append(
                tf.compat.v1.placeholder(tf.float32,
                                         (None, config.C_HEIGHT, config.C_WIDTH, 4),
                                         "input2"))

        inputs += [
            tf.compat.v1.placeholder(tf.bool,
                           (None, ),
                           "left"),
            tf.compat.v1.placeholder(tf.int32,
                           (None, ),
                           "label"),
            tf.compat.v1.placeholder(tf.uint8,
                           (None, 32, ),
                           "lbd.descriptor")
        ]
        return inputs

    def loss(self, labels, img_sides, embeddings):
        return triplet_loss(labels, img_sides, embeddings,
                            5., squared=False, scope="loss")

    def build_graph(self, *inputs):
        """Build the Tensor Graph.
        WARNING: don't use function calls other than `tf` or `tp` here.

        Arguments:
        inputs - input tensors from `_get_inputs`"""

        if (config.debug_cutout):
            em, heights, in1, img_sides, labels, desc = inputs
        else:
            em, heights, img_sides, labels, desc = inputs
        embeddings = self.embed(em, heights, 8, config.fixed_length,
                                config.prob, config.brightness_d)

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
            tf.identity(embeddings, name="emb")

        cost, pos_dist, neg_dist, num_lines = self.loss(labels,
                                                        img_sides,
                                                        embeddings)

        self.cost = tf.identity(cost, name="cost")

        tp.summary.add_moving_summary(pos_dist, neg_dist, self.cost, num_lines)

        return self.cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4,
                             trainable=False)
        # return tf.train.AdamOptimizer(lr)
        return tf.train.GradientDescentOptimizer(lr)


def setup_dataflow(base_dir = None, range=None, rnd=None):
    """Setup data generator

    Returns: Dataflow
    """
    from cnn.modules.fio import FBytesIO
    import os

    client_or_folder = None
    if not base_dir:
        client_or_folder = Client(config.ip, config.port)
        (connected, answer) = client_or_folder.connect()
        assert(not answer.error)
        data = answer.data

    else:
        client_or_folder = base_dir
        welcome_file = os.path.join(base_dir, "-1")
        if not os.path.exists(welcome_file):
            _L.critical("The given folder {} \
            doesn't seem to contain expected data".format(base_dir))
            exit(1)
        with open(welcome_file, "rb") as f:
            data = f.read()

    bt = FBytesIO(data)
    it = bt.extract("c")
    # read away the message ID and colon
    while next(it) != b":":
        pass

    # extract cutout width and height for Setup
    cw, ch = bt.unpack("!II")
    config.C_HEIGHT = ch if ch else None
    config.C_WIDTH = cw if cw else None

    _L.debug("Cutout size: {}x{}".format(cw, ch))

    # get Data from Server or Files
    lds = LineData(client_or_folder, range, rnd)  # RNGDataFlow
    ds = lds

    if config.cmd == "train" or not config.return_results:
        # do that in a different process
        ds = MyPrefetchDataZMQ(ds, nr_proc=1)

    # strip unnecessary tmp values
    ds = StripData(ds)
    # batch to max `BATCH_SIZE`
    ds = tp.BatchData(ds, config.BATCH_SIZE, remainder=True, use_list=False)

    if config.debug:
        ds = tp.PrintData(ds)

    ds.client = lds.client

    return ds


def setup_npz_dataflow(base_dir, prefetch=True):
    """Setup data generator

    Returns: Dataflow
    """
    from cnn.modules.npy_dataflow import MyPrefetchDataZMQ as MyNpzPrefetchDataZMQ

    # get Data from Server or Files
    lds = LineDataNPZ(base_dir, True if config.random_data else False, config.range)  # RNGDataFlow
    ds = lds

    if prefetch and (config.cmd == "train" or not config.return_results):
        # do that in a different process
        ds = MyNpzPrefetchDataZMQ(ds, nr_proc=1)

    # strip unnecessary tmp values
    ds = StripData(ds)

    # batch to max `BATCH_SIZE`
    ds = tp.BatchData(ds, config.BATCH_SIZE, remainder=True, use_list=False)
    if config.debug:
        ds = tp.PrintData(ds)

    #ds.client = lds.client

    return ds


def setup_imagereader_dataflow():
    # extract parameters from config (args)
    image_path = str(config.imagereader_image)
    cutout_width = int(config.imagereader_cutout_width)
    cutout_height = int(config.imagereader_cutout_height)
    min_length = int(config.min_len)
    keylines_path = str(config.imagereader_keylines)
    use_right = bool(config.imagereader_keylines_use_right)

    # create ImageReader dataflow
    ds = LineDataImageReader(image_path=image_path, cutout_width=cutout_width, cutout_height=cutout_height,
                             min_length=min_length, keylines_path=keylines_path, use_right=use_right)

    # set cutout width and height for setup
    config.C_WIDTH = cutout_width
    config.C_HEIGHT = cutout_height

    # print debug information about cutouts
    _L.debug("Processed {} cutouts".format(len(ds)))
    _L.debug("Cutout minimum length: {}".format(min_length))
    _L.debug("Cutout size: {}x{}".format(cutout_width, cutout_height))

    # batch to max 'BATCH_SIZE'
    ds = tp.BatchData(ds, config.BATCH_SIZE, remainder=True, use_list=False)
    if config.debug:
        ds = tp.PrintData(ds)

    return ds


def get_config(ds, model, valid_ds=None):
    """Connect data, save model etc.."""
    period = max(100, config.steps_per_epoch // 5)

    # session creator & config
    sess_config = tp.tfutils.get_default_sess_config()
    sess_config.log_device_placement = config.LOG_DEVICES
    session_creator = tp.tfutils.sesscreate.NewSessionCreator(
        config=sess_config)

    # setup callbacks
    callbacks = [
            # tp.ModelSaver(),
            tp.PeriodicTrigger(tp.ModelSaver(max_to_keep=50),
                               every_k_steps=period,
                               every_k_epochs=1),
            tp.ScheduledHyperParamSetter(
                "learning_rate",
                # TODO test this
                # [(0, 1e-3), (10, 1e-4), (20, 1e-7), (30, 1e-9)]),
                [(0, 1e-3), (10, 1e-4), (20, 1e-7), (30, 1e-9), (40, 1e-11)]),
            AbortCallback()
        ]
    if valid_ds is not None:
        callbacks += [ ROCValidationCallback(valid_ds) ] # add ROC validation
        # callbacks += [ tp.InferenceRunner(valid_ds, tp.ScalarStats(['cost'], prefix='val')) ] # add loss validation

    return tp.AutoResumeTrainConfig(
        dataflow=ds,
        model=model,
        session_creator=session_creator,
        callbacks=callbacks,
        extra_callbacks=[
            tp.MovingAverageSummary(),
            tp.ProgressBar(["cost", "loss/Pd", "loss/Nd", "loss/NLines"]),
            tp.MergeAllSummaries(period=period),
            tp.RunUpdateOps()
        ],
        max_epoch=config.max_epoch,
        steps_per_epoch=config.steps_per_epoch
    )


# modified from tensorpack/utils/logger.py
def auto_set_dir(action=None, name=[]):
    """
    Use :func:`logger.set_logger_dir` to set log directory to
    "./train_log/{scriptname}.{name}".
    "scriptname" is the name of the main python file currently running"""
    import sys
    import os
    if name:
        auto_dirname = os.path.join("train_log", ".".join(name))
    else:
        mod = sys.modules['__main__']
        basename = os.path.basename(mod.__file__)
        auto_dirname = os.path.join('train_log',
                                    basename[:basename.rfind('.')])

    tp.logger.set_logger_dir(auto_dirname, action=action)


def run():
    """Start the Training/Testing"""
    import os
    log_name = [str(el) for el in [config.cmd, config.depth, config.out] if el]
    log_action = "k" if not config.log_ask else None
    auto_set_dir(name=log_name, action=log_action)

    with change_gpu(config.gpu):

        if config.npz_input_folder:
            # setup dataflow and get some data to know the height and width
            ds = setup_npz_dataflow(config.npz_input_folder)
            ds.reset_state()
            ds_iter = iter(ds)
            datapoint = next(ds_iter)
            config.C_HEIGHT = datapoint[1][0]
            config.C_WIDTH = datapoint[0].shape[2]
            del ds, ds_iter
            ds = setup_npz_dataflow(config.npz_input_folder)
        elif config.imagereader_image:
            ds = setup_imagereader_dataflow()
        else:
            ds = setup_dataflow(config.input_folder, config.range, config.random_data)

        # setup validation dataflow if requested
        valid_ds = setup_npz_dataflow(config.npz_validation_input_folder, prefetch=False) if config.npz_validation_input_folder else None
        if config.validation_input_folder:
            valid_ds = setup_dataflow(config.validation_input_folder)
            valid_ds.reset_state()

        model = None

        if (config.cmd != "test") or not config.load_frozen:
            model = Model(config.depth, config.mode)

        # SAVE MODEL
        if config.cmd == "save":
            cm.save_model(config.model, model, config.to, config.compact)

        # TEST MODEL
        elif config.cmd == "test":
            # TODO Fix this. this is not working anymore atm.
            # if config.imgs:
            if False:
                cm.visualize(config.model, Model, config.visualize)
            else:
                cm.test(ds, config.model, model)

        # TRAIN MODEL
        else:
            # write config to model directory
            config_file = os.path.join(tp.logger.get_logger_dir(),
                                       "settings.conf")
            with open(config_file, "w") as f:
                config.write(f)

            tp_config = get_config(ds, model, valid_ds=valid_ds)
            if config.load:
                tp_config.session_init = tp.SaverRestore(config.load)
            # trainer = tp.SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
            trainer = tp.SimpleTrainer()
            tp.launch_train_with_config(tp_config, trainer)
