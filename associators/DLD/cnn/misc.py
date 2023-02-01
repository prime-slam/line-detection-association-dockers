# from misc.logger import getLogger
from time import time
from cnn import config


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    import numpy as np
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min_data = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min_data).transpose(3, 0, 1, 2)
    max_data = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max_data).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    sq = np.max(data.shape[1:3])
    half = (sq - data.shape[1:3]) / 2.0
    sqshapel = np.floor(half).astype(np.int)
    sqshaper = np.ceil(half).astype(np.int)
    padding = ((0, n ** 2 - data.shape[0]), (sqshapel[0], sqshaper[0]),
               (sqshapel[1], sqshaper[1])) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    tdim = data.ndim
    data = data.reshape((n, n) + data.shape[1:])
    data = data.transpose((0, 2, 1, 3) + tuple(range(4, tdim + 1)))
    data = data.reshape((n * data.shape[1],
                         n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)[:, :, [2, 1, 0, 3]]
    return data


def make_sprite(dataset, shape, output_path, batch_n):
    import numpy as np
    import os
    import cv2
    # if channel == 1:
    images = np.array(dataset).reshape((-1, *shape)).astype(np.float32)
    # else:
    #     images = np.array(dataset).reshape((-1, image_size,
    #                                         image_size,
    #                                         channel)).astype(np.float32)
    sprite = images_to_sprite(images)
    sprite_name = "sprite_{}.png".format(batch_n)
    cv2.imwrite(os.path.join(output_path, sprite_name), sprite)
    return sprite_name


def all_indices(l, el):
    """Find all indices of element in list"""
    indices = []
    start = 0
    while True:
        try:
            indices.append(l.index(el, start))
            start = indices[-1] + 1
        except ValueError:
            return indices


class Timer:

    def __init__(self):
        # self.l = getLogger("Timer")
        self.starts = {}
        self.times = {}

        self._name = None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self.start_timer(self._name)
        return self

    def __exit__(self, *args, **kwargs):
        self.stop_timer(self._name)
        self._name = None

    def start_timer(self, name="tmp"):
        self.starts[name] = time()

    def pause_timer(self, name="tmp"):
        if name not in self.starts:
            # self.l.error("There is no Timer `{}`".format(name))
            return

        tmp = time()
        self.times.setdefault(name, []).append(
            tmp - self.starts[name])

    def stop_timer(self, name="tmp"):
        self.pause_timer(name)

    def print_times(self, name="all", with_blocks=False):
        to_print = [name]
        if name == "all":
            to_print = list(self.times.keys())

        for tname in to_print:
            if tname not in self.times:
                continue
            times = self.times[tname]
            print("====== Timings of {}:\n".format(tname))

            if with_blocks:
                for i, t in enumerate(times):
                    print("Block{}: {}".format(i, t))

            tsum = sum(times)
            print("All {}: {:.8f} Seconds".format(len(times), tsum))
            print("Avg:  {}{:.8f} Seconds".format(" " * len(str(len(times))),
                                                  tsum / len(times)))
            print("=================={}=".format("=" * len(tname)))
            del self.starts[tname]
            del self.times[tname]


timer = Timer()
start_timer = timer.start_timer
pause_timer = timer.pause_timer
stop_timer = timer.stop_timer
print_times = timer.print_times
# finfo(np.float32).max


def split_dists(roc_data):
    import numpy as np
    float32_min = np.finfo(np.float32).min

    p_dists = [[], []]
    n_dists = [[], []]
    i = 0

    for r_data in roc_data:
        labels, left, cnn_dists, lbd_dists = r_data
        mask_pos_b = np.equal(np.expand_dims(labels, 0),
                              np.expand_dims(labels, 1))
        mask_pos = mask_pos_b.astype(np.float32)
        mask_not_pos = np.logical_not(mask_pos).astype(np.float32)
        mask_side_b = np.equal(np.expand_dims(left, 0),
                               np.expand_dims(left, 1))

        # because LBD has no sub cutouts
        mask_pos_lbd = np.logical_or(mask_side_b,
                                     np.logical_not(mask_pos_b))
        mask_pos_lbd = mask_pos_lbd.astype(np.float32)

        # positive distances CNN
        tmpp = cnn_dists + mask_not_pos * float32_min
        np.fill_diagonal(tmpp, -10)

        p_dists[0].append(tmpp.flatten())

        # positive distances LBD
        tmpp = lbd_dists + mask_pos_lbd * float32_min
        p_dists[1].append(tmpp.flatten())

        # negative distances CNN
        tmpn = cnn_dists + float32_min * mask_pos
        n_dists[0].append(tmpn.flatten())
        # negative distances LBD
        tmpn = lbd_dists + float32_min * mask_pos
        n_dists[1].append(tmpn.flatten())

    for i in range(2):
        p_dists[i] = np.concatenate(p_dists[i])
        n_dists[i] = np.concatenate(n_dists[i])
    tap_dists = np.array(p_dists)
    tan_dists = np.array(n_dists)

    return tap_dists, tan_dists


def save_model(model_paths, model, target="", compact=False):
    """Save a model to given dir"""
    from os import path
    from os import makedirs

    import tensorpack as tp

    from tensorpack.tfutils.varmanip import get_checkpoint_path
    from tensorpack.tfutils.export import ModelExporter

    import misc.logger as logger
    _L = logger.getLogger("Saver")

    save_to_modeldir = target is ""

    for model_path in model_paths:
        # get model path
        real_path = get_checkpoint_path(model_path)
        abs_p = path.realpath(model_path)
        if (not path.isfile(abs_p)):
            _L.error("{} is not a model file".format(model_path))
            continue

        # save to same folder as model
        if (save_to_modeldir):
            target = path.dirname(abs_p)

        # make sure the folder exists
        if not path.exists(target):
            makedirs(target)

        conf = tp.PredictConfig(
            session_init=tp.get_model_loader(model_path),
            model=model,
            input_names=["input"],
            output_names=["emb"])

        exporter = ModelExporter(conf)
        if (compact):
            out = path.join(target, "{}.pb".format(path.basename(real_path)))
            _L.info("saving {} to {}".format(path.basename(real_path),
                                             out))
            exporter.export_compact(out)
        else:
            _L.info("compact saving {} to {}".format(path.basename(real_path),
                                                     target))
            exporter.export_serving(target)


def get_predictor(model_path, model=None):
    import tensorpack as tp
    import tensorflow as tf

    if model:
        sess_conf = tp.tfutils.get_default_sess_config()
        sess_conf.log_device_placement = config.LOG_DEVICES
        session_creator = tf.compat.v1.train.ChiefSessionCreator(config=sess_conf)

        pred = tp.OfflinePredictor(
            tp.PredictConfig(
                session_creator=session_creator,
                session_init=tp.get_model_loader(model_path),
                model=model,
                input_names=["input", "heights"],
                output_names=["emb"]))

        def prediction(*inp):
            return pred(*inp)[0], pred.sess

    else:
        sess_conf = tf.ConfigProto(allow_soft_placement=True)
        model_file = "{}.pb".format(model_path)

        def prediction(*inp):
            with tf.Session(config=sess_conf) as sess:
                with tf.gfile.GFile(model_file, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def)
                inp_key = sess.graph.get_tensor_by_name("import/input:0")
                heights_key = sess.graph.get_tensor_by_name("import/heights:0")
                emb = sess.graph.get_tensor_by_name("import/emb:0")

                pred = sess.run(emb, {inp_key: inp[0], heights_key: inp[1]})
                return pred, sess

    return prediction


def test_batch(pairwise_distance, pred, data, batch_n):
    import os
    import tensorpack as tp

    from cnn.modules.lbdmatcher import pairwise_lbd_distance

    import numpy as np

    import misc.logger as logger
    _L = logger.getLogger("test")

    _L.h.terminator = "\r"
    _L.info("Testing Batch #{:3} / {:<3}".format(batch_n + 1,
                                                 config.batch_num))
    imgs = None
    if (config.debug_cutout):
        (batch_imgs, batch_heights, batch_imgs2, batch_left,
         batch_labels, batch_lbd_descs) = data
        imgs = batch_imgs2
    else:
        (batch_imgs, batch_heights, batch_left,
         batch_labels, batch_lbd_descs) = data
        imgs = batch_imgs

    enum_labels = list(enumerate(batch_labels))

    # get cnn descriptors
    if config.time:
        batch_cnn_descs = []
        for i in range(len(batch_imgs)):
            t_img = np.expand_dims(batch_imgs[i], 0)
            t_height = np.expand_dims(batch_heights[i], 0)
            with timer("prediction"):
                desc, session = pred(t_img, t_height)
                batch_cnn_descs.append(desc[0])
        batch_cnn_descs = np.array(batch_cnn_descs)
    else:
        batch_cnn_descs, session = pred(batch_imgs, batch_heights)

    # calculate cnn distance matrix
    cnn_dists = pairwise_distance(batch_cnn_descs)

    # calculate lbd distance matrix
    lbd_dists = pairwise_lbd_distance(batch_lbd_descs)

    if config.save_dists:
        data_folder = tp.logger.get_logger_dir() + "/data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        target = data_folder + "/dists{}.{}.npz".format(config.depth,
                                                        batch_n)
        np.savez(target, batch_labels,
                 batch_left, cnn_dists, lbd_dists)

    img_info = [[], []]

    if config.tp_imgs:
        for i, l in enum_labels:
            # get distance list for current label
            cnn_dist = cnn_dists[i]
            lbd_dist = lbd_dists[i]

            # sort labels by distance
            sorted_cnn_label_by_dist = sorted(enum_labels,
                                              key=lambda a: cnn_dist[a[0]])
            sorted_lbd_label_by_dist = sorted(enum_labels,
                                              key=lambda a: lbd_dist[a[0]])
            sorted_cnn_labels = list(zip(*sorted_cnn_label_by_dist))[1]
            sorted_lbd_labels = list(zip(*sorted_lbd_label_by_dist))[1]

            # get relative offset of index
            match_cnn_indices = all_indices(sorted_cnn_labels, l)
            match_lbd_indices = all_indices(sorted_lbd_labels, l)

            # [(<label index>, <label>, <distance>, <index in sorted by distance list>)] # noqa
            sorted_cnn_labels_dists = [(*el, cnn_dist[el[0]], di)
                                       for di, el in
                                       enumerate(sorted_cnn_label_by_dist)]
            sorted_lbd_labels_dists = [(*el, lbd_dist[el[0]], di)
                                       for di, el in
                                       enumerate(sorted_lbd_label_by_dist)]

            img_info[0].append(sorted_cnn_labels_dists[:3] +  # noqa
                               [sorted_cnn_labels_dists[
                                   match_cnn_indices[-1]]])
            img_info[1].append(sorted_lbd_labels_dists[:3] +  # noqa
                               [sorted_lbd_labels_dists[
                                   match_lbd_indices[-1]]])

    _L.h.terminator = "\n"
    _L.info("Testing Batch #{:3} / {:<3}"
            .format(batch_n + 1, config.batch_num))
    print_times("prediction")

    return (batch_cnn_descs, batch_heights, batch_labels, batch_left,
            imgs, img_info,
            cnn_dists, lbd_dists)


def fake_data(dataset, model_num=1, _L=None):
    import numpy as np
    data_out = []

    overflow = 0
    i = 0
    while True:
        if len(data_out) <= i:
            if _L is not None:
                _L.debug("more fake data")
            data_out.append(None)
            for data in dataset:
                if not data_out[i]:
                    data_out[i] = data[:]
                else:
                    for j in range(len(data_out[i])):
                        data_out[i][j] = np.concatenate((data_out[i][j],
                                                         data[j]))
        yield data_out[i]
        i = (i + 1) % config.batch_num

        if not i:
            overflow += 1
            continue

        # remove old data we don't need anymore
        if overflow >= (model_num - 1):
            # _L.debug("remove old data")
            data_out[i - 1] = None


def test(dataset, model_paths, model=None):
    import tensorpack as tp
    import tensorflow as tf

    from tensorpack.tfutils.varmanip import get_checkpoint_path
    from tensorboard.plugins import projector

    import numpy as np

    import misc.logger as logger

    import cnn.plotter as plotter

    import struct
    tf.compat.v1.disable_eager_execution()

    def pairwise(sess):
        from cnn.main import pairwise_distance
        descs = tf.compat.v1.placeholder(tf.float32, (None, 8, ), "descs")
        cnn_dists = pairwise_distance(descs, True)
        pred = tp.OnlinePredictor([descs], [cnn_dists], sess=sess)

        def calculate(*args, **kwargs):
            return pred(*args, **kwargs)[0]

        return calculate

    _L = logger.getLogger("test")  # noqa

    sess = tf.compat.v1.Session()
    cnn_summ = tf.compat.v1.summary.FileWriter(tp.logger.get_logger_dir() + "/cnn")
    lbd_summ = tf.compat.v1.summary.FileWriter(tp.logger.get_logger_dir() + "/lbd")

    cnn_logdir = cnn_summ.get_logdir()

    # [[Match...], [Label...], [Descriptor...]]
    dataset.reset_state()
    data = fake_data(dataset, len(model_paths), _L)

    pairwise_distance = pairwise(sess)

    for model_path in model_paths:
        # get global step
        real_path = get_checkpoint_path(model_path)
        reader = tf.compat.v1.train.NewCheckpointReader(real_path)
        global_step = reader.get_tensor("global_step")

        # predictor
        pred = get_predictor(real_path, model)

        img_info = [[], []]
        imgs = []

        # summaries
        cnn_summaries = tf.compat.v1.Summary()
        lbd_summaries = tf.compat.v1.Summary()

        # collected data for ROC curves
        roc_data = []

        for batch_n in range(config.batch_num):
            # test the batch
            (emb, heights, labels, left,
             timgs, tinfo,
             cnn_dists, lbd_dists) = test_batch(pairwise_distance, pred,
                                                next(data),
                                                batch_n)
            roc_data.append((labels, left, cnn_dists, lbd_dists))

            # save results of CNN to NPZ file
            # if config.save_results is not None:
            #     from pathlib import Path
            #
            #     # convert to path and use NPZ suffix
            #     file = Path(str(config.save_results))
            #     file = file.with_suffix('.npz')
            #     file = str(file)
            #
            #     # save NPZ to given path
            #     np.savez_compressed(file, cutout_list=list(timgs), cnn_desc_list=list(emb), cnn_dist_matrix=cnn_dists,
            #                         label_list=list(labels), left_list=list(left), target_height_list=list(heights))
            #
            #     _L.debug('Saved CNN results to \'{}\''.format(file))
            if config.save_results:
                nested = dataset.ds.ds
                save_op = getattr(nested, 'save_results', None)
                # check if nested dataset has a callable 'save_results' function
                if callable(save_op):
                    nested.save_results(cnn_desc_list=list(emb), label_list=list(labels), left_list=list(left))
                # TODO: warn user that save results is not yet implemented!
                else:
                    pass

            if config.return_results and hasattr(dataset, 'client') and dataset.client:
                message = struct.pack('II', *cnn_dists.shape[:2])
                message += cnn_dists.tobytes()
                dataset.client.send('c', message, wait=False)

            # generate image output
            if config.tp_imgs:
                cimgs = [np.resize(el, (heights[i], *el.shape[1:]))
                         for i, el in enumerate(timgs)]
                imgs.append(cimgs)
                for i in [0, 1]:
                    img_info[i].append(tinfo[i])

            # generate projection output
            if config.tp_proj:
                mdata_name = "metadata_{}.tsv".format(batch_n)
                mdata_path = "{}/{}".format(cnn_logdir, mdata_name)
                with open(mdata_path, "w") as mfile:
                    for label in labels:
                        mfile.write("{}\n".format(label))
                sprite = make_sprite(timgs, timgs[0].shape,
                                     cnn_logdir, batch_n)
                sprite_size = max(timgs[0].shape)

                embv = tf.Variable(emb, name="embeddings")
                initop = tf.variables_initializer([embv])
                pconf = projector.ProjectorConfig()
                embconf = pconf.embeddings.add()
                embconf.tensor_name = embv.name
                embconf.metadata_path = mdata_name
                embconf.sprite.image_path = sprite
                embconf.sprite.single_image_dim.extend([sprite_size,
                                                        sprite_size])
                projector.visualize_embeddings(cnn_summ, pconf)
                sess.run(initop)
                saver = tf.train.Saver()
                saver.save(sess, "{}/embeddings.ckpt".format(cnn_logdir),
                           batch_n)

        # generate ROC
        tap_dists, tan_dists = split_dists(roc_data)

        cnn_plots = plotter.plot_roc(tap_dists[0], tan_dists[0],
                                     "cnn: {}".format(config.depth),
                                     color="g")
        plotter.plot_roc(tap_dists[1], tan_dists[1], "lbd",
                         color="b", figs=cnn_plots)

        suffix = ""
        for i in range(2):
            img = plotter.plot_to_np(cnn_plots[i])
            img = np.expand_dims(img, axis=0)

            s = tp.summary.create_image_summary("ROC{}/{}".format(suffix,
                                                                  global_step),
                                                img)
            cnn_summaries.value.extend(s.value)
            suffix = "_zoomed"

        # get images
        # add image summary
        if config.tp_imgs:
            for i in range(len(img_info[0])):
                for j in range(len(img_info[0][i])):
                    cnn_info = img_info[0][i][j]
                    lbd_info = img_info[1][i][j]

                    generate_tensorboard_img_summary(cnn_summaries, cnn_info,
                                                     imgs[i], "imgs/cnn",
                                                     global_step, i)
                    generate_tensorboard_img_summary(lbd_summaries, lbd_info,
                                                     imgs[i], "imgs/lbd",
                                                     global_step, i)

        cnn_summ.add_summary(cnn_summaries, global_step)
        lbd_summ.add_summary(lbd_summaries, global_step)
    lbd_summ.flush()
    cnn_summ.flush()
    lbd_summ.close()
    cnn_summ.close()
    sess.close()


def generate_tensorboard_plot_summary(summ, x, y, name, subname):
    import tensorpack as tp
    for i in range(len(x)):
        s = tp.summary.create_scalar_summary(name, y[i])
        summ.add_summary(s, x[i])


def moving_summary():
    steps = {}

    def generate_tensorboard_moving_summary(summ,
                                            values, name, subname):
        import tensorpack as tp
        steps.setdefault(subname, 0)

        for val in values:
            s = tp.summary.create_scalar_summary(name, val)
            summ.add_summary(s, steps[subname])
            steps[subname] += 1
    return generate_tensorboard_moving_summary
generate_tensorboard_moving_summary = moving_summary()  # noqa


def img_summary():
    steps = {}

    def generate_tensorboard_img_summary(summ, img_info, imgs,
                                         name, global_step, b_n):
        import cv2
        import tensorpack as tp
        import numpy as np
        steps.setdefault(name, 0)

        idt = imgs[0].dtype
        h, w = 0, 0
        c = imgs[0].shape[2]
        for (ind, _, _, _) in img_info:
            h = max(h, imgs[ind].shape[0])
            w = max(w, imgs[ind].shape[1])

        b = 10
        outimgs = [
            # query img
            np.zeros((h + b, w + 3, c), dtype=idt),
            # first guessed match
            np.zeros((h + b, w + 6, c), dtype=idt),
            # second guessed match
            np.zeros((h + b, w + 6, c), dtype=idt),
            # actual match
            np.zeros((h + b, w + 6, c), dtype=idt)
        ]

        img_offsets = []
        for i, (ind, label, dist, off) in enumerate(img_info):
            s = 0 if i == 0 else 3
            hd = h - imgs[ind].shape[0]
            if i != 0:
                channel = 1 if (label == img_info[0][1]) else 0
                outimgs[i][:, :, channel] = 233
                outimgs[i][8:h + b - hd, 1:-1, 3] = 255

            outimgs[i][9:h + 9 - hd, s:w + s, :] = imgs[ind]
            outimgs[i][9:h + 9 - hd, s:w + s, 3] = 255

            # if this cutout is one of the first two matches
            if (off in img_offsets):
                # don't show it
                outimgs[i][:, :, 3] = 0
            img_offsets.append(off)

            # resize
            outimgs[i] = cv2.resize(outimgs[i], None, fx=3.5, fy=3.5,
                                    interpolation=cv2.INTER_NEAREST)

            # add offset/distance information
            if i > 0:
                cv2.putText(outimgs[i], "{:8.4f}".format(dist),
                            (int(3 * 3.5), 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0, 255))
                cv2.putText(outimgs[i], "{:3} ({}|{})".format(off, ind, label),
                            (int(3 * 3.5), 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0, 255))
            else:
                cv2.putText(outimgs[i], "({}|{})".format(ind, label),
                            (int(3 * 3.5), 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0, 255))

        fimg = np.array([np.hstack(outimgs)])

        s = tp.summary.create_image_summary("{}/{}/{}".format(name,
                                                              b_n,
                                                              steps[name]),
                                            fimg)
        summ.value.extend(s.value)
        steps[name] += 1
    return generate_tensorboard_img_summary
generate_tensorboard_img_summary = img_summary()  # noqa


# deprecated?
def visualize(model_path, model, iimgs):
    import matplotlib.pyplot as plt
    from os import path
    from matplotlib.image import imread
    import numpy as np
    import tensorpack as tp

    import misc.logger as logger
    _L = logger.getLogger("visualize")

    _L.debug(iimgs)
    images = [imread(img) for img in iimgs]
    imgs = [np.array([img]) for img in images]

    # imgs = [np.expand_dims(img, 0) for img in imgs]
    # print(imgs[0].shape)
    # print(np.expand_dims(imgs[0], 0).shape)
    # print(np.expand_dims(imgs[0], 3).shape)
    # exit()

    config.c_width = imgs[0].shape[2]
    config.c_height = imgs[0].shape[1]

    _L.debug("{} x {} cutouts".format(config.C_WIDTH, config.C_HEIGHT))

    pred = tp.OfflinePredictor(tp.PredictConfig(
        session_init=tp.get_model_loader(model_path),
        model=model(config.depth, config.mode),
        input_names=["input"],
        output_names=["emb"]))

    preds = [pred(el)[0] for el in imgs]

    for i, p in enumerate(preds):
        print("pred{}: ".format(i), end="")
        print(p)

    dists = [np.sum((preds[0] - preds[i])**2, 1)[0]
             for i in range(1, len(preds))]

    for i, d in enumerate(dists):
        print("dist{}: ".format(i), end="")
        print(d)

    file_name = path.basename(iimgs[0])
    name_parts = file_name.split(".")
    class_id, rel_id = name_parts[0].split("_")
    ax = plt.subplot(1, len(images), 1)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.imshow(images[0])
    plt.title("Cls {}/{} - #{}".format(class_id, rel_id, 0))

    indices = sorted(list(range(len(dists))),
                     key=lambda el: dists[el])

    # for i, img in enumerate(images):
    for j, i in enumerate(indices):
        img = images[i + 1]
        file_name = path.basename(iimgs[i + 1])
        name_parts = file_name.split(".")
        class_id, rel_id = name_parts[0].split("_")

        ax = plt.subplot(1, len(images), j + 2)
        ax.set_yticks([])
        ax.set_xticks([])

        plt.imshow(img)
        plt.title("Cls {}/{} - #{}".format(class_id, rel_id, i))

        plt.ylabel("{}".format(preds[i]))
        plt.xlabel("{:E}".format(dists[i]))
    plt.show()
