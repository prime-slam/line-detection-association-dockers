import sys
sys.path.append('..')
import os
import json
import cv2
import numpy as np
import multiprocessing
import time
import shutil
import util.camera as cam
import util.bezier as bez
import util.augment as aug
from config.cfg import parse


def __parallel_handle(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        print(f'Progress: {i}')
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(X)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]


def save_npz(prefix, image, lines, centers, cfg):
    n_pts = cfg.order + 1
    image_size = (image.shape[1], image.shape[0])
    heatmap_size = cfg.heatmap_size
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    centers[:, 0] = np.clip(centers[:, 0] * sx, 0, heatmap_size[0] - 1e-4)
    centers[:, 1] = np.clip(centers[:, 1] * sy, 0, heatmap_size[1] - 1e-4)

    jmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    joff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    cmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    coff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    lvec = np.zeros(((n_pts // 2) * 2, 2,) + heatmap_size[::-1], dtype=np.float32)
    lmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)

    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.round(juncs, 3)
    juncs = np.unique(juncs, axis=0)
    lpos = lines.copy()
    lneg = []

    if n_pts % 2 == 1:
        lines = np.delete(lines, n_pts // 2, axis=1)

    def to_int(x):
        return tuple(map(int, x))

    for c, pts in zip(centers, lines):
        v0, v1 = pts[0], pts[-1]

        cint = to_int(c)
        vint0 = to_int(v0)
        vint1 = to_int(v1)
        jmap[0, vint0[1], vint0[0]] = 1
        jmap[0, vint1[1], vint1[0]] = 1
        joff[:, vint0[1], vint0[0]] = v0 - vint0 - 0.5
        joff[:, vint1[1], vint1[0]] = v1 - vint1 - 0.5
        cmap[0, cint[1], cint[0]] = 1
        coff[:, cint[1], cint[0]] = c - cint - 0.5
        lvec[:, :, cint[1], cint[0]] = pts - c

    lvec = lvec.reshape((-1,) + heatmap_size[::-1])
    lmap[0] = bez.insert_line(lmap[0], lpos, color=255) / 255.0

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        lpos=lpos,
        lneg=lneg,
        jmap=jmap,
        joff=joff,
        cmap=cmap,
        coff=coff,
        lvec=lvec,
        lmap=lmap
    )
    cv2.imwrite(f'{prefix}.png', image)


def json2npz(src_path, dst_path, split, cfg, plot=False):

    json_file = os.path.join(src_path, f'{split}.json')
    try:
        with open(json_file, 'r') as f:
            dataset = json.load(f)
    except Exception:
        return

    if os.path.exists(os.path.join(dst_path, split)):
        shutil.rmtree(os.path.join(dst_path, split))
    os.makedirs(os.path.join(dst_path, split), exist_ok=True)

    tfs = [aug.Noop(), aug.HorizontalFlip(), aug.VerticalFlip(),
           aug.Compose([aug.HorizontalFlip(), aug.VerticalFlip()])]

    def call_back(data):
        filename = data['filename']
        lines0 = np.asarray(data['lines'])
        image0 = cv2.imread(os.path.join(src_path, 'image', filename))

        if cfg.type == 'pinhole':
            camera = cam.Pinhole()
        elif cfg.type == 'fisheye':
            coeff = {'K': np.asarray(data['K']), 'D': np.asarray(data['D'])}
            camera = cam.Fisheye(coeff)
        else:
            image_size = (image0.shape[1], image0.shape[0])
            camera = cam.Spherical(image_size)

        if split == 'train':
            for i in range(len(tfs)):
                image, lines = tfs[i](image0, lines0)
                if cfg.type == 'spherical':
                    lines = camera.truncate_line(lines)
                lines = camera.remove_line(lines, thresh=10.0)
                pts_list = camera.interp_line(lines)
                lines = bez.fit_line(pts_list, order=2)[0]
                centers = lines[:, 1]
                lines = bez.fit_line(pts_list, order=cfg.order)[0]

                prefix = os.path.join(dst_path, split, filename.split('.')[0] + f'_{i}')
                save_npz(prefix, image, lines.copy(), centers, cfg)

                if plot:
                    bez.insert_line(image, lines, color=[0, 0, 255])
                    bez.insert_point(image, lines, color=[255, 0, 0], thickness=2)
                    cv2.namedWindow('image', 0)
                    cv2.imshow('image', image)
                    cv2.waitKey()

        else:
            image, lines = image0.copy(), lines0.copy()
            if cfg.type == 'spherical':
                lines = camera.truncate_line(lines)
                lines = camera.remove_line(lines, thresh=10.0)
            pts_list = camera.interp_line(lines)
            lines = bez.fit_line(pts_list, order=2)[0]
            centers = lines[:, 1]
            lines = bez.fit_line(pts_list, order=cfg.order)[0]

            prefix = os.path.join(dst_path, split, filename.split('.')[0])
            save_npz(f'{prefix}', image, lines.copy(), centers, cfg)

            if plot:
                bez.insert_line(image, lines, color=[0, 0, 255])
                bez.insert_point(image, lines, color=[255, 0, 0], thickness=2)
                cv2.namedWindow('image', 0)
                cv2.imshow('image', image)
                cv2.waitKey()

    parmap(call_back, dataset)


if __name__ == "__main__":
    os.chdir('..')
    # Parameter
    cfg = parse()
    print(cfg)

    # Path
    src_path = cfg.raw_dataset_path
    dst_path = cfg.train_dataset_path
    os.makedirs(dst_path, exist_ok=True)

    start = time.time()
    for split in ['train', 'test']:
        json2npz(src_path, dst_path, split, cfg, plot=False)

    end = time.time()
    print('Time: %f s' % (end - start))
