# WLD: A Wavelet and Learning based Line Descriptor for Line Feature Matching #

This repository contains the implementation of our WLD paper.

**Notice:** This repository does not contain the C++ Code which is a part of the project used for training and testing.
However, we added code that enables the testing functionality solely using python (special thanks goes to Robin Niebergall).
We found that due to small numerical differences, most but not all of the Cutouts are identical when created from this Python version instead of from the original C++ implementation.
Thus the results are not exactly the same, but very similar.

**Publication:**
```
@inproceedings {lange2020wld,
booktitle = {Vision, Modeling, and Visualization},
editor = {Krüger, Jens and Niessner, Matthias and St\"uckler, J\"org},
title = {{WLD: A Wavelet and Learning based Line Descriptor for Line Feature Matching}},
author = {Lange, Manuel and Raisch, Claudio and Schilling, Andreas},
year = {2020},
publisher = {The Eurographics Association},
ISBN = {978-3-03868-123-6},
DOI = {10.2312/vmv.20201186}
}
```

## 1. Prerequisites
We used Python (3.6.9) and the following libraries:
```
opencv-contrib-python (4.0.1.24)
tensorflow (1.12.0)
tensorpack (0.8.9)
pandas (1.0.5)
matplotlib (3.1.1)
```

## 2. Setup
We recommend setting up a python venv. The dependencies can be installed using the provided ```requirements.txt``` file and pip:
```
pip install -r requirements.txt
```
> 👉 Note: Python 3.6 and an appropriate pip version are required.

## 3. Important Options
The code can be used with precomputed lines (and ground truth information) in order to test the network on the same lines which we used for the benchmarks in our paper.  
*--keylines_in "{PATH_TO_KEYLINES}"* reads the line locations from the *cpp.npz* keylines file which we provide for the images of our corridor dataset (see below).

The *cpp.npz* contains the lines for the first/left and for the second/right image.  
*--keylines_use_second_image* reads the second/right images' lines from the file, make sure to provide the path to the right image *--image_in "{PATH_TO_IMAGE}"*.

*--save_results* causes the code to save the descriptors (next to the image file) after calculating them. The descriptors have to be calculated for the left and the right image separately.

*im0.npz/im1.npz* are the files containing the descriptors. These can be read and visualized using the [*LineLearningInterface*](https://github.com/manuellange/LineLearningInterface).  

The corridor dataset including our labeled Ground Truth line information (*cpp.npz*) and also including already computed descriptors (*im0.npz/im1.npz*) from the DLD can be found here [*LineDataset*](https://github.com/manuellange/LineDataset).

## 4. Usage
The results are saved next to the images in `{name}.npz` files containing the descriptors and lines.

Usage with precomputed keylines:
```powershell
python3 cld.py --image_in "{PATH_TO_IMAGE}" --keylines_in "{PATH_TO_KEYLINES}" --cutout_width 27 --cutout_height 100 --gpu 0 test "{PATH_TO_MODEL}" -n 1 --depth 10 --disable_bn --debug --min_len 15 --save_results
```

Usage with precomputed keylines (provide second/right image):
```powershell
python3 cld.py --image_in "{PATH_TO_IMAGE}" --keylines_in "{PATH_TO_KEYLINES}" --keylines_use_second_image --cutout_width 27 --cutout_height 100 --gpu 0 test "{PATH_TO_MODEL}" -n 1 --depth 10 --disable_bn --debug --min_len 15 --save_results
```

Usage without precomputed keylines (lines will be detected):
```powershell
python3 cld.py --image_in "{PATH_TO_IMAGE}" --cutout_width 27 --cutout_height 100 --gpu 0 test "{PATH_TO_MODEL}" -n 1 --depth 10 --disable_bn --debug --min_len 25 --save_results
```

There are additional command line arguments which can be used to influence the process (`image_in`, `cutout_{width, height, count}`, `save_results`).
> 👉 Note: `--save_results` will try to save the `.npz` next to the image file.

