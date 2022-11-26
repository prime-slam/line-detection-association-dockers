// Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "GreedyMerger.h"
#include "LsdOpenCV.h"
#include "Utils.h"

namespace py = pybind11;

py::array_t<float> ComputeFsg(py::array &image, double length_threshold) {
    cv::Mat img(image.shape(0), image.shape(1), CV_8UC1,
                static_cast<uint8_t *>(image.mutable_data()));
    upm::GreedyMerger merger(img.size());

    upm::Segments lines;
    upm::LsdOpenCV().detect(img, lines);
    upm::SegmentClusters detected_clusters;
    upm::Segments merged_lines;
    merger.mergeSegments(lines, merged_lines, detected_clusters);

    upm::Segments filtered_lines, noisy_lines;
    upm::filterSegments(lines, detected_clusters, filtered_lines, noisy_lines, length_threshold);

    auto rows = static_cast<int>(filtered_lines.size());
    auto cols = 4;

    py::array_t<float> py_lines({rows, cols});

    for (auto i = 0; i < rows; i++) {
        for (auto j = 0; j < cols; ++j) {
            py_lines.mutable_at(i, j) = filtered_lines[i][j];
        }
    }

    return py_lines;
}

PYBIND11_MODULE(pyfsg, m) {
    m.def("detect", &ComputeFsg, R"(
        Detects line segments using fast segments grouping.
    )",
          py::arg("img"), py::arg("length_threshold") = 30);
}