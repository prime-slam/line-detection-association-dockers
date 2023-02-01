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

#include "LSWMS.h"

namespace py = pybind11;

py::tuple ComputeFsg(py::array &image) {
    cv::Mat img(image.shape(0), image.shape(1), CV_8UC1,
                static_cast<uint8_t *>(image.mutable_data()));

    auto accuracy_level = 3;
    LSWMS lswms(cv::Size(img.cols, img.rows), accuracy_level, 0, false);

    std::vector<LSEG> lines;
    std::vector<double> scores;

    lswms.run(img, lines, scores);

    auto rows = static_cast<int>(lines.size());
    auto cols = 4;

    py::array_t<float> py_scores(rows);
    py::array_t<float> py_lines({rows, cols});

    for (auto i = 0; i != rows; ++i) {
        py_scores.mutable_at(i) = scores[i];
        for (auto point : {0, 1}) {
            py_lines.mutable_at(i, 2 * point) = lines[i][point].x;
            py_lines.mutable_at(i, 2 * point + 1) = lines[i][point].y;
        }
    }

    return py::make_tuple(py_lines, py_scores);
}

PYBIND11_MODULE(pylswms, m) {
    m.def("detect", &ComputeFsg, R"(
        Detects line segments Weighted Mean Shift.
    )",
          py::arg("img"));
}