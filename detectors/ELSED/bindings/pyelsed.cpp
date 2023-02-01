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

#include "ELSED.h"

namespace py = pybind11;

py::tuple ComputeElsed(py::array& image, float sigma = 1, float gradient_threshold = 30,
                       int min_line_length = 15, double line_fit_error_threshold = 0.2,
                       double pixels_to_segment_dist_threshold = 1.5,
                       double validation_threshold = 0.15, bool validate = true,
                       bool treat_junctions = true) {

    cv::Mat img(image.shape(0), image.shape(1), CV_8UC1,
                static_cast<uint8_t*>(image.mutable_data()));

    upm::ELSEDParams params;
    params.sigma = sigma;
    params.ksize = cvRound(sigma * 3 * 2 + 1) | 1;  // Automatic kernel size detection
    params.gradientThreshold = gradient_threshold;
    params.minLineLen = min_line_length;
    params.lineFitErrThreshold = line_fit_error_threshold;
    params.pxToSegmentDistTh = pixels_to_segment_dist_threshold;
    params.validationTh = validation_threshold;
    params.validate = validate;
    params.treatJunctions = treat_junctions;

    upm::ELSED elsed(params);

    auto lines = elsed.detectSalient(img);
    auto rows = static_cast<int>(lines.size());
    auto cols = 4;

    py::array_t<float> py_scores(rows);
    py::array_t<float> py_lines({rows, cols});

    for (auto i = 0; i != rows; ++i) {
        py_scores.mutable_at(i) = lines[i].salience;
        for (auto j = 0; j != cols; ++j) {
            py_lines.mutable_at(i, j) = lines[i].segment[j];
        }
    }

    return py::make_tuple(py_lines, py_scores);
}

PYBIND11_MODULE(pyelsed, m) {
    m.def("detect", &ComputeElsed, R"(
        Computes Enhanced Line SEgment Drawing (ELSED) in the input image.
    )",
          py::arg("img"), py::arg("sigma") = 1, py::arg("gradient_threshold") = 30,
          py::arg("minLineLen") = 15, py::arg("line_fit_error_threshold") = 0.2,
          py::arg("pixels_to_segment_dist_threshold") = 1.5, py::arg("validation_threshold") = 0.15,
          py::arg("validate") = true, py::arg("treat_junctions") = true);
}