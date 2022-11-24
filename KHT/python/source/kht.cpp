/* Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
 *
 * author     : Fernandes, Leandro A. F.
 * e-mail     : laffernandes@ic.uff.br
 * home page  : http://www.ic.uff.br/~laffernandes
 * 
 * This file is part of the reference implementation of the Kernel-Based
 * Hough Transform (KHT). The complete description of the implemented
 * techinique can be found at:
 * 
 *     Leandro A. F. Fernandes, Manuel M. Oliveira
 *     Real-time line detection through an improved Hough transform
 *     voting scheme, Pattern Recognition (PR), Elsevier, 41:1, 2008,
 *     pp. 299-314.
 * 
 *     DOI.........: https://doi.org/10.1016/j.patcog.2007.04.003
 *     Project Page: http://www.ic.uff.br/~laffernandes/projects/kht
 *     Repository..: https://github.com/laffernandes/kht
 * 
 * KHT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * KHT is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with KHT. If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <kht/kht.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

// The gateway function.
py::list kht_wrapper(np::ndarray &binary_image, std::int32_t cluster_min_size, std::double_t cluster_min_deviation, std::double_t delta, std::double_t kernel_min_height, std::double_t n_sigmas) {
    if ((binary_image.get_dtype() != np::dtype::get_builtin<std::uint8_t>() && binary_image.get_dtype() != np::dtype::get_builtin<std::int8_t>()) || binary_image.get_nd() != 2) throw std::runtime_error("The binary_image argument must be a single-channel 8-bit binary image.");
    
    static kht::ListOfLines lines;
    kht::run_kht(lines, (std::uint8_t*)binary_image.get_data(), binary_image.shape(1), binary_image.shape(0), cluster_min_size, cluster_min_deviation, delta, kernel_min_height, n_sigmas);

    py::list result;
    for (auto const &line : lines) {
        result.append(py::make_tuple(line.rho, line.theta));
    }
    return result;
}

// The Python module.
BOOST_PYTHON_MODULE(kht) {
    np::initialize();
    py::def("kht", kht_wrapper, (py::arg("binary_image"), py::arg("cluster_min_size") = 10, py::arg("cluster_min_deviation") = 2.0, py::arg("delta") = 0.5, py::arg("kernel_min_height") = 0.002, py::arg("n_sigmas") = 2.0), "Performs the KHT procedure over a given binary image and returns a list with the (rho, theta) parameters (rho in pixels and theta in degrees) of the detected lines.");
}
