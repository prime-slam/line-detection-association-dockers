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

#include "../include/kht/kht.hpp"

namespace kht {

    // Kernel-based Hough transform (KHT) for detecting straight lines in images.
    void run_kht(ListOfLines &result, std::uint8_t *binary_image, std::size_t image_width, std::size_t image_height, std::int32_t cluster_min_size, std::double_t cluster_min_deviation, std::double_t delta, std::double_t kernel_min_height, std::double_t n_sigmas) {
        using namespace detail;
        
        static ListOfChains chains;
        static ListOfClusters clusters;
        static Accumulator accumulator;

        // Group feature pixels from an input binary into clusters of approximately collinear pixels.
        find_chains(chains, binary_image, image_width, image_height, cluster_min_size);

        find_clusters(clusters, chains, cluster_min_deviation, cluster_min_size);

        // Perform the proposed Hough transform voting scheme.
        accumulator.init(image_width, image_height, delta);
        voting(accumulator, clusters, kernel_min_height, n_sigmas);

        // Retrieve the most significant straight lines from the resulting voting map.
        peak_detection(result, accumulator);
    }

}
