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

#include <algorithm>
#include "../include/kht/kht.hpp"

namespace kht {

    namespace detail {

        // Subdivides the chain of feature pixels into sets of most perceptually significant straight line segments.
        inline std::double_t subdivision_procedure(ListOfClusters &clusters, ChainOfPixels const &chain, std::size_t first_index, std::size_t last_index, std::double_t min_deviation, std::int32_t min_size) {
            /* D. G. Lowe
            * Three-dimensional object recognition from single two-dimensional images
            * Artificial Intelligence, Elsevier, 31, 1987, pp. 355-395.
            *
            * Section 4.6
            */

            std::size_t clusters_count = clusters.size();

            Pixel const &first = chain[first_index];
            Pixel const &last = chain[last_index];
            
            // Compute the length of the straight line segment defined by the endpoints of the cluster.
            std::int32_t x = first.x_index - last.x_index;
            std::int32_t y = first.y_index - last.y_index;
            std::double_t length = sqrt(static_cast<std::double_t>(x * x + y * y));
            
            // Find the pixels with maximum deviation from the line segment in order to subdivide the cluster.
            std::size_t max_pixel_index = 0;
            std::double_t deviation, max_deviation = -1.0;

            for (std::size_t i = first_index, count = chain.size(); i != last_index; i = (i + 1) % count)
            {
                Pixel const &current = chain[i];
                
                deviation = static_cast<std::double_t>(abs((current.x_index - first.x_index) * (first.y_index - last.y_index) + (current.y_index - first.y_index) * (last.x_index - first.x_index)));

                if (deviation > max_deviation) {
                    max_pixel_index = i;
                    max_deviation = deviation;
                }
            }
            max_deviation /= length;

            // Compute the ratio between the length of the segment and the maximum deviation.
            std::double_t ratio = length / std::max(max_deviation, min_deviation);

            // Test the number of pixels of the sub-clusters.
            if (static_cast<std::int32_t>(max_pixel_index - first_index + 1) >= min_size && static_cast<std::int32_t>(last_index - max_pixel_index + 1) >= min_size)
            {
                std::double_t ratio1 = subdivision_procedure(clusters, chain, first_index, max_pixel_index, min_deviation, min_size);
                std::double_t ratio2 = subdivision_procedure(clusters, chain, max_pixel_index, last_index, min_deviation, min_size);

                // Test the quality of the sub-clusters against the quality of the current cluster.
                if (ratio1 > ratio || ratio2 > ratio) {
                    return std::max(ratio1, ratio2);
                }
            }

            // Remove the sub-clusters from the list of clusters.
            clusters.resize(clusters_count);

            // Keep current cluster
            clusters.emplace_back(chain.cbegin() + first_index, chain.cbegin() + (last_index + 1));

            return ratio;
        }

        // Creates a list of clusters of approximately collinear feature pixels.
        void find_clusters(ListOfClusters &clusters, ListOfChains const &chains, std::double_t min_deviation, std::int32_t min_size) {
            clusters.clear();
            for (ChainOfPixels const &chain : chains) {
                subdivision_procedure(clusters, chain, 0, chain.size() - 1, min_deviation, min_size);
            }
        }

    }

}
