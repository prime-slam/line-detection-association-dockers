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

        static std::int32_t const X_OFFSET[8] = {0,  1,  0, -1,  1, -1, -1,  1};
        static std::int32_t const Y_OFFSET[8] = {1,  0, -1,  0,  1,  1, -1, -1};

        // This function complements the linking procedure.
        inline bool next(std::int32_t &x_seed, std::int32_t &y_seed, std::uint8_t const *binary_image, std::size_t image_width, std::size_t image_height) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Algorithm 6
            */

            for (std::size_t i = 0; i != 8; ++i) {
                std::int32_t x = x_seed + X_OFFSET[i];
                if (0 <= x && x < static_cast<std::int32_t>(image_width)) {
                    std::int32_t y = y_seed + Y_OFFSET[i];
                    if (0 <= y && y < static_cast<std::int32_t>(image_height)) {
                        if (binary_image[y * image_width + x]) {
                            x_seed = x;
                            y_seed = y;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        // Creates a chain of neighboring edge pixels.
        inline void linking_procedure(ChainOfPixels &chain, std::uint8_t *binary_image, std::size_t image_width, std::size_t image_height, std::int32_t x_ref, std::int32_t y_ref, std::double_t half_width, std::double_t half_height) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Algorithm 5
            */

            std::int32_t x, y;

            chain.clear();
            
            // Find and add feature pixels to the end of the chain.
            x = x_ref;
            y = y_ref;
            do {
                chain.emplace_back(x, y, x - half_width, y - half_height);
                binary_image[y * image_width + x] = 0;
            }
            while (next(x, y, binary_image, image_width, image_height));

            std::reverse(chain.begin(), chain.end());

            // Find and add feature pixels to the begin of the chain.
            x = x_ref;
            y = y_ref;
            if (next(x, y, binary_image, image_width, image_height)) {
                do {
                    chain.emplace_back(x, y, x - half_width, y - half_height);
                    binary_image[y * image_width + x] = 0;
                }
                while (next(x, y, binary_image, image_width, image_height));
            }
        }

        // Creates a list of chains of neighboring edge pixels.
        void find_chains(ListOfChains &chains, std::uint8_t *binary_image, std::size_t image_width, std::size_t image_height, std::int32_t min_size) {
            std::double_t half_width = 0.5 * image_width;
            std::double_t half_height = 0.5 * image_height;

            chains.clear();
            for (std::int32_t y = 1, y_end = static_cast<std::int32_t>(image_height) - 1; y != y_end; ++y) {
                for (std::int32_t x = 1, x_end = static_cast<std::int32_t>(image_width) - 1; x != x_end; ++x) {
                    if (binary_image[y * image_width + x]) {
                        chains.emplace_back();
                        ChainOfPixels &chain = chains.back();
                        linking_procedure(chain, binary_image, image_width, image_height, x, y, half_width, half_height);
                        if (static_cast<std::int32_t>(chain.size()) < min_size) {
                            chains.pop_back();
                        }
                    }
                }
            }
        }

    }

}
