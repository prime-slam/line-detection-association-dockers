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

        // The coordinates of a bin of the accumulator.
        struct Bin {
            std::size_t rho_index;   // [1,rho_size] range.
            std::size_t theta_index; // [1,theta_size] range.

            std::int32_t votes;

            inline Bin(std::size_t _rho_index, std::size_t _theta_index, std::int32_t _votes) :
                rho_index(_rho_index),
                theta_index(_theta_index),
                votes(_votes) {
            }

            inline Bin() :
                Bin(0, 0, 0) {
            }

            inline Bin(Bin const &) = default;
            inline Bin(Bin &&) = default;

            inline Bin & operator=(Bin const &) = default;
            inline Bin & operator=(Bin &&) = default;
        };

        // Specifies a list of accumulator bins.
        using ListOfBins = std::vector<Bin>;

        // An auxiliar data structure that identifies which accumulator bin was visited by the peak detection procedure.
        class VisitedMap {
        public:

            // Initializes the map.
            inline void init(std::size_t accumulator_width, std::size_t accumulator_height) {
                if (m_rho_capacity < (accumulator_width + 2) || m_theta_capacity < (accumulator_height + 2)) {
                    m_rho_capacity = accumulator_width + 2;
                    m_theta_capacity = accumulator_height + 2;

                    m_map = realloc_2d(m_map, m_theta_capacity, m_rho_capacity);
                }

                memset_2d(m_map, false, m_theta_capacity, m_rho_capacity);
            }
            
            // Sets a given accumulator bin as visited.
            inline void set_visited(std::size_t rho_index, std::size_t theta_index) {
                m_map[theta_index][rho_index] = true;
            }

            // Class constructor.
            inline VisitedMap() :
                m_map(nullptr),
                m_rho_capacity(0),
                m_theta_capacity(0) {
            }

            // Class destructor()
            inline ~VisitedMap() {
                free(m_map);
            }

            // Returns whether a neighbour bin was visited already.
            inline bool visited_neighbour(std::size_t rho_index, std::size_t theta_index) const {
                return m_map[theta_index - 1][rho_index - 1] || m_map[theta_index - 1][rho_index] || m_map[theta_index - 1][rho_index + 1] ||
                       m_map[theta_index    ][rho_index - 1] ||                                      m_map[theta_index    ][rho_index + 1] ||
                       m_map[theta_index + 1][rho_index - 1] || m_map[theta_index + 1][rho_index] || m_map[theta_index + 1][rho_index + 1];
            }

        private:

            // The map of flags ([1,theta_size][1,rho_size] range).
            bool **m_map;

            // Specifies the size of allocated storage for the map (rho dimention).
            std::size_t m_rho_capacity;

            // Specifies the size of allocated storage for the map (theta dimention).
            std::size_t m_theta_capacity;
        };

        // Computes the convolution of the given cell with a (discrete) 3x3 Gaussian kernel.
        inline std::int32_t convolution(std::int32_t const **bins, std::size_t rho_index, std::size_t theta_index) {
            return bins[theta_index - 1][rho_index - 1] + bins[theta_index - 1][rho_index + 1] + bins[theta_index + 1][rho_index - 1] + bins[theta_index + 1][rho_index + 1] +
                   bins[theta_index - 1][rho_index    ] + bins[theta_index - 1][rho_index    ] + bins[theta_index    ][rho_index - 1] + bins[theta_index    ][rho_index - 1] + bins[theta_index    ][rho_index + 1] + bins[theta_index    ][rho_index + 1] + bins[theta_index + 1][rho_index    ] + bins[theta_index + 1][rho_index    ] +
                   bins[theta_index    ][rho_index    ] + bins[theta_index    ][rho_index    ] + bins[theta_index    ][rho_index    ] + bins[theta_index    ][rho_index    ];
        }

        // Identify the peaks of votes (most significant straight lines) in the accmulator.
        void peak_detection(ListOfLines &lines, Accumulator const &accumulator) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Section 3.4
            */

            std::int32_t const **bins = accumulator.bins();
            std::vector<std::double_t> const &rho = accumulator.rho();
            std::vector<std::double_t> const &theta = accumulator.theta();

            // Create a list with all cells that receive at least one vote.
            static ListOfBins used_bins;
            
            std::size_t used_bins_count = 0;
            for (std::size_t theta_index = 1, theta_end = accumulator.height() + 1; theta_index != theta_end; ++theta_index) {
                for (std::size_t rho_index = 1, rho_end = accumulator.width() + 1; rho_index != rho_end; ++rho_index) {
                    if (bins[theta_index][rho_index]) {
                        used_bins_count++;
                    }
                }
            }
            
            used_bins.clear();
            used_bins.reserve(used_bins_count);

            for (std::size_t theta_index = 1, theta_end = accumulator.height() + 1; theta_index != theta_end; ++theta_index) {
                for (std::size_t rho_index = 1, rho_end = accumulator.width() + 1; rho_index != rho_end; ++rho_index) {
                    if (bins[theta_index][rho_index]) {
                        used_bins.emplace_back(rho_index, theta_index, convolution(bins, rho_index, theta_index)); // Convolution of the cells with a 3x3 Gaussian kernel
                    }
                }
            }
            
            // Sort the list in descending order according to the result of the convolution.
            std::sort(used_bins.begin(), used_bins.end(), [](Bin const &bin1, Bin const &bin2) { return bin2.votes < bin1.votes; });

            // Use a sweep plane that visits each cell of the list.
            static VisitedMap visited;
            visited.init(accumulator.width(), accumulator.height());

            lines.clear();
            lines.reserve(used_bins_count);

            for (Bin &bin : used_bins) {
                if (!visited.visited_neighbour(bin.rho_index, bin.theta_index)) {
                    lines.emplace_back(rho[bin.rho_index], theta[bin.theta_index]);
                }
                visited.set_visited(bin.rho_index, bin.theta_index);
            }
        }

    }

}
