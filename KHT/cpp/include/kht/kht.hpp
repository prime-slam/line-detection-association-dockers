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

#ifndef _KHT_
#define _KHT_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

// The namespace of the KHT library.
namespace kht {

    // A 2D line (normal equation parameters).
    struct Line {
        std::double_t rho;
        std::double_t theta;

        inline Line(std::double_t _rho, std::double_t _theta) :
            rho(_rho),
            theta(_theta) {
        }

        inline Line() :
            Line(0.0, 0.0) {
        }

        inline Line(Line const &) = default;
        inline Line(Line &&) = default;

        inline Line & operator=(Line const &) = default;
        inline Line & operator=(Line &&) = default;
    };

    // A list of lines.
    using ListOfLines = std::vector<Line>;

    /* Kernel-based Hough transform (KHT) for detecting straight lines in images.
    *
    * This function performs the KHT procedure over a given binary image and returns a
    * list with the [rho theta] parameters (rho in pixels and theta in degrees) of the
    * detected lines. This implementation assumes that the binary image was obtained
    * using, for instance, a Canny edge detector plus thresholding and thinning.
    *
    * The resulting lines are in the form:
    *
    *     rho = x * cos(theta) + y * sin(theta)
    *
    * and we assume that the origin of the image coordinate system is at the center of
    * the image, with the x-axis growing to the right and the y-axis growing down.
    *
    * The function parameters are:
    *
    *                'result' : This list will be populated with detected lines, sorted
    *                           in descending order of relevance.
    *
    *          'binary_image' : Input binary image buffer (single channel), where 0
    *                           denotes black and 1 to 255 denotes feature pixels.
    *
    *           'image_width' : Image width.
    *
    *          'image_height' : Image height.
    *
    *      'cluster_min_size' : Minimum number of pixels in the clusters of approximately
    *                           collinear feature pixels. The default value is 10.
    *
    * 'cluster_min_deviation' : Minimum accepted distance between a feature pixel and
    *                           the line segment defined by the end points of its cluster.
    *                           The default value is 2.
    *
    *                 'delta' : Discretization step for the parameter space. The default
    *                           value is 0.5.
    *
    *     'kernel_min_height' : Minimum height for a kernel pass the culling operation.
    *                           This property is restricted to the [0,1] range. The
    *                           default value is 0.002.
    *
    *              'n_sigmas' : Number of standard deviations used by the Gaussian kernel
    *                           The default value is 2.
    *
    * It is important to notice that the linking procedure implemented by the kht()
    * function destroys the original image.
    */
    void run_kht(ListOfLines &result, std::uint8_t *binary_image, std::size_t image_width, std::size_t image_height, std::int32_t cluster_min_size = 10, std::double_t cluster_min_deviation = 2.0, std::double_t delta = 0.5, std::double_t kernel_min_height = 0.002, std::double_t n_sigmas = 2.0);

    // The private namespace of the KHT library (don't touch it!).
    namespace detail {

        // Sets 2D buffers to a specified character.
        template<typename Type>
        Type** memset_2d(Type **memblock, Type value, std::size_t size1, std::size_t size2) {
            if (memblock) {
                std::size_t pointers_size = size1 * sizeof(void*);
                std::size_t items_size = size1 * size2 * sizeof(Type);

                std::int8_t *items = &((std::int8_t*)memblock)[pointers_size];
                memset(items, static_cast<std::int32_t>(value), items_size);
            }

            return memblock;
        }

        // Reallocate 2D memory blocks.
        template<typename Type>
        Type** realloc_2d(Type **memblock, std::size_t size1, std::size_t size2) {
            std::size_t pointers_size = size1 * sizeof(void*);
            std::size_t items_size = size1 * size2 * sizeof(Type);

            memblock = static_cast<Type**>(realloc(static_cast<void*>(memblock), pointers_size + items_size));

            void **pointers = (void**)memblock;
            std::int8_t *items = &((std::int8_t*)memblock)[pointers_size];

            for (std::size_t i = 0, j = 0, j_inc = size2 * sizeof(Type); i != size1; ++i, j += j_inc) {
                pointers[i] = &items[j];
            }

            return memblock;
        }

        // Lower and upper bounds definition.
        struct Bounds {
            std::double_t lower;
            std::double_t upper;

            inline Bounds(std::double_t _lower, std::double_t _upper) :
                lower(_lower),
                upper(_upper) {                    
            };

            inline Bounds() :
                Bounds(0.0, 0.0) {
            }

            inline Bounds(Bounds const &) = default;
            inline Bounds(Bounds &&) = default;

            inline Bounds & operator=(Bounds const &) = default;
            inline Bounds & operator=(Bounds &&) = default;
        };

        // A simple accumulator class implementation.
        class Accumulator {
        public:

            // Initialization constructor.
            Accumulator(std::size_t image_width, std::size_t image_height, std::double_t delta) :
                m_image_width(0),
                m_image_height(0),
                m_delta(0.0),
                m_bins(nullptr),
                m_width(0),
                m_height(0),
                m_rho(),
                m_rho_bounds(0.0, 0.0),
                m_theta(),
                m_theta_bounds(0.0, 0.0) {
                init(image_width, image_height, delta);
            }

            // Default class constructor.
            Accumulator() :
                Accumulator(0, 0, 0.0) {
            }
        
            // Class destructor.
            ~Accumulator() {
                free(m_bins);
            }

            // Returns the expected image width.
            inline std::size_t image_width() const {
                return m_image_width;
            }

            // Returns the expected image height.
            inline std::size_t image_height() const {
                return m_image_height;
            }

            // Returns the discretization step.
            inline std::double_t delta() const {
                return m_delta;
            }

            // Returns the accumulator bins ([1,height][1,width] range).
            inline std::int32_t** bins() {
                return m_bins;
            }

            // Returns the accumulator bins ([1,height][1,width] range).
            inline std::int32_t const** bins() const {
                return const_cast<std::int32_t const**>(m_bins);
            }

            // Returns the accumulator width (rho dimention).
            inline std::size_t width() const {
                return m_width;
            }

            // Returns the accumulator height (theta dimention).
            inline std::size_t height() const {
                return m_height;
            }

            // Returns the discretization of the rho dimention (in pixels, [1,width] range).
            inline std::vector<std::double_t> const& rho() const {
                return m_rho;
            }

            // Returns the parameters space limits (rho dimention, in pixels).
            inline Bounds const& rho_bounds() const {
                return m_rho_bounds;
            }

            // Returns the discretization of the theta dimention (in degrees, [1,height] range).
            inline std::vector<std::double_t> const& theta() const {
                return m_theta;
            }

            // Returns the parameters space limits (theta dimentions, in degrees).
            inline
            Bounds const& theta_bounds() const {
                return m_theta_bounds;
            }

            // Set zeros to the accumulator bins.
            inline void clear() {
                memset_2d(m_bins, 0, m_height + 2, m_width + 2);
            }

            // Initializes the accumulator.
            inline void init(std::size_t image_width, std::size_t image_height, std::double_t delta) {
                if (m_delta != delta || m_image_width != image_width || m_image_height != image_height) {
                    m_delta = delta;
                    m_image_width = image_width;
                    m_image_height = image_height;

                    // Rho.
                    std::double_t r = sqrt(static_cast<std::double_t>(image_width * image_width + image_height * image_height));

                    m_width = static_cast<std::size_t>((r + 1) / delta);

                    m_rho.resize(m_width + 2);

                    m_rho[1] = -0.5 * r;
                    for (std::size_t i = 2; i <= m_width; ++i) {
                        m_rho[i] = m_rho[i - 1] + delta;
                    }
                    m_rho[0] = m_rho[m_width];
                    m_rho[m_width + 1] = m_rho[1];
                    
                    m_rho_bounds.lower = -0.5 * r;
                    m_rho_bounds.upper =  0.5 * r;

                    // Theta.
                    m_height = static_cast<std::size_t>(180.0 / delta);

                    m_theta.resize(m_height + 2);

                    m_theta[1] = 0.0;
                    for (std::size_t i = 2; i <= m_height; ++i) {
                        m_theta[i] = m_theta[i - 1] + delta;
                    }
                    m_theta[0] = m_theta[m_height];
                    m_theta[m_height + 1] = m_theta[1];

                    m_theta_bounds.lower = 0.0;
                    m_theta_bounds.upper = 180.0 - delta;

                    // Accumulator bins.
                    m_bins = realloc_2d(m_bins, m_height + 2, m_width + 2);
                }

                clear();
            }
            
        private:

            // Expected image width.
            std::size_t m_image_width;

            // Expected image height.
            std::size_t m_image_height;

            // Discretization step.
            std::double_t m_delta;
            
            // Accumulator bins ([1,height][1,width] range).
            std::int32_t **m_bins;

            // Accumulator width (rho dimention).
            std::size_t m_width;

            // Accumulator height (theta dimention).
            std::size_t m_height;

            // Specifies the discretization of the rho dimention ([1,width] range).
            std::vector<std::double_t> m_rho;

            // Parameters space limits (rho dimention, in pixels).
            Bounds m_rho_bounds;

            // Specifies the discretization of the theta dimention ([1,height] range).
            std::vector<std::double_t> m_theta;
            
            // Parameters space limits (theta dimentions, in degrees).
            Bounds m_theta_bounds;
        };

        // A 2x2 matrix.
        using Matrix = std::array<std::double_t, 4>;

        // A feature pixel.
        struct Pixel {
            std::int32_t x_index;
            std::int32_t y_index;

            std::double_t x;
            std::double_t y;

            inline Pixel(std::int32_t _x_index, std::int32_t _y_index, std::double_t _x, std::double_t _y) :
                x_index(_x_index),
                y_index(_y_index),
                x(_x),
                y(_y) {
            }

            inline Pixel() :
                Pixel(0, 0, 0.0, 0.0) {
            }

            inline Pixel(Pixel const &) = default;
            inline Pixel(Pixel &&) = default;

            inline Pixel & operator=(Pixel const &) = default;
            inline Pixel & operator=(Pixel &&) = default;
        };

        // A chain of adjacent feature pixels.
        using ChainOfPixels = std::vector<Pixel>;

        // A cluster of approximately collinear feature pixels.
        struct Cluster {
            ChainOfPixels::const_iterator begin;
            ChainOfPixels::const_iterator end;

            inline Cluster(ChainOfPixels::const_iterator &&_begin, ChainOfPixels::const_iterator &&_end) :
                begin(std::move(_begin)),
                end(std::move(_end)) {
            }

            inline Cluster() :
                Cluster(ChainOfPixels::const_iterator(), ChainOfPixels::const_iterator()) {
            }

            inline Cluster(Cluster const &) = default;
            inline Cluster(Cluster &&) = default;

            inline Cluster & operator=(Cluster const &) = default;
            inline Cluster & operator=(Cluster &&) = default;
        };

        // A list of approximately collinear feature pixels.
        using ListOfClusters = std::vector<Cluster>;

        // A list of chains of feature pixels.
        using ListOfChains = std::vector<ChainOfPixels>;

        // Computes the decomposition of a matrix into matrices composed of its eigenvectors and eigenvalues.
        void eigen(Matrix &vectors, Matrix &values, Matrix const &matrix);

        // Creates a list of clusters of approximately collinear feature pixels.
        void find_clusters(ListOfClusters &clusters, ListOfChains const &chains, std::double_t min_deviation, std::int32_t min_size);

        // Creates a list of chains of neighboring edge pixels.
        void find_chains(ListOfChains &chains, std::uint8_t *binary_image, std::size_t image_width, std::size_t image_height, std::int32_t min_size);
    
        // Identify the peaks of votes (most significant straight lines) in the accmulator.
        void peak_detection(ListOfLines &lines, Accumulator const &accumulator);

        // Performs the proposed Hough transform voting scheme.
        void voting(Accumulator &accumulator, ListOfClusters const &clusters, std::double_t kernel_min_height, std::double_t n_sigmas);

    }

}

#endif
