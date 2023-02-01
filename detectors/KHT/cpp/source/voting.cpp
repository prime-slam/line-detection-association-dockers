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

#include <limits>
#include "../include/kht/kht.hpp"

namespace kht {

    namespace detail {

        // pi value.
        static std::double_t const pi = 4 * std::atan(1.0);

        // Conversion factor from radians to degrees.
        static std::double_t const rad_to_deg = 45.0 / std::atan(1.0);

        // An elliptical-Gaussian kernel.
        struct Kernel {
            Cluster const *pcluster;

            std::double_t rho;
            std::double_t theta;

            Matrix lambda;      // [sigma^2_rho sigma_rhotheta; sigma_rhotheta sigma^2_theta]

            std::size_t rho_index;   // [1,rho_size] range
            std::size_t theta_index; // [1,theta_size] range

            std::double_t height;

            inline Kernel(Cluster const *_pcluster, std::double_t _rho, std::double_t _theta, Matrix const &_lambda, std::size_t _rho_index, std::size_t _theta_index, std::double_t _height) :
                pcluster(_pcluster),
                rho(_rho),
                theta(_theta),
                lambda(_lambda),
                rho_index(_rho_index),
                theta_index(_theta_index),
                height(_height) {
            }

            inline Kernel() :
                Kernel(nullptr, 0.0, 0.0, Matrix(), 0, 0, 0.0) {
            }

            inline Kernel(Kernel const &) = default;
            inline Kernel(Kernel &&) = default;

            inline Kernel & operator=(Kernel const &) = default;
            inline Kernel & operator=(Kernel &&) = default;
        };

        // Specifies a list of Gaussian kernels.
        using ListOfKernels = std::vector<Kernel>;

        // Specifies a list of pointers to Gaussian kernels.
        using ListOfKernelsPtr = std::vector<Kernel*>;

        // Bi-variated Gaussian distribution.
        inline std::double_t gauss(std::double_t rho, std::double_t theta, std::double_t sigma2_rho, std::double_t sigma2_theta, std::double_t sigma_rho_sigma_theta, std::double_t two_r, std::double_t a, std::double_t b) {
            return a * exp(-(((rho * rho) / sigma2_rho) - ((two_r * rho * theta) / sigma_rho_sigma_theta) + ((theta * theta) / sigma2_theta)) * b);
        }

        // Bi-variated Gaussian distribution.
        inline std::double_t gauss(std::double_t rho, std::double_t theta, std::double_t sigma2_rho, std::double_t sigma2_theta, std::double_t sigma_rho_theta) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Equation 15
            */

            std::double_t sigma_rho_sigma_theta = sqrt(sigma2_rho * sigma2_theta);
            std::double_t r = (sigma_rho_theta / sigma_rho_sigma_theta), two_r = 2.0 * r;
            std::double_t a = 1.0 / (2.0 * pi * sigma_rho_sigma_theta * sqrt(1.0 - r * r));
            std::double_t b = 1.0 / (2.0 * (1.0 - r * r));
            return gauss(rho, theta, sigma2_rho, sigma2_theta, sigma_rho_sigma_theta, two_r, a, b);
        }

        // Solves the uncertainty propagation.
        inline void solve(Matrix &result, Matrix const &nabla, Matrix const &lambda) {
            Matrix temp;

            std::fill(temp.begin(), temp.end(), 0.0);
            for (std::size_t i = 0, i_line = 0; i < 2; ++i, i_line += 2) {
                for (std::size_t j = 0; j < 2; ++j) {
                    for (std::size_t k = 0; k < 2; ++k) {
                        temp[i_line + j] += nabla[i_line + k] * lambda[k * 2 + j];
                    }
                }
            }

            std::fill(result.begin(), result.end(), 0.0);
            for (std::size_t i = 0, i_line = 0; i < 2; ++i, i_line += 2) {
                for (std::size_t j = 0; j < 2; ++j) {
                    for (std::size_t k = 0; k < 2; ++k) {
                        result[i_line + j] += temp[i_line + k] * nabla[j * 2 + k];
                    }
                }
            }
        }

        // This function complements the proposed voting process.
        inline void vote(Accumulator &accumulator, std::size_t rho_start_index, std::size_t theta_start_index, std::double_t rho_start, std::double_t theta_start, std::ptrdiff_t inc_rho_index, std::ptrdiff_t inc_theta_index, std::double_t sigma2_rho, std::double_t sigma2_theta, std::double_t sigma_rho_theta, std::double_t scale) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Algorithm 4
            */

            std::int32_t **bins = accumulator.bins();
            
            std::size_t rho_size = accumulator.width(), theta_size = accumulator.height();
            std::double_t delta = accumulator.delta();
            std::double_t inc_rho = delta * inc_rho_index, inc_theta = delta * inc_theta_index;
                
            std::double_t sigma_rho_sigma_theta = sqrt(sigma2_rho * sigma2_theta);
            std::double_t r = (sigma_rho_theta / sigma_rho_sigma_theta), two_r = 2.0 * r;
            std::double_t a = 1.0 / (2.0 * pi * sigma_rho_sigma_theta * sqrt(1.0 - r * r));
            std::double_t b = 1.0 / (2.0 * (1.0 - r * r));

            bool theta_voted;
            std::double_t rho, theta;
            std::int32_t votes, theta_not_voted = 0;
            std::size_t rho_index, theta_index, theta_count = 0;
            
            // Loop for the theta coordinates of the parameter space.
            theta_index = theta_start_index;
            theta = theta_start;
            do {
                // Test if the kernel exceeds the parameter space limits.
                if (theta_index == 0 || theta_index == (theta_size + 1)) {
                    rho_start_index = rho_size - rho_start_index + 1;
                    theta_index = theta_index == 0 ? theta_size : 1;
                    inc_rho_index = -inc_rho_index;
                }

                // Loop for the rho coordinates of the parameter space.
                theta_voted = false;

                rho_index = rho_start_index;
                rho = rho_start;
                while ((votes = static_cast<std::int32_t>((gauss(rho, theta, sigma2_rho, sigma2_theta, sigma_rho_sigma_theta, two_r, a, b) * scale) + 0.5)) > 0 && rho_index >= 1 && rho_index <= rho_size) {
                    bins[theta_index][rho_index] += votes;
                    theta_voted = true;

                    rho_index += inc_rho_index;
                    rho += inc_rho;
                }

                if (!theta_voted) {
                    theta_not_voted++;
                }

                theta_index += inc_theta_index;
                theta += inc_theta;
                theta_count++;
            }
            while (theta_not_voted != 2 && theta_count < theta_size);
        }

        // Performs the proposed Hough transform voting scheme.
        void voting(Accumulator &accumulator, ListOfClusters const &clusters, std::double_t kernel_min_height, std::double_t n_sigmas) {
            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Algorithm 2
            */
            static ListOfKernels kernels;
            static ListOfKernelsPtr used_kernels;

            kernels.clear();
            kernels.reserve(clusters.size());

            used_kernels.clear();
            used_kernels.reserve(clusters.size());

            Matrix M, V, S;
            std::double_t &Sxx = M[0], &Syy = M[3], &Sxy = M[1], &Syx = M[2];
            std::double_t &u_x = V[0], &u_y = V[2];
            std::double_t &v_x = V[1], &v_y = V[3];

            std::double_t delta = accumulator.delta();
            std::double_t one_div_delta = 1.0 / delta;
            std::double_t n_sigmas2 = n_sigmas * n_sigmas;
            std::double_t rho_max = accumulator.rho_bounds().upper;

            for (Cluster const &cluster : clusters) {
                std::double_t one_div_npixels = 1 / static_cast<std::double_t>(std::distance(cluster.begin, cluster.end));

                // Alternative reference system definition.
                std::double_t mean_x = 0.0;
                std::double_t mean_y = 0.0;
                for (auto pixel_itr = cluster.begin; pixel_itr != cluster.end; ++pixel_itr) {
                    mean_x += pixel_itr->x;
                    mean_y += pixel_itr->y;
                }
                mean_x *= one_div_npixels;
                mean_y *= one_div_npixels;
                
                Sxx = Syy = Sxy = 0.0;
                for (auto pixel_itr = cluster.begin; pixel_itr != cluster.end; ++pixel_itr) {
                    std::double_t x = pixel_itr->x - mean_x;
                    std::double_t y = pixel_itr->y - mean_y;
                
                    Sxx += (x * x);
                    Syy += (y * y);
                    Sxy += (x * y);
                }
                Syx = Sxy;

                eigen(V, S, M);

                // y_v >= 0 condition verification.
                if (v_y < 0.0) {
                    v_x *= -1.0;
                    v_y *= -1.0;
                }

                // Normal equation parameters computation (Eq. 3).
                std::double_t rho = v_x * mean_x + v_y * mean_y;
                std::double_t theta = acos(v_x) * rad_to_deg;

                std::size_t rho_index = static_cast<std::size_t>(std::abs((rho + rho_max) * one_div_delta)) + 1;
                std::size_t theta_index = static_cast<std::size_t>(std::abs(theta * one_div_delta)) + 1;

                // sigma^2_m' and sigma^2_b' computation, substituting Eq. 5 in Eq. 10.
                std::double_t aux = sqrt(1.0 - v_x * v_x);
                Matrix nabla = {-(u_x * mean_x + u_y * mean_y), 1.0, aux != 0.0 ? (u_x / aux) * rad_to_deg : 0.0, 0.0};

                aux = 0.0;
                for (auto pixel_itr = cluster.begin; pixel_itr != cluster.end; ++pixel_itr) {
                    std::double_t x = (u_x * (pixel_itr->x - mean_x)) + (u_y * (pixel_itr->y - mean_y));
                    aux += (x * x);
                }

                Matrix lambda = {1.0 / aux, 0.0, 0.0, one_div_npixels};

                // Uncertainty from sigma^2_m' and sigma^2_b' to sigma^2_rho,  sigma^2_theta and sigma_rho_theta.
                solve(lambda, nabla, lambda);

                if (lambda[3] == 0.0) {
                    lambda[3] = 0.1;
                }

                lambda[0] *= n_sigmas2;
                lambda[3] *= n_sigmas2;

                // Compute the height of the kernel.
                std::double_t height = gauss(0.0, 0.0, lambda[0], lambda[3], lambda[1]);

                // Keep kernel.
                kernels.emplace_back(&cluster, rho, theta, lambda, rho_index, theta_index, height);
            }

            /* Leandro A. F. Fernandes, Manuel M. Oliveira
            * Real-time line detection through an improved Hough transform voting scheme
            * Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314.
            *
            * Algorithm 3
            */

            // Discard groups with very short kernels.
            std::double_t norm = std::numeric_limits<std::double_t>::min();

            for (Kernel &kernel : kernels) {
                if (norm < kernel.height) {
                    norm = kernel.height;
                }
                used_kernels.push_back(&kernel);
            }
            norm = 1.0 / norm;

            std::size_t i = 0;
            for (std::size_t k = 0, end = used_kernels.size(); k != end; ++k) {
                if ((used_kernels[k]->height * norm) >= kernel_min_height) {
                    if (i != k) {
                        Kernel *temp = used_kernels[i];
                        used_kernels[i] = used_kernels[k];
                        used_kernels[k] = temp;
                    }
                    i++;
                }
            }
            used_kernels.resize(i);

            // Find the g_min threshold and compute the scale factor for integer votes.
            std::double_t kernels_scale = std::numeric_limits<std::double_t>::min();
            for (Kernel *kernel : used_kernels) {
                eigen(V, S, kernel->lambda);
                
                std::double_t radius = sqrt(S[3]);
                std::double_t scale = gauss(V[1] * radius, V[3] * radius, kernel->lambda[0], kernel->lambda[3], kernel->lambda[1]);
                scale = scale < 1.0 ? (1.0 / scale) : 1.0;

                if (kernels_scale < scale) {
                    kernels_scale = scale;
                }
            }

            // Vote for each selected kernel.
            for (Kernel *kernel : used_kernels) {
                vote(accumulator, kernel->rho_index,     kernel->theta_index,        0.0,    0.0,  1,  1, kernel->lambda[0], kernel->lambda[3], kernel->lambda[1], kernels_scale);
                vote(accumulator, kernel->rho_index,     kernel->theta_index - 1,    0.0, -delta,  1, -1, kernel->lambda[0], kernel->lambda[3], kernel->lambda[1], kernels_scale);
                vote(accumulator, kernel->rho_index - 1, kernel->theta_index,     -delta,    0.0, -1,  1, kernel->lambda[0], kernel->lambda[3], kernel->lambda[1], kernels_scale);
                vote(accumulator, kernel->rho_index - 1, kernel->theta_index - 1, -delta, -delta, -1, -1, kernel->lambda[0], kernel->lambda[3], kernel->lambda[1], kernels_scale);
            }
        }

    }

}
