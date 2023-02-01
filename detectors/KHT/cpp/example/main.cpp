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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <kht/kht.hpp>

std::double_t const DEGREES_TO_RADIANS = std::atan(1.0) / 45.0;

// The main function.
int main(int argc, char *argv[]) {
    char window_name[512];

    cv::Point p1, p2;
    cv::Mat im, gray, bw;
    kht::ListOfLines lines;

    cv::Scalar yellow(0, 255, 255);

    // Set sample image files and number of most relevant lines.
    std::array<cv::String, 8> filenames{"simple.jpg", "chess.jpg", "road.jpg", "wall.jpg", "board.jpg", "church.jpg", "building.jpg", "beach.jpg"};
    std::array<std::int32_t, 8> relevant_lines{8, 25, 15, 36, 38, 40, 19, 19};

    // Process each one of the images.
    for (std::size_t i = 0; i != filenames.size(); ++i) {
        auto const &filename = filenames[i];
        auto const &lines_count = relevant_lines[i];

        // Load input image.
        im = cv::imread("../../../extra/" + filename);
        std::int32_t height = im.rows, width = im.cols;

        // Convert the input image to a binary edge image.
        cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, bw, 80, 200);

        // Call the kernel-base Hough transform function.
        kht::run_kht(lines, bw.ptr(), width, height);

        // Show current image and its most relevant detected lines.
        sprintf(window_name, "KHT - Image '%s' - %d most relevant lines", filename.c_str(), lines_count);
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        for (std::size_t j = 0; j != lines_count; ++j) {
            auto const &line = lines[j];
            std::double_t rho = line.rho;
            std::double_t theta = line.theta * DEGREES_TO_RADIANS;
            std::double_t cos_theta = cos(theta), sin_theta = sin(theta);

            // Convert from KHT to OpenCV window coordinate system conventions.
            // The KHT implementation assumes row-major memory alignment for
            // images. Also, it assumes that the origin of the image coordinate
            // system is at the center of the image, with the x-axis growing to
            // the right and the y-axis growing down.
            if (sin_theta != 0.0) {
                p1.x = -width * 0.5; p1.y = (rho - p1.x * cos_theta) / sin_theta;
                p2.x = width * 0.5 - 1; p2.y = (rho - p2.x * cos_theta) / sin_theta;
            }
            else {
                p1.x = rho; p1.y = -height * 0.5;
                p2.x = rho; p2.y = height * 0.5 - 1;
            }
            p1.x += width * 0.5; p1.y += height * 0.5;
            p2.x += width * 0.5; p2.y += height * 0.5;

            cv::line(im, p1, p2, yellow, 2, cv::LINE_8);
        }

        cv::imshow(window_name, im);
    }

    cv::waitKey(0);

    return EXIT_SUCCESS;
}
