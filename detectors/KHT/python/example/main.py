# Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
#
# author     : Fernandes, Leandro A. F.
# e-mail     : laffernandes@ic.uff.br
# home page  : http://www.ic.uff.br/~laffernandes
# 
# This file is part of the reference implementation of the Kernel-Based
# Hough Transform (KHT). The complete description of the implemented
# techinique can be found at:
# 
#     Leandro A. F. Fernandes, Manuel M. Oliveira
#     Real-time line detection through an improved Hough transform
#     voting scheme, Pattern Recognition (PR), Elsevier, 41:1, 2008,
#     pp. 299-314.
# 
#     DOI.........: https://doi.org/10.1016/j.patcog.2007.04.003
#     Project Page: http://www.ic.uff.br/~laffernandes/projects/kht
#     Repository..: https://github.com/laffernandes/kht
# 
# KHT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# KHT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with KHT. If not, see <https://www.gnu.org/licenses/>.

import cv2, os
from kht import kht
from math import cos, sin, radians
from matplotlib import pyplot as plt
from os import path


# The main function.
def main():
    # Set sample image files and number of most relevant lines.
    base_folder = path.dirname(os.path.abspath(__file__))
    filenames = ["simple.jpg", "chess.jpg", "road.jpg", "wall.jpg", "board.jpg", "church.jpg", "building.jpg", "beach.jpg"]
    relevant_lines = [8, 25, 15, 36, 38, 40, 19, 19]

    # Process each one of the images.
    for (filename, lines_count) in zip(filenames, relevant_lines):
        # Load input image.
        im = cv2.cvtColor(cv2.imread(path.join(base_folder, "..", "..", "extra", filename)), cv2.COLOR_BGR2RGB)
        height, width, _ = im.shape

        # Convert the input image to a binary edge image.
        bw = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 80, 200)

        # Call the kernel-base Hough transform function.
        lines = kht(bw)
        
        # Show current image and its most relevant detected lines.
        plt.imshow(im)

        plt.title("KHT - Image '%s' - %d most relevant lines" % (filename, lines_count))
        plt.autoscale(enable=False)
        plt.xticks([])
        plt.yticks([])

        for (rho, theta) in lines[:lines_count]:
            theta = radians(theta)
            cos_theta, sin_theta = cos(theta), sin(theta)

            # Convert from KHT to Matplotlib's coordinate system conventions.
            # The KHT implementation assumes row-major memory alignment for
            # images. Also, it assumes that the origin of the image coordinate
            # system is at the center of the image, with the x-axis growing to
            # the right and the y-axis growing down.
            if sin_theta != 0:
                x = (-width / 2, width / 2 - 1)
                y = ((rho - x[0] * cos_theta) / sin_theta, (rho - x[1] * cos_theta) / sin_theta)
            else:
                x = (rho, rho)
                y = (-height / 2, height / 2 - 1)
            x = (x[0] + width / 2, x[1] + width / 2)
            y = (y[0] + height / 2, y[1] + height / 2)

            plt.plot(x, y, color='yellow', linewidth=1.0)

        plt.show()


if __name__ == "__main__":
    main()