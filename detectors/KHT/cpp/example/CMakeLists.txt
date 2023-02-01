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

cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0074 NEW)

set(VERSION_MAJOR 2)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

project(KHT-Example
  VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
  DESCRIPTION "Kernel-Based Hough Transform (KHT) -- Example"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(KHT REQUIRED)
find_package(OpenCV REQUIRED)

link_libraries(${KHT_LIBRARIES} ${OpenCV_LIBS})
include_directories(${KHT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)