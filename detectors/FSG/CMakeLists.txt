# Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0.0)
project(FSG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
add_library(fsg ${LIB_SOURCES} src/main.cpp)
target_link_libraries(fsg ${OpenCV_LIBS})
target_include_directories(fsg PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(fsg_main src/main.cpp)
target_link_libraries(fsg_main fsg ${OpenCV_LIBS})

add_subdirectory(bindings)