cmake_minimum_required(VERSION 3.0.0)
project(lswms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

add_library(lswms src/LSWMS.cpp src/LSWMS.h)
target_link_libraries(lswms ${OpenCV_LIBS})
target_include_directories(lswms PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(lswms_demo main.cpp)
target_link_libraries(lswms_demo lswms ${OpenCV_LIBS})

add_subdirectory(bindings)


