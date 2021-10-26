cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(GTest 1.10 REQUIRED)
find_package(OpenCV 4.3 REQUIRED COMPONENTS highgui)

add_subdirectory(test_gaussian)
add_subdirectory(test_gradient)
add_subdirectory(test_ccl)