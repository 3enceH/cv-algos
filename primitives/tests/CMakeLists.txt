cmake_minimum_required(VERSION 3.5)

find_package(GTest 1.10 REQUIRED)

add_subdirectory(test_gaussian)
add_subdirectory(test_gradient)
add_subdirectory(test_ccl)