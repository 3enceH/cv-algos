cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(GLOBAL_PROJECT_NAME "cva")
set(CMAKE_DEBUG_POSTFIX "_d")
set(DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(cv-algos)

add_subdirectory(core)
add_subdirectory(primitives)
#add_subdirectory(algorithms)
