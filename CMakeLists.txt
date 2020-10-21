cmake_minimum_required(VERSION 3.5)
set(GLOBAL_PROJECT_NAME "cva")
set(CMAKE_DEBUG_POSTFIX "_d")
set(DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data")

project(cv-algos)

add_subdirectory(core)
add_subdirectory(primitives)
add_subdirectory(algorithms)
