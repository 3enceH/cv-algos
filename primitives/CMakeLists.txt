cmake_minimum_required(VERSION 3.5)

set(LIB_NAME "primitives")
set(PROJECT_NAME "${GLOBAL_PROJECT_NAME}_${LIB_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(OpenCV 4.3.0 REQUIRED COMPONENTS core)

add_library(${PROJECT_NAME} SHARED
    include/primitives.h
    include/debug.h
    include/gaussianfilter.h
    include/cuda_gaussianfilter.cuh
    include/gradient.h
    include/nonmaxsupress.h
    include/doublethreshold.h
    include/ccl.h
    
    src/debug.cpp
    src/gaussianfilter.cpp
    src/cuda_gaussianfilter.cu
    src/gradient.cpp
    src/nonmaxsupress.cpp
    src/doublethreshold.cpp
    src/ccl.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME}
	PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        opencv_core

        cva_core
)

add_subdirectory(tests)