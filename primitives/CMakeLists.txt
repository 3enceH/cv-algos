cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME "primitives")
set(PROJECT_NAME "${GLOBAL_PROJECT_NAME}_${LIB_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(OpenCV 4.3 REQUIRED COMPONENTS core)

add_library(${PROJECT_NAME} SHARED
    include/primitives.h
    include/debug.h
    include/gaussian.h
    include/cuda_gaussian.cuh
    include/gradient.h
    include/nonmaxsupress.h
    include/threshold.h
    include/ccl.h
    include/hysteresis.h
    
    src/debug.cpp
    src/gaussian.cpp
    src/cuda_gaussian.cu
    src/gradient.cpp
    src/nonmaxsupress.cpp
    src/threshold.cpp
    src/ccl.cpp
    src/hysteresis.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 50)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        opencv_core
        
        cva_core
)

add_subdirectory(tests)

install(TARGETS ${PROJECT_NAME})

