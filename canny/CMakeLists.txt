cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "canny")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(OpenCV 4.3.0 REQUIRED COMPONENTS core highgui)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(CUDA 11.0 REQUIRED)

add_executable(${PROJECT_NAME}
    include/gaussianfilter.h
    include/sobelfilter.h
    include/cuda_env.h
    include/cuda_gaussianfilter.cuh

    src/main.cpp
    src/gaussianfilter.cpp
    src/sobelfilter.cpp
    src/cuda_gaussianfilter.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
	PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Eigen3::Eigen
        opencv_core
        opencv_highgui
)
