cmake_minimum_required(VERSION 3.5)

set(LIB_NAME "algorithms")
set(PROJECT_NAME "${GLOBAL_PROJECT_NAME}_${LIB_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(OpenCV 4.3.0 REQUIRED COMPONENTS highgui)
#find_package(CUDA 11.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED
    include/canny.h

    src/canny.cpp
    #src/canny_test.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
	PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../primitives/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        opencv_core
        opencv_highgui

        cva_core
        cva_primitives
)

add_subdirectory(tests)

install(TARGETS ${PROJECT_NAME})
