cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME "core")
set(PROJECT_NAME "${GLOBAL_PROJECT_NAME}_${LIB_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(CUDAToolkit 10.2 REQUIRED)

add_library(${PROJECT_NAME} SHARED
    include/core.h
    include/img.h
    include/cudaenv.h
    include/custd.h

    src/img.cpp
    src/cudaenv.cpp
    src/custd.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        CUDA::cudart_static
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

add_subdirectory(tests)