cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME "algorithms")
set(PROJECT_NAME "${GLOBAL_PROJECT_NAME}_${LIB_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
    include/canny.h

    src/canny.cpp
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
    PUBLIC
        cva_primitives
)

add_subdirectory(tests)

install(TARGETS ${PROJECT_NAME})
