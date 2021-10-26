cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(TEST_NAME "gradient")
set(PROJECT_NAME "test_${TEST_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    ${PROJECT_NAME}.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE DATA_ROOT=${DATA_ROOT})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        GTest::gtest
        opencv_highgui

        cva_primitives
)

install(TARGETS ${PROJECT_NAME})

