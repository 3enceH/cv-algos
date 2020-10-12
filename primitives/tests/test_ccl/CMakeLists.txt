cmake_minimum_required(VERSION 3.5)

set(TEST_NAME ccl)
set(PROJECT_NAME "test_${TEST_NAME}")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    ${PROJECT_NAME}.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)

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
        cudart_static

        cva_core
        cva_primitives
)


