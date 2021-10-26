#include "cudaenv.h"
#include <iostream>

CUDAEnv::CUDAEnv()
{
    CHECK_CUDA(cudaGetDeviceCount(&_deviceCount));
    CHECK(_deviceCount > 0, "No devices");
    _deviceProps.resize(_deviceCount);
    for(int i = 0; i < _deviceCount; i++)
        CHECK_CUDA(cudaGetDeviceProperties(&_deviceProps[i], i));
    cudaSetDevice(activeDeviceId);
}

void CUDAMemDeleter::operator()(void* ptr)
{
    CHECK_CUDA(cudaFree(ptr));
}

CUDAImage::CUDAImage(int width, int height) : width(width), height(height)
{
    init();
}

void CUDAImage::init(void* hostPtr /*= nullptr*/)
{
    void* devPtr;
    size_t bytes = (size_t)width * height;
    CHECK_CUDA(cudaMalloc(&devPtr, bytes));
    devBuffer.reset(devPtr);
    if(hostPtr != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(devPtr, hostPtr, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
}
