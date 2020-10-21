#include "cudaenv.h"
#include <iostream>

CUDAEnv::CUDAEnv() {
    checkCuda(cudaGetDeviceCount(&_deviceCount));
    check(_deviceCount > 0, "No devices");
    _deviceProps.resize(_deviceCount);
    for (int i = 0; i < _deviceCount; i++)
        checkCuda(cudaGetDeviceProperties(&_deviceProps[i], i));
    cudaSetDevice(activeDeviceId);
}

void CUDAMemDeleter::operator()(void* ptr) {
    checkCuda(cudaFree(ptr));
}

CUDAImage::CUDAImage(int width, int height) : width(width), height(height) {
    init();
}

void CUDAImage::init(void* hostPtr /*= nullptr*/) {
    void* devPtr;
    size_t bytes = (size_t)width * height;
    checkCuda(cudaMalloc(&devPtr, bytes));
    devBuffer.reset(devPtr);
    if (hostPtr != nullptr) {
        checkCuda(cudaMemcpy(devPtr, hostPtr, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
}
