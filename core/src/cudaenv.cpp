#include "cudaenv.h"

CUDAEnv::CUDAEnv() {
	checkCuda(cudaGetDeviceCount(&deviceCount));
	check(deviceCount > 0, "No devices");
	deviceProps.resize(deviceCount);
	for (int i = 0; i < deviceCount; i++)
		checkCuda(cudaGetDeviceProperties(&deviceProps[i], i));
	cudaSetDevice(activeDevice);
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
