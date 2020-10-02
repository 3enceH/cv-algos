#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <sstream>

inline void throwError(const std::string message,
	const char* fileName,
	const std::size_t lineNumber)
{
	std::ostringstream stream;
	stream << message << ", file " << fileName << " line " << lineNumber;
	throw std::runtime_error(stream.str());
}

#define check(condition, msg)  { if(!(condition)) throwError(msg, __FILE__, __LINE__); }
#define checkCuda(status)  { if((status) != cudaSuccess) throwError("CUDA error: " + std::string(cudaGetErrorString(status)), __FILE__, __LINE__); }

class CUDAEnv {
private:
	int deviceCount;
	std::vector<cudaDeviceProp> deviceProps;
	int activeDevice = 0;

public:
	CUDAEnv() {
		checkCuda(cudaGetDeviceCount(&deviceCount));
		check(deviceCount > 0, "No devices");
		deviceProps.resize(deviceCount);
		for (int i = 0; i < deviceCount; i++)
			checkCuda(cudaGetDeviceProperties(&deviceProps[i], i));
		cudaSetDevice(activeDevice);
	}
};

struct CUDAMemDeleter {
	void operator()(void* ptr) { checkCuda(cudaFree(ptr)); }
};

class CUDAImage {
private:
	std::unique_ptr<void, CUDAMemDeleter> devBuffer;
	int width = -1, height = -1;

public:
	CUDAImage(int width, int height) : width(width), height(height) {
		init();
	}
	CUDAImage(const cv::Mat& hostImage)  {
		width = hostImage.cols;
		height = hostImage.rows;
		init(hostImage.data);
	}

private:
	void init(void* hostPtr = nullptr) {
		void* devPtr;
		size_t bytes = (size_t)width * height;
		checkCuda(cudaMalloc(&devPtr, bytes));
		devBuffer.reset(devPtr);
		if (hostPtr != nullptr) {
			checkCuda(cudaMemcpy(devPtr, hostPtr, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
	}
};