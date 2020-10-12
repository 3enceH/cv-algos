#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <sstream>

#include "core.h"

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
	EXPORT CUDAEnv();
};

struct CUDAMemDeleter {
	 void EXPORT operator()(void* ptr);
};

class CUDAImage {
private:
	std::unique_ptr<void, CUDAMemDeleter> devBuffer;
	int width = -1, height = -1;

public:
	CUDAImage(int width, int height);

private:
	void init(void* hostPtr = nullptr);
};