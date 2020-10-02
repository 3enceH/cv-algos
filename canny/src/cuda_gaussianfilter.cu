#include "cuda_gaussianfilter.cuh"

#define OFFSET(x, y) (y * width + x)

__global__ void gaussianFilterKernelHorizontal(float* filterFunction, int filterSize, unsigned char* data, int width, int height)
{
    const int idy = blockIdx.x * blockDim.x + threadIdx.x;
	const int k = filterSize / 2;

	if (idy >= height) return;

	// first, horizontal pass
	for (int x = 0; x < width; x++) {
		float sum = 0;
		for (int i = -k; i <= k; i++) {
			int xx = x + i;
			if (xx < 0 || xx >= width) continue;
			sum += filterFunction[i + k] * data[OFFSET(xx, idy)];
		}
		unsigned char value = (unsigned char)sum;
		data[OFFSET(x, idy)] = value > 255 ? 255 : value;
	}
}

__global__ void gaussianFilterKernelVertical(float* filterFunction, int filterSize, unsigned char* data, int width, int height)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int k = filterSize / 2;

	if (idx >= width) return;

	// second, vertical pass 
	for (int y = 0; y < height; y++) {
		float sum = 0;
		for (int i = -k; i <= k; i++) {
			int yy = y + i;
			if (yy < 0 || yy >= height) continue;
			sum += filterFunction[i + k] * data[OFFSET(idx, yy)];
		}
		unsigned char value = (unsigned char)sum;
		data[OFFSET(idx, y)] = value > 255 ? 255 : value;
	}
}

GaussianFilterCUDA::GaussianFilterCUDA(int k, float sigma) : GaussianFilter(k, sigma)
{
	deviceEnv.reset(new CUDAEnv);

	void* devPtr;
	void* hostPtr = (void*)_kernel.data();
	size_t fSize = _kernel.size() * sizeof(float);
	checkCuda(cudaMalloc(&devPtr, fSize));
	devFunctionBuffer.reset(devPtr);
	checkCuda(cudaMemcpy(devPtr, hostPtr, fSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
}

GaussianFilterCUDA::GaussianFilterCUDA(int k, float sigma, std::shared_ptr<CUDAEnv>& deviceEnv) : GaussianFilter(k, sigma), deviceEnv(deviceEnv)
{
}

void GaussianFilterCUDA::initBuffers(int width, int height) {
	if (devDataBuffer.get() == nullptr) {
		void* devPtr;
		checkCuda(cudaMalloc(&devPtr, (size_t)width * height));
		devDataBuffer.reset(devPtr);
	}
}

#define ALIGN_UP(x,size) ( (x+(size-1))&(~(size-1)) )

void GaussianFilterCUDA::applyOnImage(cv::Mat& image, int times) {
	const int width = image.cols;
	const int height = image.rows;
	uchar* data = image.data;
	initBuffers(width, height);

	unsigned char* d_data = (unsigned char*)devDataBuffer.get();
	float* d_kernel = (float*)devFunctionBuffer.get();
	int size = (int)_kernel.size();
    size_t bytes = width * height;
    checkCuda(cudaMemcpy(d_data, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

	for (int n = 0; n < times; n++) {
		int unit = 32;
		{
			dim3 block(unit);
			dim3 grid(ALIGN_UP(height, unit));
			gaussianFilterKernelHorizontal << <grid, block >> > (d_kernel, size, d_data, width, height);
		}
		checkCuda(cudaDeviceSynchronize());

		{
			dim3 block(unit);
			dim3 grid(ALIGN_UP(width, unit));
			gaussianFilterKernelVertical << <grid, block >> > (d_kernel, size, d_data, width, height);
		}
		checkCuda(cudaDeviceSynchronize());
	}

    checkCuda(cudaMemcpy(data, d_data, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
