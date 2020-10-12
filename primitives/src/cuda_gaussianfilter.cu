#include "cuda_gaussianfilter.cuh"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

__global__ void gaussianFilterKernelHorizontal(float* filterKernel, int filterSize, const uchar* input, uchar* output, int width, int height)
{
    const int idy = blockIdx.x * blockDim.x + threadIdx.x;
	const int k = filterSize / 2;

	if (idy >= height) return;

	// first, horizontal pass
	for (int x = 0; x < width; x++) {
		float sum = 0;
		for (int i = -k; i <= k; i++) {
			int xx = x + i;
			// wrap mirroring
			if (xx < 0) xx = -xx;
			else if (xx >= width) xx -= xx - width + 1;
			sum += filterKernel[i + k] * input[OFFSET(xx, idy)];
		}
		uchar value = (uchar)sum;
		output[OFFSET(x, idy)] = value > 255 ? 255 : value;
	}
}

__global__ void gaussianFilterKernelVertical(float* filterKernel, int filterSize, const uchar* input, uchar* output, int width, int height)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int k = filterSize / 2;

	if (idx >= width) return;

	// second, vertical pass 
	for (int y = 0; y < height; y++) {
		float sum = 0;
		for (int i = -k; i <= k; i++) {
			int yy = y + i;
			// wrap mirroring
			if (yy < 0) yy = -yy;
			else if (yy >= height) yy -= yy - height + 1;
			sum += filterKernel[(size_t)i + k] * input[OFFSET(idx, yy)];
		}
		uchar value = (uchar)sum;
		output[OFFSET(idx, y)] = value > 255 ? 255 : value;
	}
}

GaussianFilterCUDA::GaussianFilterCUDA(std::shared_ptr<CUDAEnv>& deviceEnv, int k, float sigma) : GaussianFilterBase(k, sigma), deviceEnv(deviceEnv)
{
}

GaussianFilterCUDA::GaussianFilterCUDA(int k, float sigma) : GaussianFilterBase(k, sigma)
{
	deviceEnv.reset(new CUDAEnv);

	void* devPtr;
	void* hostPtr = (void*)_kernel.data();
	size_t fSize = _kernel.size() * sizeof(float);
	checkCuda(cudaMalloc(&devPtr, fSize));
	devFunctionBuffer.reset(devPtr);
	checkCuda(cudaMemcpy(devPtr, hostPtr, fSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
}

void GaussianFilterCUDA::initBuffers(int width, int height) {
	if (devDataBuffers.empty()) {
		for (int i = 0; i < 2; i++) {
			void* devPtr;
			checkCuda(cudaMalloc(&devPtr, (size_t)width * height));
			devDataBuffers.emplace_back(devPtr);
		}
	}
}

void GaussianFilterCUDA::applyOnImage(const cv::Mat& input, cv::Mat& output) {
	const int width = input.cols;
	const int height = input.rows;
	uchar* data = input.data;
	initBuffers(width, height);

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, input.type()));
	}
	uchar* const outputData = output.data;

	float* d_kernel = (float*)devFunctionBuffer.get();
	int size = (int)_kernel.size();
    size_t bytes = width * height;
	uchar* d_data1 = (uchar*)devDataBuffers[0].get();
	uchar* d_data2 = (uchar*)devDataBuffers[1].get();

    checkCuda(cudaMemcpy(d_data1, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

	int unit = 32;
	{
		dim3 block(unit);
		dim3 grid(ALIGN_UP(height, unit));
		gaussianFilterKernelHorizontal << <grid, block >> > (d_kernel, size, d_data1, d_data2, width, height);
	}
	checkCuda(cudaDeviceSynchronize());

	{
		dim3 block(unit);
		dim3 grid(ALIGN_UP(width, unit));
		gaussianFilterKernelVertical << <grid, block >> > (d_kernel, size, d_data2, d_data1, width, height);
	}
	checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(outputData, d_data1, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
