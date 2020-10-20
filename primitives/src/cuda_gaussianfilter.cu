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
			int xx = BORDER_MIRROR(x + i, width);
			sum += filterKernel[i + k] * input[OFFSET(xx, idy)];
		}
		output[OFFSET(x, idy)] = CLAMP(sum, 0, 255);
	}
}

__global__ void gaussianFilterKernelVertical(float* filterKernel, int filterSize, const uchar* input, uchar* output, int width, int height)
{
	//TODO copy kernel, line to shared
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int k = filterSize / 2;

	if (idx >= width) return;

	// second, vertical pass 
	for (int y = 0; y < height; y++) {
		float sum = 0;
		for (int i = -k; i <= k; i++) {
			int yy = BORDER_MIRROR(y + i, height);
			sum += filterKernel[i + k] * input[OFFSET(idx, yy)];
		}
		output[OFFSET(idx, y)] = CLAMP(sum, 0, 255);
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
	d_kernel.reset((float*)devPtr);

	checkCuda(cudaMemcpy(devPtr, hostPtr, fSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
}

void GaussianFilterCUDA::apply(const cv::Mat& input, cv::Mat& output) {
	const int width = input.cols;
	const int height = input.rows;
	uchar* data = input.data;

	if (d_data.get() == nullptr) {
		void* devPtr;
		checkCuda(cudaMalloc(&devPtr, (size_t)width * height));

		d_data.reset((uchar*)devPtr);
	}

	if (d_buffer.get() == nullptr) {
		void* devPtr;
		checkCuda(cudaMalloc(&devPtr, (size_t)width * height));

		d_buffer.reset((uchar*)devPtr);
	}

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, input.type()));
	}

	uchar* const outputData = output.data;

	
	int size = (int)_kernel.size();
    size_t bytes = width * height;

    checkCuda(cudaMemcpy(d_data.get(), data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

	int unit = 32;
	{
		dim3 block(unit);
		dim3 grid(ALIGN_UP(height, unit));
		gaussianFilterKernelHorizontal << <grid, block >> > (d_kernel.get(), size, d_data.get(), d_buffer.get(), width, height);
	}
	checkCuda(cudaDeviceSynchronize());

	{
		dim3 block(unit);
		dim3 grid(ALIGN_UP(width, unit));
		gaussianFilterKernelVertical << <grid, block >> > (d_kernel.get(), size, d_buffer.get(), d_data.get(), width, height);
	}
	checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(outputData, d_data.get(), bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
