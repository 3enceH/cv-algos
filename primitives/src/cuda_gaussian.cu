#include "cuda_gaussian.cuh"
#include <iostream>

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)
#define BORDER_WRAP BORDER_REFLECT_101

constexpr int cKernelSizeMax = 32;

#define cKernel gaussKernel1D

__constant__ float cKernel[cKernelSizeMax];

__global__ void gaussianHorizontalShared(int kernelSize, const uchar* input, uchar* output, int width, int height)
{
    extern __shared__ uchar sInput[]; // blockDim.y * width image data | kernel's bytes

    const int gY = blockIdx.y * blockDim.y + threadIdx.y;
    const int lY = threadIdx.y;
    const int k = kernelSize / 2;

    if (gY >= height) return;

    for (int x = 0; x < width; x++) {
        sInput[OFFSET_ROW_MAJOR(x, lY, width, 1)] = input[OFFSET(x, gY)];
    }

    __syncthreads();

    // first, horizontal pass
    for (int x = 0; x < width; x++) {
        float sum = 0;
        for (int i = -k; i <= k; i++) {
            int xx = BORDER_WRAP(x + i, width);
            sum += cKernel[i + k] * sInput[OFFSET_ROW_MAJOR(xx, lY, width, 1)];
        }
        output[OFFSET(x, gY)] = CLAMP(sum, 0, 255);
    }
}

__global__ void gaussianVerticalShared(int kernelSize, const uchar* input, uchar* output, int width, int height)
{
    extern __shared__ uchar sInput[]; // blockDim.x * height image data

    const int gX = blockIdx.x * blockDim.x + threadIdx.x;
    const int lX = threadIdx.x;
    const int k = kernelSize / 2;

    if (gX >= width) return;

    for (int y = 0; y < height; y++) {
        sInput[OFFSET_ROW_MAJOR(lX, y, blockDim.x, 1)] = input[OFFSET(gX, y)];
    }
    __syncthreads();


    // second, vertical pass 
    for (int y = 0; y < height; y++) {
        float sum = 0;
        for (int i = -k; i <= k; i++) {
            int yy = BORDER_WRAP(y + i, height);
            sum += cKernel[i + k] * sInput[OFFSET_ROW_MAJOR(lX, yy, blockDim.x, 1)];
        }
        output[OFFSET(gX, y)] = CLAMP(sum, 0, 255);
    }
}

__global__ void gaussianHorizontal(float* kernel, int filterSize, const uchar* input, uchar* output, int width, int height)
{
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = filterSize / 2;

    if (idy >= height) return;

    // first, horizontal pass
    for (int x = 0; x < width; x++) {
        float sum = 0;
        for (int i = -k; i <= k; i++) {
            int xx = BORDER_WRAP(x + i, width);
            sum += kernel[i + k] * input[OFFSET(xx, idy)];
        }
        output[OFFSET(x, idy)] = CLAMP(sum, 0, 255);
    }
}

__global__ void gaussianVertical(float* kernel, int filterSize, const uchar* input, uchar* output, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = filterSize / 2;

    if (idx >= width) return;

    // second, vertical pass 
    for (int y = 0; y < height; y++) {
        float sum = 0;
        for (int i = -k; i <= k; i++) {
            int yy = BORDER_WRAP(y + i, height);
            sum += kernel[i + k] * input[OFFSET(idx, yy)];
        }
        output[OFFSET(idx, y)] = CLAMP(sum, 0, 255);
    }
}

GaussianCUDA::GaussianCUDA(std::shared_ptr<CUDAEnv>& deviceEnv, int size, float sigma, int threadsPerBlock) : GaussianBase(size, sigma), deviceEnv(deviceEnv), threadsPerBlock(threadsPerBlock)
{
    assert(size <= cKernelSizeMax);
}

GaussianCUDA::GaussianCUDA(int size, float sigma, int threadsPerBlock) : GaussianCUDA(std::make_shared<CUDAEnv>(), size, sigma, threadsPerBlock)
{
}


void GaussianCUDA::apply(const cv::Mat& input, cv::Mat& output) {
    if (output.empty()) {
        output = std::move(cv::Mat(input.rows, input.cols, input.type()));
    }
    
    const size_t sharedMemPerBlock = deviceEnv->currentDeviceProp().sharedMemPerBlock;
    const size_t totalConstMem = deviceEnv->currentDeviceProp().totalConstMem;
    const int maxThreadsPerBlock = deviceEnv->currentDeviceProp().maxThreadsPerBlock;

    int domainSize = MAX(input.cols, input.rows);
    size_t sharedBytesNeeded = 0;

    if (threadsPerBlock == ThreadsPerBlockDefault && ALIGN_UP(domainSize, 32) >= 2 << 6 ) {
        threadsPerBlock = ALIGN_UP(MIN(maxThreadsPerBlock, domainSize), 16);
        while (ALIGN_UP(threadsPerBlock * MAX(input.cols, input.rows) + _kernel.size() * sizeof(float), 4) > sharedMemPerBlock && threadsPerBlock >= 32) {
            threadsPerBlock /= 2;
        }
        sharedBytesNeeded = ALIGN_UP(threadsPerBlock * MAX(input.cols, input.rows) + _kernel.size() * sizeof(float), 4);
    }
    

    if (sharedBytesNeeded > 0 && sharedBytesNeeded <= sharedMemPerBlock) {
        apply_v2(input.data, output.data, input.cols, input.rows);
        //std::cout << "GaussianFilterCUDA.v2 threadsPerBlock " << threadsPerBlock << std::endl;
    }
    else {
        threadsPerBlock = 32;
        apply_v1(input.data, output.data, input.cols, input.rows);
        //std::cout << "GaussianFilterCUDA.v1 threadsPerBlock " << threadsPerBlock << std::endl;
    }
}

void GaussianCUDA::apply_v1(const uchar* input, uchar* output, int width, int height) {

    if (d_kernel.get() == nullptr) {
        void* devPtr;
        void* hostPtr = (void*)_kernel.data();
        size_t kernelBytes = _kernel.size() * sizeof(float);

        CHECK_CUDA(cudaMalloc(&devPtr, kernelBytes));
        d_kernel.reset((float*)devPtr);

        CHECK_CUDA(cudaMemcpy(devPtr, hostPtr, kernelBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    if (d_data.get() == nullptr) {
        void* devPtr;
        CHECK_CUDA(cudaMalloc(&devPtr, (size_t)width * height));

        d_data.reset((uchar*)devPtr);
    }

    if (d_buffer.get() == nullptr) {
        void* devPtr;
        CHECK_CUDA(cudaMalloc(&devPtr, (size_t)width * height));

        d_buffer.reset((uchar*)devPtr);
    }
    
    int kernelSize = (int)_kernel.size();
    size_t inputBytes = width * height;

    CHECK_CUDA(cudaMemcpy(d_data.get(), input, inputBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

    {
        dim3 block(1, threadsPerBlock);
        dim3 grid(1, ALIGN_UP(height, block.y));
        gaussianHorizontal << <grid, block >> > (d_kernel.get(), kernelSize, d_data.get(), d_buffer.get(), width, height);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    {
        dim3 block(threadsPerBlock);
        dim3 grid(ALIGN_UP(width, block.x));
        gaussianVertical << <grid, block >> > (d_kernel.get(), kernelSize, d_buffer.get(), d_data.get(), width, height);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(output, d_data.get(), inputBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void GaussianCUDA::apply_v2(const uchar* input, uchar* output, int width, int height) {

    if (!symbolUploaded) {
        CHECK_CUDA(cudaMemcpyToSymbol(cKernel, (void*)_kernel.data(), _kernel.size() * sizeof(float)));
        symbolUploaded = true;
    }

    if (d_data.get() == nullptr) {
        void* devPtr;
        CHECK_CUDA(cudaMalloc(&devPtr, (size_t)width * height));

        d_data.reset((uchar*)devPtr);
    }

    if (d_buffer.get() == nullptr) {
        void* devPtr;
        CHECK_CUDA(cudaMalloc(&devPtr, (size_t)width * height));

        d_buffer.reset((uchar*)devPtr);
    }

    int kernelSize = (int)_kernel.size();
    size_t inputBytes = width * height;

    CHECK_CUDA(cudaMemcpy(d_data.get(), input, inputBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

    {
        dim3 block(1, threadsPerBlock);
        dim3 grid(1, ALIGN_UP(height, block.y));
        size_t sharedBytes = ALIGN_UP(block.y * width, 4);
        CHECK(sharedBytes <= deviceEnv->currentDeviceProp().sharedMemPerBlock, "Not enough shared memory");
        gaussianHorizontalShared << <grid, block, sharedBytes >> > (kernelSize, d_data.get(), d_buffer.get(), width, height);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    {
        dim3 block(threadsPerBlock);
        dim3 grid(ALIGN_UP(width, block.x));
        size_t sharedBytes = ALIGN_UP(block.x * height, 4);
        CHECK(sharedBytes <= deviceEnv->currentDeviceProp().sharedMemPerBlock, "Not enough shared memory");
        gaussianVerticalShared << <grid, block, sharedBytes >> > (kernelSize, d_buffer.get(), d_data.get(), width, height);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(output, d_data.get(), inputBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
