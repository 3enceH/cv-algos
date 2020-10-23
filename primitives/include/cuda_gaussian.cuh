#include "gaussian.h"
#include "cudaenv.h"

class GaussianCUDA : public GaussianBase {
private:
    std::shared_ptr<CUDAEnv> deviceEnv;

    std::unique_ptr<float, CUDAMemDeleter> d_kernel;
    std::unique_ptr<uchar, CUDAMemDeleter> d_buffer, d_data;
    int threadsPerBlock;
    bool symbolUploaded = false;

public:
    static const int ThreadsPerBlockDefault = -1;
    EXPORT GaussianCUDA(std::shared_ptr<CUDAEnv>& deviceEnv, int size, float sigma = 1.f, int threadsPerBlock = ThreadsPerBlockDefault);
    EXPORT GaussianCUDA(int size, float sigma = 1.f, int threadsPerBlock = ThreadsPerBlockDefault);

    void EXPORT apply(const cv::Mat& input, cv::Mat& output) override;

private:
    void apply_v1(const uchar* input, uchar* output, int width, int height);
    void apply_v2(const uchar* input, uchar* output, int width, int height);
};
