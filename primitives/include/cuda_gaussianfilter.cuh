#include "gaussianfilter.h"
#include "cudaenv.h"

class GaussianFilterCUDA : public GaussianFilterBase {
private:
	std::shared_ptr<CUDAEnv> deviceEnv;

	std::unique_ptr<float, CUDAMemDeleter> d_kernel;
	std::unique_ptr<uchar, CUDAMemDeleter> d_buffer, d_data;

public:
	EXPORT GaussianFilterCUDA(std::shared_ptr<CUDAEnv>& deviceEnv, int k, float sigma = 0.f);
	EXPORT GaussianFilterCUDA(int k, float sigma = 0.f);

	void EXPORT apply(const cv::Mat& image, cv::Mat& output) override;
};
