#include "gaussianfilter.h"
#include "cuda_env.h"

class GaussianFilterCUDA : public GaussianFilter {
private:
	std::shared_ptr<CUDAEnv> deviceEnv;

	std::unique_ptr<void, CUDAMemDeleter> devDataBuffer;
	std::unique_ptr<void, CUDAMemDeleter> devFunctionBuffer;

public:
	GaussianFilterCUDA() : GaussianFilter() {}
	GaussianFilterCUDA(int k, float sigma, std::shared_ptr<CUDAEnv>& deviceEnv);
	GaussianFilterCUDA(int k, float sigma);

	void applyOnImage(cv::Mat& image, int times = 1) override;

private:
	void initBuffers(int width, int height);
};
