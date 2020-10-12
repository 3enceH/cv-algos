#include "gaussianfilter.h"
#include "cudaenv.h"

class GaussianFilterCUDA : public GaussianFilterBase {
private:
	std::shared_ptr<CUDAEnv> deviceEnv;

	std::vector<std::unique_ptr<void, CUDAMemDeleter>> devDataBuffers;
	std::unique_ptr<void, CUDAMemDeleter> devFunctionBuffer;

public:
	EXPORT GaussianFilterCUDA(std::shared_ptr<CUDAEnv>& deviceEnv, int k, float sigma = 0.f);
	EXPORT GaussianFilterCUDA(int k, float sigma = 0.f);

	void EXPORT applyOnImage(const cv::Mat& image, cv::Mat& output) override;

private:
	void initBuffers(int width, int height) override;
};
