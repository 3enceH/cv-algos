#pragma once

#include <opencv2/core.hpp>

class GaussianFilter {
protected:
	std::vector<float> _kernel;
	int _size;

public:
	GaussianFilter() : _size(-1) {}

	GaussianFilter(int k, float sigma = 1.f);

	virtual ~GaussianFilter() {}

	//const std::vector<float>& function() const { return _kernel; }

	virtual void applyOnImage(cv::Mat& image, int times = 1);
};