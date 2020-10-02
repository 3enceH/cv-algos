#pragma once

#include <opencv2/core.hpp>

class SobelFilter {
protected:
	std::array<int, 9> _kernelX = { 1, 2, 1, 0, 0, 0, -1 , -2, -1 };
	std::array<int, 9> _kernelY = { 1, 0, -1, 2, 0, -2, 1 , 0, -1 };

public:
	virtual ~SobelFilter() {}

	virtual void applyOnImage(cv::Mat& image, cv::Mat& gradients);
};