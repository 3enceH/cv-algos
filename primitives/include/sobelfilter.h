#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class SobelFilter {
protected:
	std::array<int, 9> _kernelX = { 1, 2, 1, 0, 0, 0, -1 , -2, -1 };
	std::array<int, 9> _kernelY = { 1, 0, -1, 2, 0, -2, 1 , 0, -1 };

public:
	// input: CV_8UC1, output: CV_32FC2
	void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output);
};