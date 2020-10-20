#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class Gradient {
private:
	cv::Mat bufferX, bufferY;

public:
	// input: CV_8UC1, output: CV_32FC2
	void EXPORT apply(const cv::Mat& input, cv::Mat& output);
};