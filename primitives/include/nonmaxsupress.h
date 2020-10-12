#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class NonMaxSupress {
public:
	// input: CV_32FC2, output: CV_32FC1
	void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output);
};
