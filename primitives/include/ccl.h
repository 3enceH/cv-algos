#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class ConnectedComponentLabeling {
public:
	// input CV_32FC1, output: CV_32SC1
	void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output);
};