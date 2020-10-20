#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class DoubleThreshold {
private:
	float lowThreshold;
	float highThreshold;

public:
	EXPORT DoubleThreshold(float highThreshold, float lowThreshold) : highThreshold(highThreshold), lowThreshold(lowThreshold){}
	EXPORT DoubleThreshold(float threshold) : highThreshold(threshold), lowThreshold(threshold) {}

	// input, output: CV_32FC1
	void EXPORT apply(const cv::Mat& input, cv::Mat& output);
};
