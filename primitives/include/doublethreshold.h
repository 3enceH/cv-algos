#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class DoubleThreshold {
private:
	float lowThreshold;
	float highThreshold;

public:
	EXPORT DoubleThreshold(float highThreshold, float lowThreshold = NAN) : highThreshold(highThreshold), lowThreshold(lowThreshold){}

	// input, output: CV_32FC1
	void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output);
};
