#pragma once

#include <opencv2/core.hpp>

class Gradient {
public:
	void applyOnImage(const cv::Mat& input, cv::Mat& output);
};