#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class Threshold
{
private:
    float lowThreshold;
    float highThreshold;

public:
    EXPORT Threshold(float lowThreshold, float highThreshold) : lowThreshold(lowThreshold), highThreshold(highThreshold)
    {
    }
    EXPORT Threshold(float threshold) : highThreshold(threshold), lowThreshold(threshold) {}

    // input, output: CV_32FC1
    void EXPORT apply(const cv::Mat& input, cv::Mat& output);
};
