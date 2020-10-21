#pragma once

#include "core.h"
#include <opencv2/core.hpp>

class Hysteresis {
private:
    float weakValue;
    float strongValue;

public:
    EXPORT Hysteresis(float weakValue, float strongValue) : weakValue(weakValue), strongValue(strongValue) {}

    // input CV_32FC1, output CV_8UC1
    void EXPORT apply(const cv::Mat& input, cv::Mat& output);
};
