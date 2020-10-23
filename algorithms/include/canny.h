#pragma once

#include "core.h"
#include "primitives.h"

class Canny
{
private:
    std::unique_ptr<GaussianCUDA> gaussFilter;
    std::unique_ptr<Gradient> gradient;
    std::unique_ptr<NonMaxSupress> nonMaxSupressor;
    std::unique_ptr<Threshold> threshold;
    std::unique_ptr<Hysteresis> hys;

public:
    EXPORT Canny(float lowThreshold, float highThreshold);

    void EXPORT apply(const cv::Mat& input, cv::Mat& output);
};