#include "canny.h"

#include <iostream>
#include "debug.h"

Canny::Canny(float lowThreshold, float highThreshold)
{
    gaussFilter = std::make_unique<GaussianCUDA>(2);
    gradient = std::make_unique<Gradient>();
    nonMaxSupressor = std::make_unique<NonMaxSupress>();
    thresholder = std::make_unique<Threshold>(lowThreshold, highThreshold);
    labeler = std::make_unique<CCL>();
    hys = std::make_unique<Hysteresis>(lowThreshold, highThreshold);
}

void Canny::apply(const cv::Mat& input, cv::Mat& output) {
    PerformanceTimer perfTimer;
    perfTimer.start();

    cv::Mat bufferU8;
    gaussFilter->apply(input, bufferU8);
    perfTimer.tag("gauss");

    cv::Mat combinedGradients;
    gradient->apply(bufferU8, combinedGradients);
    perfTimer.tag("gradient");

    cv::Mat gradients;
    nonMaxSupressor->apply(combinedGradients, gradients);
    perfTimer.tag("nonmax");

    thresholder->apply(gradients, gradients);
    perfTimer.tag("threshold");

    hys->apply(gradients, output);
    perfTimer.tag("hysteresis");

    std::cout << perfTimer.summary();
}