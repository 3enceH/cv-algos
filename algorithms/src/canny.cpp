#include "canny.h"

#include "debug.h"
#include <iostream>

Canny::Canny(float lowThreshold, float highThreshold)
{
    gaussFilter = std::make_unique<GaussianCUDA>(5, 1.f);
    gradient = std::make_unique<Gradient>();
    nonMaxSupressor = std::make_unique<NonMaxSupress>();
    threshold = std::make_unique<Threshold>(lowThreshold, highThreshold);
    hys = std::make_unique<Hysteresis>(lowThreshold, highThreshold);
}

void Canny::apply(const cv::Mat& input, cv::Mat& output)
{
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

    threshold->apply(gradients, gradients);
    perfTimer.tag("threshold");

    // output = cv::Mat(gradients.rows, gradients.cols, CV_8UC1);
    // cv::convertScaleAbs(gradients, output);

    hys->apply(gradients, output);
    perfTimer.tag("hysteresis");

    std::cout << perfTimer.summary();
}