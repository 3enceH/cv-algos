#pragma once

#include "debug.h"
#include "primitives.h"

class CannyEdgeDetector
{
private:
    std::unique_ptr<GaussianFilterCUDA> gaussFilter;
    std::unique_ptr<Gradient> gradient;
    std::unique_ptr<NonMaxSupress> nonMaxSupressor;
    std::unique_ptr<DoubleThreshold> thresholder;
    std::unique_ptr<ConnectedComponentLabeling> labeler;
    
    PerformanceTimer perfTimer;

public:
    CannyEdgeDetector(float lowThreshold, float highThreshold)
    {
        gaussFilter = std::make_unique<GaussianFilterCUDA>(2);
        gradient = std::make_unique<Gradient>();
        nonMaxSupressor = std::make_unique<NonMaxSupress>();
        thresholder = std::make_unique<DoubleThreshold>(lowThreshold, highThreshold);
        labeler = std::make_unique<ConnectedComponentLabeling>();
    }

    void apply(const cv::Mat& input, cv::Mat& output) {
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

        labeler->apply(gradients, output);
        perfTimer.tag("CCL");
    }

    std::string summary() const { return perfTimer.summary(); }
};