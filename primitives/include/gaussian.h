#pragma once

#include <opencv2/core.hpp>

#include "core.h"

class GaussianBase {
protected:
    std::vector<float> _kernel;
    int _size;

public:
    EXPORT GaussianBase(int k, float sigma = 0.f);

    virtual EXPORT ~GaussianBase();

    const std::vector<float>& kernel() const { return _kernel; }

    virtual void EXPORT apply(const cv::Mat& input, cv::Mat& output) = 0;
};

class Gaussian : public GaussianBase {
private:
    cv::Mat buffer;

public:
    EXPORT Gaussian(int k, float sigma = 0.f);

    // input, output: CV_8UC1
    void EXPORT apply(const cv::Mat& input, cv::Mat& output) override;
};