#pragma once

#include <opencv2/core.hpp>

#include "core.h"

class GaussianFilterBase {
protected:
	std::vector<float> _kernel;
	int _size;

public:
	EXPORT GaussianFilterBase(int k, float sigma = 0.f);

	virtual EXPORT ~GaussianFilterBase();

	const std::vector<float>& kernel() const { return _kernel; }

	virtual void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output) = 0;

protected:
	virtual void initBuffers(int width, int height) = 0;
};

class GaussianFilter : public GaussianFilterBase {
private:
	cv::Mat buffer;

public:
	EXPORT GaussianFilter(int k, float sigma = 0.f);

	// input, output: CV_8UC1
	void EXPORT applyOnImage(const cv::Mat& input, cv::Mat& output) override;

private:
	void initBuffers(int width, int height) override;
};