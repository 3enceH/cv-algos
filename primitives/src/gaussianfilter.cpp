#include "gaussianfilter.h"

#include <numeric>
#include <iostream>

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

GaussianFilterBase::GaussianFilterBase(int k, float sigma) {
	_size = 2 * k + 1;
	if (sigma <= 0) {
		sigma = 0.3f * ((_size - 1) * 0.5f - 1) + 0.8f;
	}
	
	_kernel = std::vector<float>(_size);

	const float _2xSqrSigma = 2 * sigma * sigma;
	for (int i = 0; i < _size; i++) {
		int arg = (i - k) * (i - k);
		_kernel[i] = exp(-arg / _2xSqrSigma);
	}
	float alpha = 1.f / std::accumulate(_kernel.begin(), _kernel.end(), 0.f);
	for (auto& value : _kernel) value *= alpha;
}

GaussianFilterBase::~GaussianFilterBase() {}

GaussianFilter::GaussianFilter(int k, float sigma) : GaussianFilterBase(k, sigma) {}

void GaussianFilter::applyOnImage(const cv::Mat& input, cv::Mat& output) {
	assert(_size > 0);
	const int width = input.cols;
	const int height = input.rows;
	const int k = _size / 2;

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, input.type()));
	}
	initBuffers(width, height);

	uchar* inputData = input.data;
	uchar* outputData = buffer.data;

	// first, horizontal pass
#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float sum = 0;
			for (int i = -k; i <= k; i++) {
				int xx = wrapMirror(x, width);
				sum += _kernel[(size_t)i + k] * inputData[OFFSET(xx, y)];
			}
			uchar value = (uchar)sum;
			outputData[OFFSET(x, y)] = value > 255 ? 255 : value;
		}
	}

	inputData = outputData;
	outputData = output.data;

	// second, vertical pass 
#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float sum = 0;
			for (int i = -k; i <= k; i++) {
				int yy = wrapMirror(y, height);
				sum += _kernel[(size_t)i + k] * inputData[OFFSET(x, yy)];
			}
			uchar value = (uchar)sum;
			outputData[OFFSET(x, y)] = value > 255 ? 255 : value;
		}
	}
}

void GaussianFilter::initBuffers(int width, int height) {
	if (buffer.empty()) {
		buffer = cv::Mat(height, width, CV_8UC1);
	}
}