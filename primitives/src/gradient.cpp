#include "gradient.h"

#include "core.h"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width)

void Gradient::applyOnImage(const cv::Mat& input, cv::Mat& output) {
	int width = input.cols;
	int height = input.rows;
	const uchar* const inputData = input.data;

	if (output.empty()) {
		output = std::move(cv::Mat(width, height, input.type()));
	}
	if (gradients.empty()) {
		gradients = std::move(cv::Mat(width, height, CV_32FC1));
	}
	float* const gradDataPtr = (float*)gradients.data;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int gX = _kernelX[AT(0, 0)] * inputData[OFFSET(x - 1, y - 1)] + _kernelX[AT(0, 1)] * inputData[OFFSET(x - 1, y)] + _kernelX[AT(0, 2)] * inputData[OFFSET(x - 1, y + 1)] +
				_kernelX[AT(1, 0)] * inputData[OFFSET(x, y - 1)] + _kernelX[AT(1, 1)] * inputData[OFFSET(x, y)] + _kernelX[AT(1, 2)] * inputData[OFFSET(x + 1, y + 1)] +
				_kernelX[AT(2, 0)] * inputData[OFFSET(x + 1, y - 1)] + _kernelX[AT(2, 1)] * inputData[OFFSET(x + 1, y)] + _kernelX[AT(2, 2)] * inputData[OFFSET(x + 1, y + 1)];

			int gY = _kernelY[AT(0, 0)] * inputData[OFFSET(x - 1, y - 1)] + _kernelY[AT(0, 1)] * inputData[OFFSET(x - 1, y)] + _kernelY[AT(0, 2)] * inputData[OFFSET(x - 1, y + 1)] +
				_kernelY[AT(1, 0)] * inputData[OFFSET(x, y - 1)] + _kernelY[AT(1, 1)] * inputData[OFFSET(x, y)] + _kernelY[AT(1, 2)] * inputData[OFFSET(x + 1, y + 1)] +
				_kernelY[AT(2, 0)] * inputData[OFFSET(x + 1, y - 1)] + _kernelY[AT(2, 1)] * inputData[OFFSET(x + 1, y)] + _kernelY[AT(2, 2)] * inputData[OFFSET(x + 1, y + 1)];
			
			double mag = std::hypot(gX, gY);
			output.data[OFFSET(x, y)] = mag > 255.f ? 255 : (uchar)mag;

			float deg = (float)((atan2(gX, gY) / M_PI) * 180.f);
			deg = deg < 0 ? deg + 180 : deg;
			int degX2 = (int)(deg * 2);
			float grad;
			if (degX2 <= 45 || degX2 > 315) grad = 0.f;
			else if (degX2 > 45 && degX2 <= 135) grad = 45.f;
			else if (degX2 > 135 && degX2 <= 225) grad = 90.f;
			else grad = 135.f;
			gradDataPtr[OFFSET(x, y)] = grad;
		}
	}
}
