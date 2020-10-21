#include "nonmaxsupress.h"
#include <iostream>

#define OFFSET_IN(x, y) OFFSET_ROW_MAJOR(x, y, width, 2)
#define OFFSET_OUT(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void NonMaxSupress::apply(const cv::Mat& input, cv::Mat& output) {
	int width = input.cols;
	int height = input.rows;
	const float* const inputData = (float*)input.data;

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, CV_32FC1, cv::Scalar(0)));
	}
	float* const outputData = (float*)output.data;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			bool flag = false;
			float mag = inputData[OFFSET_IN(x, y) + 0];
			float rad = inputData[OFFSET_IN(x, y) + 1];
			float deg = (rad * 180.f) / M_PI;
			deg = deg < 0 ? deg + 180.f : deg;

			// east
			if (deg <= 22.5f || deg > 157.5f) {
				flag = mag >= inputData[OFFSET_IN(x - 1, y)] && mag >= inputData[OFFSET_IN(x + 1, y)];
			}
			// north-east
			else if (deg > 22.5f && deg <= 67.5f) {
				flag = mag >= inputData[OFFSET_IN(x + 1, y - 1)] && mag >= inputData[OFFSET_IN(x - 1, y + 1)];
			}
			// north
			else if (deg > 67.5f && deg <= 112.5f) {
				flag = mag >= inputData[OFFSET_IN(x, y + 1)] && mag >= inputData[OFFSET_IN(x, y - 1)];
			}
			// north-west
			else if (deg > 112.5f && deg <= 157.5f) {
				flag = mag >= inputData[OFFSET_IN(x - 1, y - 1)] && mag >= inputData[OFFSET_IN(x + 1, y + 1)];
			}
			else assert(false);

			if (flag)
				outputData[OFFSET_OUT(x, y)] = mag;
		}
	}
}
