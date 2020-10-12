#include "nonmaxsupress.h"

#define OFFSET_IN(x, y) OFFSET_ROW_MAJOR(x, y, width, 2)
#define OFFSET_OUT(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void NonMaxSupress::applyOnImage(const cv::Mat& input, cv::Mat& output) {
	int width = input.cols;
	int height = input.rows;
	const float* const inputData = (float*)input.data;

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, CV_32FC1));
	}
	float* const outputData = (float*)output.data;

	for (int y = 0; y < height; y++) {
		int yy = wrapMirror(y, height);
		for (int x = 0; x < width; x++) {
			int xx = wrapMirror(x, width);

			bool flag = false;
			float mag = inputData[OFFSET_IN(xx, yy) + 0];
			float deg = inputData[OFFSET_IN(xx, yy) + 1];
			deg = deg < 0 ? deg + 180.f : deg;
			int degX2 = (int)(deg * 2);
			// east
			if (degX2 <= 45 || degX2 > 315) {
				flag = mag > std::hypot(inputData[OFFSET_IN(xx - 1, yy)], inputData[OFFSET_IN(xx + 1, yy)]);
			}
			// north-east
			else if (degX2 > 45 && degX2 <= 135) {
				flag = mag > std::hypot(inputData[OFFSET_IN(xx - 1, yy + 1)], inputData[OFFSET_IN(xx + 1, yy - 1)]);
			}
			// north
			else if (degX2 > 135 && degX2 <= 225) {
				flag = mag > std::hypot(inputData[OFFSET_IN(xx, yy + 1)], inputData[OFFSET_IN(xx, yy - 1)]);
			}
			// north-west
			else {
				flag = mag > std::hypot(inputData[OFFSET_IN(xx - 1, yy - 1)], inputData[OFFSET_IN(xx + 1, yy + 1)]);
			}

			outputData[OFFSET_OUT(x, y)] = flag ? mag : 0.f;
		}
	}
}
