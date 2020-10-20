#include "nonmaxsupress.h"
#include <iostream>

#define OFFSET_IN(x, y) OFFSET_ROW_MAJOR(x, y, width, 2)
#define OFFSET_OUT(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void NonMaxSupress::apply(const cv::Mat& input, cv::Mat& output) {
	int width = input.cols;
	int height = input.rows;
	const float* const inputData = (float*)input.data;

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, CV_32FC1));
	}
	float* const outputData = (float*)output.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			bool flag = false;
			float mag = inputData[OFFSET_IN(x, y) + 0];
			float rad = inputData[OFFSET_IN(x, y) + 1];
			float deg = (rad / M_PI) * 180.f;
			deg = deg < 0 ? deg + 180.f : deg;
			int degX2 = (int)(deg * 2);
			int xp = BORDER_MIRROR(x + 1, width);
			int xm = BORDER_MIRROR(x - 1, width);
			int yp = BORDER_MIRROR(y + 1, height);
			int ym = BORDER_MIRROR(y - 1, height);

			// east
			if (degX2 <= 45 || degX2 > 315) {
				std::cout << rad << " " << deg << " E";
				flag = mag > inputData[OFFSET_IN(xm, y)] && mag > inputData[OFFSET_IN(xp, y)];
			}
			// north-east
			else if (degX2 > 45 && degX2 <= 135) {
				std::cout << rad << " " << deg << " NE";
				flag = mag > inputData[OFFSET_IN(xm, yp)] && mag > inputData[OFFSET_IN(xp, ym)];
			}
			// north
			else if (degX2 > 135 && degX2 <= 225) {
				std::cout << rad << " " << deg << " N";
				flag = mag > inputData[OFFSET_IN(x, yp)] && mag > inputData[OFFSET_IN(x, ym)];
			}
			// north-west
			else {
				std::cout << rad << " " << deg << " NW";
				flag = mag > inputData[OFFSET_IN(xm, ym)] && mag > inputData[OFFSET_IN(xp, yp)];
			}
			std::cout << " " << flag << " val " << (flag ? mag : 0.f) << std::endl;

			outputData[OFFSET_OUT(x, y)] = flag ? mag : 0.f;
		}
	}
}
