#include "sobelfilter.h"

#include "core.h"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)
#define AT(x, y) OFFSET_COL_MAJOR(x, y, 3, 1)

#define OFFSET_OUT(x, y) OFFSET_ROW_MAJOR(x, y, width, 2)

void SobelFilter::applyOnImage(const cv::Mat& input, cv::Mat& output) {
	int width = input.cols;
	int height = input.rows;
	const uchar* const inputData = input.data;

	if (output.empty()) {
		output = std::move(cv::Mat(height, width, CV_32FC2));
	}
	float* const gradientData = (float*)output.data;

	for (int y = 0; y < height; y++) {
		int yy = wrapMirror(y, height);
		for (int x = 0; x < width; x++) {
			int xx = wrapMirror(x, width);
			
			int gX = _kernelX[AT(0, 0)] * inputData[OFFSET(xx - 1, yy - 1)] + _kernelX[AT(0, 1)] * inputData[OFFSET(xx - 1, yy)] + _kernelX[AT(0, 2)] * inputData[OFFSET(xx - 1, yy + 1)] +
				_kernelX[AT(1, 0)] * inputData[OFFSET(xx, yy - 1)] + _kernelX[AT(1, 1)] * inputData[OFFSET(xx, yy)] + _kernelX[AT(1, 2)] * inputData[OFFSET(xx + 1, yy + 1)] +
				_kernelX[AT(2, 0)] * inputData[OFFSET(xx + 1, yy - 1)] + _kernelX[AT(2, 1)] * inputData[OFFSET(xx + 1, yy)] + _kernelX[AT(2, 2)] * inputData[OFFSET(xx + 1, yy + 1)];

			int gY = _kernelY[AT(0, 0)] * inputData[OFFSET(xx - 1, yy - 1)] + _kernelY[AT(0, 1)] * inputData[OFFSET(xx - 1, yy)] + _kernelY[AT(0, 2)] * inputData[OFFSET(xx - 1, yy + 1)] +
				_kernelY[AT(1, 0)] * inputData[OFFSET(xx, yy - 1)] + _kernelY[AT(1, 1)] * inputData[OFFSET(xx, yy)] + _kernelY[AT(1, 2)] * inputData[OFFSET(xx + 1, yy + 1)] +
				_kernelY[AT(2, 0)] * inputData[OFFSET(xx + 1, yy - 1)] + _kernelY[AT(2, 1)] * inputData[OFFSET(xx + 1, yy)] + _kernelY[AT(2, 2)] * inputData[OFFSET(xx + 1, yy + 1)];
			
			gradientData[OFFSET_OUT(x, y) + 0] = (float)std::hypot(gX, gY);
			gradientData[OFFSET_OUT(x, y) + 1] = (float)atan2(gX, gY);
		}
	}
}
