#include "sobelfilter.h"

constexpr float M_PI = 3.14159265358979323846f;

void SobelFilter::applyOnImage(cv::Mat& image, cv::Mat& gradients) {
	int width = image.cols;
	int height = image.rows;
	const uchar* const input = image.data;

	cv::Mat output = cv::Mat(width, height, image.type());
	cv::Mat _gradients = cv::Mat(width, height, CV_32FC1);

	auto offset = [=](int x, int y) { return y * width + x; };
	auto at = [=](int i, int j) { return j * 3 + i; };

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int gX = _kernelX[at(0, 0)] * input[offset(x - 1, y - 1)] + _kernelX[at(0, 1)] * input[offset(x - 1, y)] + _kernelX[at(0, 2)] * input[offset(x - 1, y + 1)] +
				_kernelX[at(1, 0)] * input[offset(x, y - 1)] + _kernelX[at(1, 1)] * input[offset(x, y)] + _kernelX[at(1, 2)] * input[offset(x + 1, y + 1)] +
				_kernelX[at(2, 0)] * input[offset(x + 1, y - 1)] + _kernelX[at(2, 1)] * input[offset(x + 1, y)] + _kernelX[at(2, 2)] * input[offset(x + 1, y + 1)];

			int gY = _kernelY[at(0, 0)] * input[offset(x - 1, y - 1)] + _kernelY[at(0, 1)] * input[offset(x - 1, y)] + _kernelY[at(0, 2)] * input[offset(x - 1, y + 1)] +
				_kernelY[at(1, 0)] * input[offset(x, y - 1)] + _kernelY[at(1, 1)] * input[offset(x, y)] + _kernelY[at(1, 2)] * input[offset(x + 1, y + 1)] +
				_kernelY[at(2, 0)] * input[offset(x + 1, y - 1)] + _kernelY[at(2, 1)] * input[offset(x + 1, y)] + _kernelY[at(2, 2)] * input[offset(x + 1, y + 1)];
			
			double mag = sqrt(gX * gX + gY * gY);
			output.data[offset(x, y)] = mag > 255.f ? 255 : (uchar)mag;

			float deg = (float)((atan2(gX, gY) / M_PI) * 180.f);
			deg = deg < 0 ? deg + 180 : deg;
			int degX2 = (int)(deg * 2);
			float grad;
			if (degX2 <= 45 || degX2 > 315) grad = 0.f;
			else if (degX2 > 45 && degX2 <= 135) grad = 45.f;
			else if (degX2 > 135 && degX2 <= 225) grad = 90.f;
			else grad = 135.f;
			_gradients.data[sizeof(float) * offset(x, y)] = grad;
		}
	}

	image = std::move(output);
	gradients = std::move(_gradients);
}
