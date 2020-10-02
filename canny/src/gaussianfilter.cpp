#include "gaussianfilter.h"

constexpr float M_PI = 3.14159265358979323846f;

GaussianFilter::GaussianFilter(int k, float sigma) {
	assert(k > 0 && sigma >= 1.f);
	_size = 2 * k + 1;
	_kernel = std::vector<float>(_size);
	const float sqrSigmaX2 = 2 * sigma * sigma;
	const float coeff = (1 / sqrt(M_PI * sqrSigmaX2));
	for (int i = 0; i < _size; i++) {
		int arg = (i + 1 - (k + 1)) * (i + 1 - (k + 1));
		_kernel[i] = coeff * exp(-arg / sqrSigmaX2);
	}
}

void GaussianFilter::applyOnImage(cv::Mat& image, int times) {
	assert(_size > 0);
	const int width = image.cols;
	const int height = image.rows;
	const int k = _size / 2;
	uchar* const data = image.data;
	auto offset = [=](int x, int y) { return y * width + x; };

	for (int n = 0; n < times; n++) {
		// first, horizontal pass

#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float sum = 0;
				for (int i = -k; i <= k; i++) {
					int xx = x + i;
					if (xx < 0 || xx >= width) continue;
					sum += _kernel[(size_t)i + k] * data[offset(xx, y)];
				}
				uchar value = (uchar)sum;
				data[offset(x, y)] = value > 255 ? 255 : value;
			}
		}

		// second, vertical pass 
#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				float sum = 0;
				for (int i = -k; i <= k; i++) {
					int yy = y + i;
					if (yy < 0 || yy >= height) continue;
					sum += _kernel[(size_t)i + k] * data[offset(x, yy)];
				}
				uchar value = (uchar)sum;
				data[offset(x, y)] = value > 255 ? 255 : value;
			}
		}
	}
}
