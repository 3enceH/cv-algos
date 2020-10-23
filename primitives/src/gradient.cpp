#include "gradient.h"

#include "core.h"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)
#define OFFSET_OUT(x, y) OFFSET_ROW_MAJOR(x, y, width, 2)
#define BORDER_WRAP BORDER_REFLECT_101

#include <opencv2/imgproc.hpp>

void Gradient::apply(const cv::Mat& input, cv::Mat& output) {

    // Sobel operators derivative 1
    const int kX[3] = { -1, 0, 1 };
    const int kY[3] = { 1, 2, 1 };

    int width = input.cols;
    int height = input.rows;
    const uchar* const inputData = input.data;

    if (output.empty()) {
        output = std::move(cv::Mat(height, width, CV_32FC2));
    }

    if (bufferX.empty() || bufferY.empty() || bufferX.cols != width || bufferX.rows != height) {
        bufferX = cv::Mat(height, width, CV_32SC1);
        bufferY = cv::Mat(height, width, CV_32SC1);
    }

    float* const gradientData = (float*)output.data;
    int* hX = (int*)bufferX.data;
    int* hY = (int*)bufferY.data;

    // horizontal pass
#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int leftIdx = OFFSET(BORDER_WRAP(x - 1, width), y);
            int midIdx = OFFSET(x, y);
            int rightIdx = OFFSET(BORDER_WRAP(x + 1, width), y);

            hX[OFFSET(x, y)] = kX[0] * inputData[leftIdx] + kX[1] * inputData[midIdx] + kX[2] * inputData[rightIdx];
            hY[OFFSET(x, y)] = kY[0] * inputData[leftIdx] + kY[1] * inputData[midIdx] + kY[2] * inputData[rightIdx];
        }
    }

    //cv::Mat sobelX(height, width, CV_32FC1);
    //cv::Mat sobelY(height, width, CV_32FC1);
    //cv::Mat cvSobelX = sobelX.clone();
    //cv::Mat cvSobelY = sobelY.clone();

    // vertical pass
#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#endif
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int upIdx = OFFSET(x, BORDER_WRAP(y - 1, height));
            int midIdx = OFFSET(x, y);
            int downIdx = OFFSET(x, BORDER_WRAP(y + 1, height));

            int gX = kY[0] * hX[upIdx] + kY[1] * hX[midIdx] + kY[2] * hX[downIdx];
            int gY = kX[0] * hY[upIdx] + kX[1] * hY[midIdx] + kX[2] * hY[downIdx];

            //((float*)sobelX.data)[OFFSET(x, y)] = (float)gX;
            //((float*)sobelY.data)[OFFSET(x, y)] = (float)gY;

            gradientData[OFFSET_OUT(x, y) + 0] = (float)hypot(gX, gY);
            gradientData[OFFSET_OUT(x, y) + 1] = (float)atan2(gY, gX);
        }
    }

    //cv::Sobel(input, cvSobelX, CV_32FC1, 1, 0);
    //cv::Sobel(input, cvSobelY, CV_32FC1, 0, 1);

    //cv::Mat diffX, diffY;
    //cv::absdiff(sobelX, cvSobelX, diffX);
    //cv::absdiff(sobelY, cvSobelY, diffY);
    //double diff = cv::sum(diffX)[0];
    //assert(diff == 0);
    //diff = cv::sum(diffY)[0];
    //assert(diff == 0);
}
