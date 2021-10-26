#include "threshold.h"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void Threshold::apply(const cv::Mat& input, cv::Mat& output)
{
    int width = input.cols;
    int height = input.rows;
    const float* const inputData = (float*)input.data;

    if(output.empty())
    {
        output = cv::Mat(height, width, input.type());
    }
    float* const outputData = (float*)output.data;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            float value = inputData[OFFSET(x, y)];
            outputData[OFFSET(x, y)] = (value >= highThreshold)  ? highThreshold
                                       : (value >= lowThreshold) ? lowThreshold
                                                                 : 0.f;
        }
    }
}
