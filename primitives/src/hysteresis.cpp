#include "hysteresis.h"

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void Hysteresis::apply(const cv::Mat& input, cv::Mat& output)
{
    int width = input.cols;
    int height = input.rows;
    const float* const inputData = (float*)input.data;

    if(output.empty())
    {
        output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    }
    uchar* const outputData = (uchar*)output.data;

    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            float value = inputData[OFFSET(x, y)];
            if(value == strongValue)
            {
                outputData[OFFSET(x, y)] = 255;
            }
            else if(value == weakValue)
            {
                if(inputData[OFFSET(x + 1, y + 0)] == strongValue || // E
                   inputData[OFFSET(x + 1, y + 1)] == strongValue || // SE
                   inputData[OFFSET(x + 0, y + 1)] == strongValue || // S
                   inputData[OFFSET(x - 1, y + 1)] == strongValue || // SW
                   inputData[OFFSET(x - 1, y + 0)] == strongValue || // W
                   inputData[OFFSET(x + 1, y - 1)] == strongValue || // NW
                   inputData[OFFSET(x + 0, y - 1)] == strongValue || // N
                   inputData[OFFSET(x - 1, y - 1)] == strongValue)   // NE
                {
                    outputData[OFFSET(x, y)] = 255;
                }
                else
                    outputData[OFFSET(x, y)] = 0;
            }
        }
    }
}
