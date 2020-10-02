#include <string>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gaussianfilter.h"
#include "cuda_gaussianfilter.cuh"

#include "sobelfilter.h"

template<class T>
std::ostream& operator <<(std::ostream& s, const std::vector<T>& v) {
    s << "(";
    for (unsigned int i = 0; i < v.size(); i++) {
        s << v[i];
        if (i != v.size() - 1)  s << ",";
    }
    s << ")";
    return s;
}

std::unique_ptr<GaussianFilter> gaussFilter;
std::unique_ptr<SobelFilter> sobelFilter;

cv::Mat colorFrame;
cv::Mat greyFrame;

void image();
void videoPlay(int skipFrames = 0);
void calculate();

int main(int arc, char** argc)
{
    try {
        gaussFilter.reset(new GaussianFilterCUDA(5, 1.f));
        sobelFilter.reset(new SobelFilter);

        image();

        cv::waitKey(0);

        sobelFilter.reset();
        gaussFilter.reset();
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void image() {
    std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";

    colorFrame = cv::imread(filename);

    int width = colorFrame.cols;
    int height = colorFrame.rows;
    greyFrame = cv::Mat(width, height, CV_8UC1);

    cv::cvtColor(colorFrame, greyFrame, cv::COLOR_RGB2GRAY);

    cv::namedWindow("w", 1);
    /*for (;;)*/ {
        calculate();
        cv::imshow("w", greyFrame);

        cv::waitKey(20);
    }
}

void videoPlay(int skipFrames) {
    std::string filename = "d:\\Downloads\\The.Big.Short.2015.BDRiP.x264.HuN-HyperX\\thebigshort-sd-hyperx.mkv";
    
    cv::VideoCapture capture(filename);
    assert(capture.isOpened());
    capture >> colorFrame;
    
    int width = colorFrame.cols;
    int height = colorFrame.rows;
    greyFrame = cv::Mat(width, height, CV_8UC1);
    
    for(int i = 0; i < skipFrames - 1; i++)  capture >> colorFrame;

    cv::namedWindow("w", 1);
    for (;;)
    {
        capture >> colorFrame;
        if (colorFrame.empty())
            break;

        cv::cvtColor(colorFrame, greyFrame, cv::COLOR_RGB2GRAY);

        calculate();

        cv::imshow("w", greyFrame);

        cv::waitKey(20);
    }
}

void calculate() {
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gaussFilter->applyOnImage(greyFrame, 3);
    cv::Mat gradients;
    sobelFilter->applyOnImage(greyFrame, gradients);
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    //std::cout  << time << " ms" << std::endl;
}