#include <string>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "primitives.h"
#include "debug.h"

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

std::unique_ptr<GaussianFilterCUDA> gaussFilter;
std::unique_ptr<SobelFilter> sobelFilter;
std::unique_ptr<NonMaxSupress> nonMaxSupressor;
std::unique_ptr<DoubleThreshold> doubleThresholder;
std::unique_ptr<ConnectedComponentLabeling> conComLab;

cv::Mat colorFrame;
cv::Mat greyFrameInput;
cv::Mat greyFrameOutput;

void image();
void videoPlay(int skipFrames = 0);
void calculate();

int main(int arc, char** argc)
{
    try {
        gaussFilter = std::make_unique<GaussianFilterCUDA>(7, 1.5f);
        sobelFilter = std::make_unique<SobelFilter>();
        nonMaxSupressor = std::make_unique<NonMaxSupress>();
        doubleThresholder = std::make_unique<DoubleThreshold>(100.f, 50.f);
        conComLab = std::make_unique<ConnectedComponentLabeling>();

        image();

        cv::waitKey(0);

        conComLab.reset();
        doubleThresholder.reset();
        nonMaxSupressor.reset();
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
    greyFrameInput = cv::Mat(width, height, CV_8UC1);

    cv::cvtColor(colorFrame, greyFrameInput, cv::COLOR_RGB2GRAY);

    cv::namedWindow("w", 1);
    /*for (;;)*/ {
        calculate();
        cv::imshow("w", greyFrameInput);

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
    greyFrameInput = cv::Mat(width, height, CV_8UC1);
    
    for(int i = 0; i < skipFrames - 1; i++)  capture >> colorFrame;

    cv::namedWindow("w", 1);
    for (;;)
    {
        capture >> colorFrame;
        if (colorFrame.empty())
            break;

        cv::cvtColor(colorFrame, greyFrameInput, cv::COLOR_RGB2GRAY);

        calculate();

        cv::imshow("w", greyFrameInput);

        cv::waitKey(20);
    }
}

void calculate() {

    PerformanceTimer timer;
    timer.start();

    cv::Mat blurred;
    gaussFilter->applyOnImage(greyFrameInput, blurred);
    timer.tag("gauss");

    cv::Mat gradients;
    sobelFilter->applyOnImage(blurred, gradients);
    timer.tag("sobel");

    cv::Mat gradientsFiltered;
    nonMaxSupressor->applyOnImage(gradients, gradientsFiltered);
    timer.tag("nonmax");

    cv::Mat thresholded;
    doubleThresholder->applyOnImage(gradientsFiltered, thresholded);
    timer.tag("threshold");

    cv::Mat labels;
    conComLab->applyOnImage(thresholded, labels);
    timer.tag("CCL");
    std::cout << timer.summary() << std::endl;

    cv::Mat colored;
    labelsToColored(labels, colored);

    printf("");
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    //std::cout  << time << " ms" << std::endl;
}

