#include <numeric>

#include "cuda_gaussian.cuh"
#include "debug.h"
#include "gaussian.h"
#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

cv::Mat getSquares(int width, int height)
{
    cv::Mat squares(height, width, CV_8UC1);
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
            squares.data[y * width + x] = value;
        }
    }
    return std::move(squares);
}

void check(cv::Mat& out1, cv::Mat& out2)
{
    cv::Mat diffImg;
    cv::absdiff(out1, out2, diffImg);

    double avgDiffPerPix = cv::sum(diffImg)[0] / ((double)out1.cols * out1.rows);
    EXPECT_TRUE(avgDiffPerPix < 1.1f);

    int maxDiff = 0;
    int epsilon = 3;
    bool equals = std::all_of(diffImg.begin<uchar>(), diffImg.end<uchar>(), [&](uchar val) {
        if(val > maxDiff)
        {
            maxDiff = val;
        }
        return val < epsilon;
    });
    EXPECT_TRUE(equals);
    std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiff "
              << std::setw(5) << maxDiff << std::endl;
}

class GaussianFilterAlgorithmicTest
{
public:
    void operator()(Gaussian filter, int width, int height)
    {
        // kernel initialized correctly ?
        auto& kernel = filter.kernel();

        EXPECT_TRUE(!kernel.empty());
        EXPECT_TRUE(kernel.size() % 2 == 1);

        float sum = std::accumulate(kernel.begin(), kernel.end(), 0.f);

        EXPECT_TRUE(std::abs(1.f - sum) <= FLT_EPSILON);

        cv::Mat black(height, width, CV_8UC1, cv::Scalar(0));
        cv::Mat blackOut(height, width, CV_8UC1);
        filter.apply(black, blackOut);

        EXPECT_TRUE(std::equal(black.begin<uchar>(), black.end<uchar>(), blackOut.begin<uchar>()));

        cv::Mat grey(height, width, CV_8UC1, cv::Scalar(128));
        cv::Mat greyOut(height, width, CV_8UC1);
        filter.apply(grey, greyOut);

        EXPECT_TRUE(std::equal(grey.begin<uchar>(), grey.end<uchar>(), greyOut.begin<uchar>()));
    }
};

class GaussianFilterCUDAComparisonTest
{
public:
    void operator()(int width, int height, int size, float sigma = 1.f)
    {
        std::cout << "generated dims " << width << "x" << height << " size " << size << " sigma " << sigma << std::endl;

        cv::Mat squares = getSquares(width, height);

        cv::Mat out1(height, width, CV_8UC1);
        cv::Mat out2(height, width, CV_8UC1);

        Gaussian cpuFilter(size, sigma);
        GaussianCUDA gpuFilter(size, sigma);

        PerformanceTimer perfTimer;
        perfTimer.start();

        cpuFilter.apply(squares, out1);
        perfTimer.tag("cpu");

        gpuFilter.apply(squares, out2);
        perfTimer.tag("gpu");

        std::cout << perfTimer.summary();

        check(out1, out2);
    }
};

class GaussianFilterOpenCVComparisonTest
{
public:
    void generated(int width, int height, int size, float sigma = 1.f)
    {
        std::cout << "generated dims " << width << "x" << height << " size " << size << " sigma " << sigma << std::endl;

        cv::Mat squares = getSquares(width, height);

        cv::Mat out1(height, width, CV_8UC1);
        cv::Mat out2(height, width, CV_8UC1);

        GaussianCUDA gpuFilter1(size, sigma);

        PerformanceTimer perfTimer;
        perfTimer.start();

        cv::GaussianBlur(squares, out1, cv::Size(size, size), sigma);
        perfTimer.tag("opencv");

        gpuFilter1.apply(squares, out2);
        perfTimer.tag("gpu");

        std::cout << perfTimer.summary();

        check(out1, out2);
    }

    void picture(const std::string& name, int size, float sigma = 1.f)
    {

        std::string filename = std::string(STR(DATA_ROOT)) + "/" + name;
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);

        int width = pic.cols;
        int height = pic.rows;
        std::cout << "pic " << filename << " dims " << width << "x" << height << " size " << size << " sigma " << sigma
                  << std::endl;

        cv::Mat out1, out2;

        GaussianCUDA gpuFilter1(size, sigma);

        PerformanceTimer perfTimer;
        perfTimer.start();

        cv::GaussianBlur(pic, out1, cv::Size(size, size), sigma);
        perfTimer.tag("opencv");

        gpuFilter1.apply(pic, out2);
        perfTimer.tag("gpu");

        std::cout << perfTimer.summary();

        check(out1, out2);
    }
};

TEST(GaussianFilterTest, Precomputed)
{
    GaussianFilterAlgorithmicTest test;
    test(Gaussian(3, 1.f), 32, 32);
    test(Gaussian(5, 3.f), 32, 32);
    test(Gaussian(7, 5.f), 32, 32);
}

TEST(GaussianFilterTest, CUDAComparison)
{
    GaussianFilterCUDAComparisonTest test;
    test(32, 32, 5, 1.f);
    test(10 * 4 * 32, 10 * 4 * 32, 5, 1.f);
    test(1980, 1280, 10, 1.f);
    test(4096, 2160, 10, 1.f);
}

TEST(GaussianFilterTest, OpenCVComparison)
{
    GaussianFilterOpenCVComparisonTest test;
    test.generated(32, 32, 5, 1.f);
    test.generated(10 * 4 * 32, 10 * 4 * 32, 10, 1.f);
    test.picture("samuraijack.jpg", 5, 1.f);
    test.picture("rihanna.jpg", 10, 1.f);
}