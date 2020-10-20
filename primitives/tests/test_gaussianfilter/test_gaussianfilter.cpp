#include <numeric>

#include <gtest/gtest.h>
#include "gaussianfilter.h"
#include "cuda_gaussianfilter.cuh"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GaussianFilterAlgorithmicTest {
public:
    void operator()(GaussianFilter& filter, int width, int height) {
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

class GaussianFilterGPUComparisonTest {
public:
    void operator()(int width, int height, int k, float sigma = 1.f) {

        cv::Mat squares(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }

        cv::Mat out1(height, width, CV_8UC1);
        cv::Mat out2(height, width, CV_8UC1);

        GaussianFilter cpuFilter(k, sigma);
        cpuFilter.apply(squares, out1);

        GaussianFilterCUDA gpuFilter(k, sigma);
        gpuFilter.apply(squares, out2);

        cv::Mat diffImg;
        cv::absdiff(out1, out2, diffImg);

        double diffPerPix = cv::sum(diffImg)[0] / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        int maxDiff = 0;
        int epsilon = 3;
        bool equals = std::all_of(diffImg.begin<uchar>(), diffImg.end<uchar>(), [&](uchar val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }
};


class GaussianFilterOpenCVComparisonTest {
public:
    void generated(int width, int height, int k, float sigma = 1.f) {

        cv::Mat squares(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }

        cv::Mat out1(height, width, CV_8UC1);
        cv::Mat out2(height, width, CV_8UC1);

        cv::GaussianBlur(squares, out1, cv::Size(2 * k + 1, 2 * k + 1), sigma);

        GaussianFilterCUDA gpuFilter1(k, sigma);
        gpuFilter1.apply(squares, out2);

        cv::Mat diffImg;
        cv::absdiff(out1, out2, diffImg);

        double diffPerPix = cv::sum(diffImg)[0] / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 2.f);

        int maxDiff = 0;
        int epsilon = 3;
        bool equals = std::all_of(diffImg.begin<uchar>(), diffImg.end<uchar>(), [&](uchar val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }

    void picture(int k, float sigma = 1.f) {

        std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);
        cv::Mat out1, out2;

        cv::GaussianBlur(pic, out1, cv::Size(2 * k + 1, 2 * k + 1), sigma);

        GaussianFilterCUDA gpuFilter1(k, sigma);
        gpuFilter1.apply(pic, out2);

        cv::Mat diffImg;
        cv::absdiff(out1, out2, diffImg);

        int width = pic.cols;
        int height = pic.rows;

        double diffPerPix = cv::sum(diffImg)[0] / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        int maxDiff = 0;
        int epsilon = 30;
        bool equals = std::all_of(diffImg.begin<uchar>(), diffImg.end<uchar>(), [&](uchar val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }
};

TEST(GaussianFilterTest, Precomputed) {
    GaussianFilterAlgorithmicTest test;
    test(GaussianFilter(1), 4, 4);
    test(GaussianFilter(1), 4 * 4, 3 * 4);
    test(GaussianFilter(1, 1.f), 4 * 4, 3 * 4);
    test(GaussianFilter(2, 1.f), 4 * 4, 3 * 4);
}

TEST(GaussianFilterTest, CUDAComparison) {
    GaussianFilterGPUComparisonTest test;
    test(32, 32, 2, 1.f);
    test(10 * 4 * 32, 10 * 4 * 32, 4, 1.f);
    test(4096, 2160, 8, 1.f);
}

TEST(GaussianFilterTest, OpenCVComparison) {
    GaussianFilterOpenCVComparisonTest test;
    test.generated(32, 32, 2, 1.f);
    test.generated(10 * 4 * 32, 10 * 4 * 32, 4, 1.f);
    test.picture(2);
}