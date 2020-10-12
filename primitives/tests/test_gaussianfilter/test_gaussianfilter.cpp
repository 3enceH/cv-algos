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
        filter.applyOnImage(black, blackOut);

        EXPECT_TRUE(std::equal(black.begin<uchar>(), black.end<uchar>(), blackOut.begin<uchar>()));

        cv::Mat grey(height, width, CV_8UC1, cv::Scalar(128));
        cv::Mat greyOut(height, width, CV_8UC1);
        filter.applyOnImage(grey, greyOut);

        EXPECT_TRUE(std::equal(grey.begin<uchar>(), grey.end<uchar>(), greyOut.begin<uchar>()));

        cv::Mat squares(height, width, CV_8UC1);
        cv::Mat squaresOut(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / 4 + y / 4) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }

        filter.applyOnImage(squares, squaresOut);
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

        cv::Mat squares1(height, width, CV_8UC1);
        cv::Mat squares2(height, width, CV_8UC1);

        GaussianFilter cpuFilter(k, sigma);
        cpuFilter.applyOnImage(squares, squares1);

        GaussianFilterCUDA gpuFilter(k, sigma);
        gpuFilter.applyOnImage(squares, squares2);

       
        int epsilon = 3;
        int diff = std::accumulate(squares1.begin<uchar>(), squares1.end<uchar>(), 0) - std::accumulate(squares2.begin<uchar>(), squares2.end<uchar>(), 0);
        float diffPerPix = diff / ((double)height * width);
        bool equals = std::equal(squares1.begin<uchar>(), squares1.end<uchar>(), squares2.begin<uchar>(), [=](uchar a, uchar b) { return a > b ? a - b < epsilon : b - a < epsilon; });
        EXPECT_TRUE(equals);
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
        gpuFilter1.applyOnImage(squares, out2);

        int epsilon = 3;
        int diff = std::accumulate(out1.begin<uchar>(), out1.end<uchar>(), 0) - std::accumulate(out2.begin<uchar>(), out2.end<uchar>(), 0);
        float diffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);
        int maxDiff = 0;
        bool equals = std::equal(out1.begin<uchar>(), out1.end<uchar>(), out2.begin<uchar>(), [&](uchar a, uchar b) {
            int diff = a > b ? a - b : b - a;
            if (diff > maxDiff) maxDiff = diff;
            return diff < epsilon;
            });
        EXPECT_TRUE(equals);
    }

    void picture(int k, float sigma = 1.f) {

        std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);
        cv::Mat out1, out2;

        cv::GaussianBlur(pic, out1, cv::Size(2 * k + 1, 2 * k + 1), sigma);

        GaussianFilterCUDA gpuFilter1(k, sigma);
        gpuFilter1.applyOnImage(pic, out2);

        int epsilon = 4;
        int diff = std::accumulate(out1.begin<uchar>(), out1.end<uchar>(), 0) - std::accumulate(out2.begin<uchar>(), out2.end<uchar>(), 0);
        float diffPerPix = diff / ((double)pic.cols * pic.rows);
        EXPECT_TRUE(diffPerPix < 1.f);
        int maxDiff = 0;
        bool equals = std::equal(out1.begin<uchar>(), out1.end<uchar>(), out2.begin<uchar>(), [&](uchar a, uchar b) {
            int diff = a > b ? a - b : b - a;
            if (diff > maxDiff) maxDiff = diff;
            return diff < epsilon;
        });
        EXPECT_TRUE(equals);
    }
};

TEST(GaussianFilterTest, Precomputed) {
    GaussianFilterAlgorithmicTest test;
    test(GaussianFilter(1), 4, 4);
    test(GaussianFilter(1), 4 * 4, 3 * 4);
    test(GaussianFilter(1, 1.f), 4 * 4, 3 * 4);
    test(GaussianFilter(2, 1.f), 4 * 4, 3 * 4);
    //test(GaussianFilter(3, 1.f), 256, 256);
    //test(GaussianFilter(5, 1.f), 1024, 1024);

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
    //test.picture(2, 1.f);
}