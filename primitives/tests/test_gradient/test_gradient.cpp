#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gradient.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GradientOpenCVComparisonTest {
public:
    void generated(int width, int height) {

        cv::Mat squares(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }
        cv::GaussianBlur(squares, squares, cv::Size(5, 5), 1.f);

        cv::Mat out1, out2, outX, outY;

        cv::Sobel(squares, outX, CV_32FC1, 1, 0);
        cv::Sobel(squares, outY, CV_32FC1, 0, 1);
        combine<float>(outX, outY, out1, [](float a, float b) -> float { return (float)hypot(a, b); });

        Gradient cpuFilter;
        cpuFilter.apply(squares, out2);

        cv::Mat mag, grad;
        splitChannels(out2, mag, grad);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);
        
        cv::Mat diffImg;
        cv::absdiff(out1, mag, diffImg);
        double diff = cv::sum(diffImg)[0];

        double diffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        float maxDiff = 0;
        float epsilon = 1.f;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }

    void picture() {

        std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);

        cv::GaussianBlur(pic, pic, cv::Size(5, 5), 1.f);

        cv::Mat out1, out2, outX, outY;

        cv::Sobel(pic, outX, CV_32FC1, 1, 0);
        cv::Sobel(pic, outY, CV_32FC1, 0, 1);
        combine<float>(outX, outY, out1, [](float a, float b) -> float { return (float)hypot(a, b); });

        Gradient cpuFilter;
        cpuFilter.apply(pic, out2);

        cv::Mat mag, grad;
        splitChannels(out2, mag, grad);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);

        int width = pic.cols;
        int height = pic.rows;

        cv::Mat diffImg;
        cv::absdiff(out1, mag, diffImg);
        double diff = cv::sum(diffImg)[0];

        double diffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        float maxDiff = 0;
        float epsilon = 52.f;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }
};

TEST(GradientTest, OpenCVComparison) {
    GradientOpenCVComparisonTest test;
    test.generated(32, 32);
    test.generated(10 * 4 * 32, 10 * 4 * 32);
    test.picture();
}